/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

#include <stdio.h>

#include "util/bitset.h"

#define VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT 0x8u

struct lower_rt_io_state {
   const struct vt_nir_rt_config *config;
   uint32_t trace_depth;
};

struct lower_global_root_continuation_state {
   const struct vt_nir_rt_config *config;
   struct set *lowered_traces;
};

/* Each TraceRay payload is a linked location range within one P-byte ABI
 * frame.  The frame number is the static bounce depth selected while
 * callbacks are inlined. */
struct trace_payload_binding {
   uint32_t slot_offset;
   uint32_t size_bytes;
};

static uint32_t
payload_frame_size(const struct vt_nir_scratchpad_layout *sp)
{
   return sp->total_size_bytes - sp->payload_spill_size_bytes -
          sp->payload_base_bytes;
}

static bool
trace_payload_binding(const nir_intrinsic_instr *intrin,
                      const struct vt_nir_rt_config *config,
                      struct trace_payload_binding *binding)
{
   if (!intrin || !config || !binding)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intrin->src[10]);
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   if (!var ||
       (var->data.mode != nir_var_shader_temp &&
        var->data.mode != nir_var_shader_call_data) ||
       var->data.location < 0)
      return false;

   unsigned size = 0;
   unsigned align = 0;
   glsl_get_natural_size_align_bytes(var->type, &size, &align);
   if (!size)
      return false;

   const struct vt_nir_scratchpad_layout *sp =
      &config->scratchpad[VT_RT_LAYOUT_RAYGEN];
   const uint32_t payload_size = payload_frame_size(sp);
   const uint64_t location_offset =
      (uint64_t)(uint32_t)var->data.location * VT_RT_ABI_ALIGNMENT_BYTES;
   const uint64_t end = location_offset + ALIGN_POT(size, VT_RT_ABI_WORD_BYTES);
   if (end > payload_size)
      return false;

   binding->slot_offset = sp->payload_base_bytes + (uint32_t)location_offset;
   binding->size_bytes = ALIGN_POT(size, VT_RT_ABI_WORD_BYTES);
   return true;
}

static nir_def *
trace_payload_frame_addr(nir_builder *b,
                         const struct vt_nir_rt_config *config,
                         const struct trace_payload_binding *binding,
                         uint32_t depth, uint32_t offset)
{
   const struct vt_nir_scratchpad_layout *sp =
      &config->scratchpad[VT_RT_LAYOUT_RAYGEN];
   const uint64_t frame_offset =
      (uint64_t)binding->slot_offset +
      (uint64_t)depth * payload_frame_size(sp) + offset;
   assert(frame_offset <= UINT32_MAX);
   return vt_nir_scratchpad_addr(b, config, VT_RT_LAYOUT_RAYGEN,
                                 (uint32_t)frame_offset);
}

static void
handoff_trace_payload_frame(nir_builder *b,
                            const struct vt_nir_rt_config *config,
                            const struct trace_payload_binding *binding,
                            uint32_t source_depth, uint32_t destination_depth)
{
   for (uint32_t offset = 0; offset < binding->size_bytes;
        offset += VT_RT_ABI_WORD_BYTES) {
      nir_def *value = nir_ventus_rt_load_slot(
         b, 1, 32, trace_payload_frame_addr(b, config, binding,
                                             source_depth, offset));
      nir_ventus_rt_store_slot(
         b, value, trace_payload_frame_addr(b, config, binding,
                                             destination_depth, offset));
   }
}

/* A nested monolithic trace overwrites RTcore's sole hardware-visible PDS
 * context.  Preserve only parent words read after the child returns.  The
 * saved values are NIR SSA defs, not another PDS frame: a later nested trace
 * gets its own defs at its own call site and the register allocator decides
 * whether any value must spill. */
enum pds_context_addr {
   PDS_CONTEXT_ADDR_WORD,
   PDS_CONTEXT_ADDR_OUTSIDE,
   PDS_CONTEXT_ADDR_UNKNOWN,
};

static bool
const_u32(nir_def *def, uint32_t *value)
{
   nir_src src = nir_src_for_ssa(def);
   if (!nir_src_is_const(src))
      return false;

   *value = nir_src_as_uint(src);
   return true;
}

/* The ABI generates payload addresses as
 *   payload_base + word * 4 + active_level * payload_size.
 * The dynamic addend cannot alias the fixed RTcore context.  Other dynamic
 * PDS addresses are conservatively treated as aliases of every context word.
 */
static enum pds_context_addr
classify_pds_context_addr(nir_def *addr, uint32_t context_size,
                          uint32_t *word)
{
   uint32_t value;
   if (const_u32(addr, &value)) {
      if (value >= context_size || value % VT_RT_ABI_WORD_BYTES)
         return PDS_CONTEXT_ADDR_OUTSIDE;
      *word = value / VT_RT_ABI_WORD_BYTES;
      return PDS_CONTEXT_ADDR_WORD;
   }

   nir_instr *instr = nir_def_instr(addr);
   if (instr->type != nir_instr_type_alu)
      return PDS_CONTEXT_ADDR_UNKNOWN;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_iadd)
      return PDS_CONTEXT_ADDR_UNKNOWN;

   for (unsigned i = 0; i < 2; i++) {
      if (!const_u32(alu->src[i].src.ssa, &value))
         continue;
      if (value >= context_size)
         return PDS_CONTEXT_ADDR_OUTSIDE;
   }

   return PDS_CONTEXT_ADDR_UNKNOWN;
}

static void
transfer_pds_context_liveness(const nir_instr *instr, BITSET_WORD *live,
                              uint32_t context_words)
{
   if (instr->type != nir_instr_type_intrinsic)
      return;

   const nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   const bool load = intrin->intrinsic == nir_intrinsic_ventus_rt_load_slot;
   const bool store = intrin->intrinsic == nir_intrinsic_ventus_rt_store_slot;
   if (!load && !store)
      return;

   const unsigned addr_src = load ? 0 : 1;
   uint32_t word = 0;
   switch (classify_pds_context_addr(intrin->src[addr_src].ssa,
                                     context_words * VT_RT_ABI_WORD_BYTES,
                                     &word)) {
   case PDS_CONTEXT_ADDR_WORD:
      if (load)
         BITSET_SET(live, word);
      else
         BITSET_CLEAR(live, word);
      break;
   case PDS_CONTEXT_ADDR_UNKNOWN:
      /* An indirect PDS access may read or overwrite any ABI word. */
      for (uint32_t i = 0; i < context_words; i++)
         BITSET_SET(live, i);
      break;
   case PDS_CONTEXT_ADDR_OUTSIDE:
      break;
   }
}

static void
union_successor_liveness(nir_block *block, const BITSET_WORD *live_in,
                         unsigned bitset_words, BITSET_WORD *live)
{
   memset(live, 0, bitset_words * sizeof(*live));
   for (unsigned i = 0; i < ARRAY_SIZE(block->successors); i++) {
      nir_block *succ = block->successors[i];
      if (!succ)
         continue;
      const BITSET_WORD *succ_live = live_in + succ->index * bitset_words;
      for (unsigned j = 0; j < bitset_words; j++)
         live[j] |= succ_live[j];
   }
}

/* Return the PDS words live immediately after trace.  Backward CFG dataflow
 * makes branch paths union naturally and reaches a fixed point for loops. */
static BITSET_WORD *
nested_parent_live_words(nir_function_impl *impl, nir_intrinsic_instr *trace,
                         const struct vt_nir_rt_config *config)
{
   nir_index_blocks(impl);
   const uint32_t context_words =
      config->scratchpad[VT_RT_LAYOUT_RAYGEN].payload_base_bytes /
      VT_RT_ABI_WORD_BYTES;
   const unsigned bitset_words = BITSET_WORDS(context_words);
   /* end_block has index num_blocks and is a successor of exit blocks. */
   BITSET_WORD *live_in = calloc((impl->num_blocks + 1) * bitset_words,
                                 sizeof(*live_in));
   BITSET_WORD *live = calloc(bitset_words, sizeof(*live));
   BITSET_WORD *next = calloc(bitset_words, sizeof(*next));
   BITSET_WORD *result = calloc(bitset_words, sizeof(*result));
   assert(live_in && live && next && result);

   bool progress;
   do {
      progress = false;
      nir_foreach_block_reverse(block, impl) {
         union_successor_liveness(block, live_in, bitset_words, next);
         nir_foreach_instr_reverse(instr, block)
            transfer_pds_context_liveness(instr, next, context_words);

         BITSET_WORD *block_live = live_in + block->index * bitset_words;
         if (memcmp(block_live, next, bitset_words * sizeof(*next))) {
            memcpy(block_live, next, bitset_words * sizeof(*next));
            progress = true;
         }
      }
   } while (progress);

   nir_block *trace_block = trace->instr.block;
   union_successor_liveness(trace_block, live_in, bitset_words, live);
   bool found_trace = false;
   nir_foreach_instr_reverse(instr, trace_block) {
      if (instr == &trace->instr) {
         memcpy(result, live, bitset_words * sizeof(*result));
         found_trace = true;
         break;
      }
      transfer_pds_context_liveness(instr, live, context_words);
   }
   assert(found_trace);

   if (getenv("VT_NIR_DEBUG_PDS_LIVENESS_ALL")) {
      for (uint32_t word = 0; word < context_words; word++)
         BITSET_SET(result, word);
   }

   free(next);
   free(live);
   free(live_in);
   return result;
}

static nir_def **
save_nested_parent_context(nir_builder *b, const struct vt_nir_rt_config *config,
                           const BITSET_WORD *live)
{
   const uint32_t context_words =
      config->scratchpad[VT_RT_LAYOUT_RAYGEN].payload_base_bytes /
      VT_RT_ABI_WORD_BYTES;
   nir_def **saved = calloc(context_words, sizeof(*saved));
   assert(saved);

   for (uint32_t word = 0; word < context_words; word++) {
      if (!BITSET_TEST(live, word))
         continue;
      saved[word] = nir_ventus_rt_load_slot(
         b, 1, 32, vt_nir_scratchpad_addr(
            b, config, VT_RT_LAYOUT_RAYGEN, word * VT_RT_ABI_WORD_BYTES));
   }

   return saved;
}

static void
debug_nested_parent_live_words(const struct vt_nir_rt_config *config,
                               const BITSET_WORD *live)
{
   if (!getenv("VT_NIR_DEBUG_PDS_LIVENESS"))
      return;

   const uint32_t context_words =
      config->scratchpad[VT_RT_LAYOUT_RAYGEN].payload_base_bytes /
      VT_RT_ABI_WORD_BYTES;
   fprintf(stderr, "ventus: nested parent PDS live words:");
   for (uint32_t word = 0; word < context_words; word++) {
      if (BITSET_TEST(live, word))
         fprintf(stderr, " %u", word);
   }
   fputc('\n', stderr);
}

static void
restore_nested_parent_context(nir_builder *b, const struct vt_nir_rt_config *config,
                              nir_def **saved)
{
   const uint32_t context_words =
      config->scratchpad[VT_RT_LAYOUT_RAYGEN].payload_base_bytes /
      VT_RT_ABI_WORD_BYTES;
   for (uint32_t word = 0; word < context_words; word++) {
      if (!saved[word])
         continue;
      nir_ventus_rt_store_slot(
         b, saved[word], vt_nir_scratchpad_addr(
            b, config, VT_RT_LAYOUT_RAYGEN, word * VT_RT_ABI_WORD_BYTES));
   }
}

static nir_def *
load_dispatch_arg(nir_builder *b, uint32_t offset, unsigned bit_size)
{
   nir_intrinsic_instr *intrin =
      nir_intrinsic_instr_create(b->shader, nir_intrinsic_ventus_rt_dispatch_arg);
   intrin->src[0] = nir_src_for_ssa(nir_imm_int(b, offset));

   nir_def_init(&intrin->instr, &intrin->def, 1, bit_size);
   nir_builder_instr_insert(b, &intrin->instr);
   return &intrin->def;
}

/* Push constants are captured by the bridge in the immutable dispatch
 * snapshot.  The megakernel must replace the logical Vulkan load before
 * LLVM lowering: the NIR-to-LLVM path deliberately has no generic
 * load_push_constant fallback, and would otherwise turn its address into
 * undef. */
static bool
lower_monolithic_push_constants(nir_builder *b, nir_intrinsic_instr *intrin,
                                void *data)
{
   (void)data;
   if (intrin->intrinsic != nir_intrinsic_load_push_constant)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *base = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_PUSH_CONSTANTS_ADDR, 64);
   nir_def *offset = nir_iadd_imm(b, nir_u2u64(b, intrin->src[0].ssa),
                                  nir_intrinsic_base(intrin));
   nir_def_rewrite_uses(
      &intrin->def,
      nir_load_global(b, intrin->def.num_components, intrin->def.bit_size,
                      nir_iadd(b, base, offset)));
   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
trace_ray_has_const_flag(nir_intrinsic_instr *intrin, uint32_t flag)
{
   return nir_src_is_const(intrin->src[1]) &&
          (nir_src_as_uint(intrin->src[1]) & flag) != 0;
}

static nir_def *
load_raygen_callback_decision(nir_builder *b,
                              const struct vt_nir_rt_config *config)
{
   return vt_nir_load_control_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                   VT_RT_CONTROL_CALLBACK_DECISION, 32);
}

static void
store_raygen_callback_decision(nir_builder *b,
                               const struct vt_nir_rt_config *config,
                               enum vt_rt_callback_decision decision)
{
   vt_nir_store_control_word(b, config, VT_RT_LAYOUT_RAYGEN,
                             VT_RT_CONTROL_CALLBACK_DECISION,
                             nir_imm_int(b, decision));
}

static void
accept_candidate_hit(nir_builder *b, const struct vt_nir_rt_config *config)
{
   store_raygen_callback_decision(b, config, VT_RT_CALLBACK_ACCEPT);
}

static void
ignore_candidate_hit(nir_builder *b, const struct vt_nir_rt_config *config)
{
   store_raygen_callback_decision(b, config, VT_RT_CALLBACK_IGNORE);
}

static void
emit_trace_callbacks(nir_builder *b,
                     const struct vt_nir_rt_config *config,
                     const struct vt_nir_rt_trace_params *params,
                     const struct vt_nir_rt_result *result,
                     nir_def *complete_hit,
                     nir_def *complete_miss,
                     nir_def *candidate_non_opaque,
                     nir_def *candidate_procedural,
                     bool skip_closest_hit)
{
   const bool callbacks_enabled =
      config && config->enable_monolithic_shader_callbacks;
   const bool has_callbacks =
      callbacks_enabled &&
      (config->emit_closest_hit || config->emit_miss ||
       config->emit_any_hit || config->emit_intersection);

   if (!has_callbacks) {
      nir_push_if(b, candidate_non_opaque);
      accept_candidate_hit(b, config);
      nir_push_else(b, NULL);
      nir_push_if(b, candidate_procedural);
      ignore_candidate_hit(b, config);
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
      return;
   }

   nir_push_if(b, complete_hit);
   if (!skip_closest_hit && config->emit_closest_hit)
      config->emit_closest_hit(b, params, result, config->callback_data);

   nir_push_else(b, NULL);
   nir_push_if(b, complete_miss);
   if (config->emit_miss)
      config->emit_miss(b, params, result, config->callback_data);
   nir_push_else(b, NULL);

   nir_push_if(b, candidate_non_opaque);
   /* Normal any-hit return accepts; ignore/terminate intrinsics override it. */
   accept_candidate_hit(b, config);
   if (config->emit_any_hit)
      config->emit_any_hit(b, params, result, config->callback_data);
   nir_push_else(b, NULL);

   nir_push_if(b, candidate_procedural);
   /* Normal intersection return reports no hit; reportIntersection overrides it. */
   ignore_candidate_hit(b, config);
   if (config->emit_intersection)
      config->emit_intersection(b, params, result, config->callback_data);
   /* Report-site lowering runs any-hit immediately so a later report cannot
    * overwrite an earlier accepted candidate.  Keep the old fallback for
    * standalone compiler users that have not opted into deferred reports. */
   if (config->emit_any_hit && !config->defer_intersection_reports) {
      nir_def *reported =
         nir_ieq_imm(b, load_raygen_callback_decision(b, config),
                     VT_RT_CALLBACK_ACCEPT);
      nir_def *need_software_opacity =
         nir_ine_imm(
            b, vt_nir_load_candidate_hit_record_meta_field(
                  b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META0,
                  VT_RT_HIT_RECORD_META0_NEED_SOFTWARE_OPACITY_TEST_SHIFT, 1),
            0);
      nir_push_if(b, nir_iand(b, reported, need_software_opacity));
      /* Normal any-hit return accepts; ignore/terminate intrinsics override it. */
      accept_candidate_hit(b, config);
      config->emit_any_hit(b, params, result, config->callback_data);
      nir_pop_if(b, NULL);
   }

   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);

   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
}

static bool
lower_trace_ray(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   if (intrin->intrinsic != nir_intrinsic_trace_ray)
      return false;

   struct lower_rt_io_state *state = data;
   const struct vt_nir_rt_config *config = state->config;
   if (config->execution_model != VT_RT_EXECUTION_ITERATIVE_RAYGEN_LOOP &&
       config->execution_model != VT_RT_EXECUTION_WAVEFRONT_GLOBAL_QUEUE)
      return false;

   b->cursor = nir_after_instr(&intrin->instr);
   nir_def **saved_parent_context = NULL;

   if (state->trace_depth > 0) {
      nir_function_impl *impl = nir_shader_get_entrypoint(b->shader);
      BITSET_WORD *live = nested_parent_live_words(impl, intrin, config);
      debug_nested_parent_live_words(config, live);
      saved_parent_context = save_nested_parent_context(b, config, live);
      free(live);
   }

   /* The root raygen trace is depth 0.  Traces introduced while lowering an
    * inlined callback are assigned the next fixed payload slot. */
   vt_nir_store_active_level(b, config, VT_RT_LAYOUT_RAYGEN,
                             nir_imm_int(b, state->trace_depth));

   struct trace_payload_binding payload_binding;
   const bool has_payload_binding =
      trace_payload_binding(intrin, config, &payload_binding);
   nir_def *payload = intrin->src[10].ssa;
   if (has_payload_binding) {
      payload = nir_u2u64(
         b, trace_payload_frame_addr(b, config, &payload_binding,
                                     state->trace_depth, 0));
   }

   struct vt_nir_rt_trace_params params = {
      .accel_struct = intrin->src[0].ssa,
      .flags = intrin->src[1].ssa,
      .cull_mask = intrin->src[2].ssa,
      .sbt_base = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HIT_SBT_ADDR, 64),
      .sbt_stride_bytes =
         load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HIT_SBT_STRIDE, 64),
      .miss_sbt_base = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_MISS_SBT_ADDR, 64),
      .miss_sbt_stride_bytes =
         load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_MISS_SBT_STRIDE, 64),
      .sbt_offset = intrin->src[3].ssa,
      .sbt_stride = intrin->src[4].ssa,
      .miss_index = intrin->src[5].ssa,
      .origin = intrin->src[6].ssa,
      .tmin = intrin->src[7].ssa,
      .direction = intrin->src[8].ssa,
      .tmax = intrin->src[9].ssa,
      .payload = payload,
   };
   /* The root ray already owns frame zero.  A nested trace receives the
    * caller's inout payload in its distinct depth frame, so no stale private
    * alloca participates in the ABI. */
   if (has_payload_binding && state->trace_depth > 0)
      handoff_trace_payload_frame(b, config, &payload_binding,
                                  state->trace_depth - 1,
                                  state->trace_depth);

   bool skip_closest_hit =
      trace_ray_has_const_flag(intrin, VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT);

   /*
    * Transitional wavefront profile: publish the exact trace input through
    * the new global mailbox, then retain the proven synchronous traversal
    * loop until the dequeue worker and CPS resume entry are available.
    */
   if (config->execution_model == VT_RT_EXECUTION_WAVEFRONT_GLOBAL_QUEUE) {
      /* Raygen is not a compute shader: use the Vulkan RT launch id. */
      nir_def *id = nir_load_ray_launch_id(b);
      nir_def *width = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_WIDTH, 32);
      nir_def *height = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HEIGHT, 32);
      nir_def *row = nir_iadd(b, nir_channel(b, id, 1),
                               nir_imul(b, height, nir_channel(b, id, 2)));
      nir_def *linear = nir_iadd(b, nir_channel(b, id, 0),
                                  nir_imul(b, width, row));
      nir_def *pool = load_dispatch_arg(
         b, VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_BASE, 64);
      nir_def *stride = load_dispatch_arg(
         b, VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_STRIDE, 32);
      struct vt_nir_rt_submission_mailbox_args mailbox = {
         .base_addr = nir_iadd(b, pool, nir_imul(b, nir_u2u64(b, linear),
                                                  nir_u2u64(b, stride))),
         .generation = load_dispatch_arg(
            b, VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_GENERATION, 32),
      };
      vt_nir_rt_store_submission_mailbox(b, &mailbox, &params,
                                         nir_imm_int(b, 0),
                                         nir_imm_int(b, 0),
                                         nir_imm_int(b, 0),
                                         nir_imm_int(b, 0), id,
                                         mailbox.generation, nir_imm_int(b, 0));
   }

   nir_loop *loop = nir_push_loop(b);
   {
      struct vt_nir_rt_result result = vt_nir_build_traversal(b, &params, config);
      nir_def *complete_hit =
         nir_ieq_imm(b, result.status, VT_RT_TRAVERSAL_COMPLETE_HIT);
      nir_def *complete_miss =
         nir_ieq_imm(b, result.status, VT_RT_TRAVERSAL_COMPLETE_MISS);
      nir_def *terminated =
         nir_ieq_imm(b, result.status, VT_RT_TRAVERSAL_TERMINATED);
      nir_def *terminal = nir_ior(b, nir_ior(b, complete_hit, complete_miss),
                                  terminated);
      nir_def *candidate_non_opaque =
         nir_ieq_imm(b, result.status,
                     VT_RT_TRAVERSAL_CANDIDATE_NON_OPAQUE_TRIANGLE);
      nir_def *candidate_procedural =
         nir_ieq_imm(b, result.status, VT_RT_TRAVERSAL_CANDIDATE_PROCEDURAL_AABB);
      nir_def *candidate =
         nir_ior(b, candidate_non_opaque, candidate_procedural);

      emit_trace_callbacks(b, config, &params, &result, complete_hit,
                           complete_miss, candidate_non_opaque,
                           candidate_procedural, skip_closest_hit);

      nir_push_if(b, terminal);
      nir_jump(b, nir_jump_break);
      nir_push_else(b, NULL);

      nir_push_if(b, nir_inot(b, candidate));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, loop);

   if (has_payload_binding && state->trace_depth > 0)
      handoff_trace_payload_frame(b, config, &payload_binding,
                                  state->trace_depth,
                                  state->trace_depth - 1);

   if (saved_parent_context) {
      restore_nested_parent_context(b, config, saved_parent_context);
      free(saved_parent_context);
   }

   /* Returning from a nested trace resumes its parent payload frame.  The
    * trace itself ran at trace_depth; code following it must address the
    * enclosing frame at trace_depth - 1. */
   const uint32_t resume_trace_depth =
      state->trace_depth ? state->trace_depth - 1 : 0;
   vt_nir_store_active_level(b, config, VT_RT_LAYOUT_RAYGEN,
                             nir_imm_int(b, resume_trace_depth));

   nir_instr_remove(&intrin->instr);
   return true;
}

/* CPS must see trace_ray itself, so the global root path cannot reuse
 * lower_trace_ray(): that pass replaces the call with the old synchronous
 * traversal loop.  Instead, append the terminal callback dispatch to the
 * original call.  nir_lower_shader_calls then spills all values used here and
 * places this code in the corresponding root resume shader. */
static bool
lower_global_root_continuation(nir_builder *b, nir_intrinsic_instr *intrin,
                               void *data)
{
   if (intrin->intrinsic != nir_intrinsic_trace_ray)
      return false;

   struct lower_global_root_continuation_state *state = data;
   if (_mesa_set_search(state->lowered_traces, intrin))
      return false;
   _mesa_set_add(state->lowered_traces, intrin);

   const struct vt_nir_rt_config *config = state->config;
   b->cursor = nir_after_instr(&intrin->instr);

   struct vt_nir_rt_trace_params params = {
      .accel_struct = intrin->src[0].ssa,
      .flags = intrin->src[1].ssa,
      .cull_mask = intrin->src[2].ssa,
      .sbt_base = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HIT_SBT_ADDR, 64),
      .sbt_stride_bytes =
         load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HIT_SBT_STRIDE, 64),
      .miss_sbt_base = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_MISS_SBT_ADDR, 64),
      .miss_sbt_stride_bytes =
         load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_MISS_SBT_STRIDE, 64),
      .sbt_offset = intrin->src[3].ssa,
      .sbt_stride = intrin->src[4].ssa,
      .miss_index = intrin->src[5].ssa,
      .origin = intrin->src[6].ssa,
      .tmin = intrin->src[7].ssa,
      .direction = intrin->src[8].ssa,
      .tmax = intrin->src[9].ssa,
      .payload = intrin->src[10].ssa,
   };
   const struct vt_nir_scratchpad_layout *raygen =
      &config->scratchpad[VT_RT_LAYOUT_RAYGEN];
   nir_def *ray_state = vt_nir_scratchpad_addr(
      b, config, VT_RT_LAYOUT_RAYGEN, raygen->ray_state_base_bytes);
   nir_def *status = nir_ventus_rt_load_slot(
      b, 1, 32, nir_iadd_imm(b, ray_state, VT_RT_SLOT_STATUS * 4));
   nir_def *record_ptr = vt_nir_build_hit_shader_record_ptr(
      b, config, params.sbt_base, params.sbt_stride_bytes,
      params.sbt_offset, params.sbt_stride,
      vt_nir_load_hit_record_meta_field(
         b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META1,
         VT_RT_HIT_RECORD_META1_INSTANCE_SBT_RECORD_OFFSET_SHIFT,
         VT_RT_HIT_RECORD_META1_INSTANCE_SBT_RECORD_OFFSET_BITS),
      vt_nir_load_hit_record_word(
         b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_GEOMETRY_ID, 32));
   nir_def *miss_record_ptr = nir_iadd(
      b, params.miss_sbt_base,
      nir_iadd_imm(b,
         nir_imul(b, nir_u2u64(b, params.miss_index),
                  nir_u2u64(b, params.miss_sbt_stride_bytes)),
         config->shader_group_handle_size));
   struct vt_nir_rt_result result = {
      .status = status,
      .hit_shader_record_ptr = record_ptr,
      .miss_shader_record_ptr = miss_record_ptr,
   };
   nir_def *complete_hit =
      nir_ieq_imm(b, status, VT_RT_TRAVERSAL_COMPLETE_HIT);
   nir_def *complete_miss =
      nir_ieq_imm(b, status, VT_RT_TRAVERSAL_COMPLETE_MISS);
   emit_trace_callbacks(b, config, &params, &result, complete_hit,
                        complete_miss, nir_imm_false(b), nir_imm_false(b),
                        trace_ray_has_const_flag(
                           intrin, VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT));
   return true;
}

bool
vt_nir_lower_rt_global_root_continuations(
   nir_shader *shader, const struct vt_nir_rt_config *config)
{
   if (!shader || !config ||
       config->execution_model != VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER ||
       !config->enable_monolithic_shader_callbacks)
      return false;

   struct lower_global_root_continuation_state state = {
      .config = config,
      .lowered_traces = _mesa_set_create(NULL, _mesa_hash_pointer,
                                         _mesa_key_pointer_equal),
   };
   if (!state.lowered_traces)
      return false;
   /* Inlining a closest-hit callback may introduce another TraceRayEXT.
    * Visit to a fixed point while remembering processed instructions so each
    * call gets exactly one terminal-dispatch suffix before CPS splits it. */
   bool progress = false;
   for (uint32_t i = 0; i < 8; i++) {
      bool pass_progress = nir_shader_intrinsics_pass(
         shader, lower_global_root_continuation, nir_metadata_none, &state);
      progress |= pass_progress;
      if (!pass_progress)
         break;
   }
   _mesa_set_destroy(state.lowered_traces, NULL);
   if (progress) {
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      nir_index_blocks(impl);
      nir_index_ssa_defs(impl);
      nir_shader_gather_info(shader, impl);
   }
   return progress;
}

static bool
insert_raygen_release_before_returns(nir_shader *shader,
                                     const struct vt_nir_rt_config *config)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   nir_builder b = nir_builder_create(impl);
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_jump)
            continue;

         nir_jump_instr *jump = nir_instr_as_jump(instr);
         if (jump->type != nir_jump_return && jump->type != nir_jump_halt)
            continue;

         b.cursor = nir_before_instr(instr);
         nir_def *slot_base =
            vt_nir_scratchpad_addr(&b, config, VT_RT_LAYOUT_RAYGEN,
                                   config->scratchpad[VT_RT_LAYOUT_RAYGEN].ray_state_base_bytes);
         nir_ventus_rt_store_slot(
            &b, nir_imm_int(&b, VT_RT_SLOT_STATUS_TERMINATED), slot_base);
         nir_ventus_rt_release(&b, nir_ventus_rt_pds_warp_tid_base(&b));
         progress = true;
      }
   }

   return progress;
}

bool
vt_nir_lower_rt_io_monolithic(nir_shader *shader,
                              const struct vt_nir_rt_config *config)
{
   struct vt_nir_rt_config default_config;
   if (!config) {
      default_config = vt_nir_rt_default_config();
      config = &default_config;
   }

   struct lower_rt_io_state state = {
      .config = config,
   };

   bool progress = false;
   bool trace_progress = false;
   for (uint32_t i = 0; i < 8; i++) {
      /* Callback shaders have already had their ABI lowered before they are
       * inlined.  Fold their cloned slot addresses here for the PDS liveness
       * classifier, but do not run the ABI lowering a second time. */
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      nir_index_blocks(impl);
      nir_index_ssa_defs(impl);
      NIR_PASS(_, shader, nir_opt_constant_folding);
      state.trace_depth = i;
      bool pass_progress =
         nir_shader_intrinsics_pass(shader, lower_trace_ray, nir_metadata_none,
                                    &state);
      progress |= pass_progress;
      trace_progress |= pass_progress;
      if (!pass_progress)
         break;
   }

   if (trace_progress)
      progress |= insert_raygen_release_before_returns(shader, config);

   progress |= nir_shader_intrinsics_pass(
      shader, lower_monolithic_push_constants, nir_metadata_none, NULL);

   if (progress) {
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      if (impl) {
         nir_index_blocks(impl);
         nir_index_ssa_defs(impl);
      }
   }

   return progress;
}
