/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

#define VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT 0x8u

struct lower_rt_io_state {
   const struct vt_nir_rt_config *config;
};

struct lower_global_root_continuation_state {
   const struct vt_nir_rt_config *config;
   struct set *lowered_traces;
};

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

static bool
trace_ray_has_const_flag(nir_intrinsic_instr *intrin, uint32_t flag)
{
   return nir_src_is_const(intrin->src[1]) &&
          (nir_src_as_uint(intrin->src[1]) & flag) != 0;
}

static void
store_raygen_control(nir_builder *b, const struct vt_nir_rt_config *config,
                     enum vt_rt_control_word word, nir_def *value)
{
   vt_nir_store_control_word(b, config, VT_RT_LAYOUT_RAYGEN, word, value);
}

static nir_def *
load_raygen_control(nir_builder *b, const struct vt_nir_rt_config *config,
                    enum vt_rt_control_word word)
{
   return vt_nir_load_control_word(b, config, VT_RT_LAYOUT_RAYGEN, word, 32);
}

static void
accept_candidate_hit(nir_builder *b, const struct vt_nir_rt_config *config)
{
   store_raygen_control(b, config, VT_RT_CONTROL_ACCEPT_HIT, nir_imm_true(b));
   store_raygen_control(b, config, VT_RT_CONTROL_IGNORE_HIT, nir_imm_false(b));
   store_raygen_control(b, config, VT_RT_CONTROL_TERMINATE_RAY, nir_imm_false(b));
}

static void
ignore_candidate_hit(nir_builder *b, const struct vt_nir_rt_config *config)
{
   store_raygen_control(b, config, VT_RT_CONTROL_ACCEPT_HIT, nir_imm_false(b));
   store_raygen_control(b, config, VT_RT_CONTROL_IGNORE_HIT, nir_imm_true(b));
   store_raygen_control(b, config, VT_RT_CONTROL_TERMINATE_RAY, nir_imm_false(b));
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
   if (config->emit_any_hit) {
      nir_def *reported =
         nir_i2b(b, load_raygen_control(b, config, VT_RT_CONTROL_ACCEPT_HIT));
      nir_push_if(b, reported);
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
   if (config->execution_model != VT_RT_EXECUTION_ITERATIVE_RAYGEN_LOOP)
      return false;

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
      .abi_tag = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_ABI_TAG, 32),
   };

   bool skip_closest_hit =
      trace_ray_has_const_flag(intrin, VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT);

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
   nir_def *record_ptr = nir_pack_64_2x32_split(
      b, vt_nir_load_hit_record_word(
            b, config, VT_RT_LAYOUT_RAYGEN,
            VT_RT_HIT_RECORD_SHADER_RECORD_PTR_LO, 32),
      vt_nir_load_hit_record_word(
            b, config, VT_RT_LAYOUT_RAYGEN,
            VT_RT_HIT_RECORD_SHADER_RECORD_PTR_HI, 32));
   nir_def *miss_record_ptr = nir_iadd(
      b, params.miss_sbt_base,
      nir_iadd_imm(b,
         nir_imul(b, nir_u2u64(b, params.miss_index),
                  nir_u2u64(b, params.miss_sbt_stride_bytes)),
         config->shader_group_handle_size));
   struct vt_nir_rt_result result = {
      .status = status,
      .sbt_index = vt_nir_load_hit_record_word(
         b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_SBT_INDEX, 32),
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

   const struct vt_nir_scratchpad_layout *raygen =
      &config->scratchpad[VT_RT_LAYOUT_RAYGEN];

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
                                   raygen->ray_state_base_bytes);
         nir_ventus_rt_release(&b, slot_base);
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
   for (uint32_t i = 0; i < 8; i++) {
      bool pass_progress =
         nir_shader_intrinsics_pass(shader, lower_trace_ray, nir_metadata_none,
                                    &state);
      progress |= pass_progress;
      if (!pass_progress)
         break;
   }

   if (progress)
      progress |= insert_raygen_release_before_returns(shader, config);

   if (progress) {
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      if (impl) {
         nir_index_blocks(impl);
         nir_index_ssa_defs(impl);
      }
   }

   return progress;
}
