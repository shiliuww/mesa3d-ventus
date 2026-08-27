/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

/*
 * PDS-free root wavefront lowering.  nir_lower_shader_calls has already
 * converted traceRayEXT into rt_trace_ray(call_idx, stack_size), spilled live
 * SSA into the global CPS stack, and split every post-trace path into a resume
 * shader.  This pass publishes that generic call as a mailbox record.
 *
 * The reflections MVP intentionally has one active trace per root launch
 * invocation.  Its frame ordinal is therefore the root launch's linear ID;
 * a future child-frame allocator must replace root_cps_frame(), not reuse a
 * physical lane or a PDS address.
 */

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

static nir_def *callback_ray_ref(nir_builder *b);
static nir_def *load_completion_field(
   nir_builder *b, nir_def *ray_ref, enum vt_rt_global_completion_field field);
static nir_def *global_callback_slot_addr(
   nir_builder *b, const struct vt_nir_rt_config *config, uint32_t slot,
   nir_def *ray_ref);
static bool constant_def_u32(nir_def *def, uint32_t *value);

/* Candidate callbacks are separate launches, so only state consumed entirely
 * within one callback may live in LDS.  Keep one fixed lane slice: it is
 * intentionally small (144 B) and stays below the Vulkan-advertised 32 KiB
 * workgroup limit for the 32-lane compact dispatch. */
#define VT_RT_CALLBACK_LDS_LANE_STRIDE 144u
#define VT_RT_CALLBACK_LDS_CANDIDATE_BASE 0u
#define VT_RT_CALLBACK_LDS_HIT_ATTRIB_BASE 96u
#define VT_RT_CALLBACK_LDS_MAX_LANES VT_RT_GLOBAL_WARP_SIZE
#define VT_RT_CALLBACK_LDS_SIZE_BYTES \
   (VT_RT_CALLBACK_LDS_LANE_STRIDE * VT_RT_CALLBACK_LDS_MAX_LANES)

struct callback_lds_state {
   const struct vt_nir_rt_config *config;
   bool candidate_word[VT_RT_HIT_RECORD_WORD_COUNT];
   bool hit_attrib_word[VT_RT_ABI_HIT_ATTRIB_SIZE_BYTES / 4];
   uint32_t lds_size_bytes;
};

static bool
lower_global_push_constants(nir_builder *b, nir_intrinsic_instr *intrin,
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
   nir_def_rewrite_uses(&intrin->def,
      nir_load_global(b, intrin->def.num_components, intrin->def.bit_size,
                      nir_iadd(b, base, offset)));
   nir_instr_remove(&intrin->instr);
   return true;
}

/* Completion owns the logical frame identity.  All callback and resume
 * accesses to mutable ray state must derive their physical CPS address from
 * that one identity; do not introduce a second payload-address authority in
 * the completion plane. */
static nir_def *
completion_cps_frame_addr(nir_builder *b, nir_def *ray_ref)
{
   nir_def *arena = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_CPS_ARENA_ADDR, 64);
   nir_def *stride = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_CPS_FRAME_STRIDE, 32);
   nir_def *frame = load_completion_field(
      b, ray_ref, VT_RT_GLOBAL_COMPLETION_CPS_FRAME);
   return nir_iadd(b, arena,
      nir_imul(b, nir_u2u64(b, frame), nir_u2u64(b, stride)));
}

struct global_lower_state {
   const struct vt_nir_rt_config *config;
   bool resume_shader;
};

static nir_def *
root_cps_frame(nir_builder *b, nir_def *launch_id)
{
   nir_def *width = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_WIDTH, 32);
   nir_def *height = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HEIGHT, 32);
   nir_def *row = nir_iadd(b, nir_channel(b, launch_id, 1),
                            nir_imul(b, height,
                                     nir_channel(b, launch_id, 2)));
   return nir_iadd(b, nir_channel(b, launch_id, 0), nir_imul(b, width, row));
}

static nir_def *
root_cps_frame_addr(nir_builder *b, nir_def *launch_id)
{
   nir_def *arena = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_CPS_ARENA_ADDR, 64);
   nir_def *stride = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_CPS_FRAME_STRIDE, 32);
   return nir_iadd(b, arena,
                   nir_imul(b, nir_u2u64(b, root_cps_frame(b, launch_id)),
                            nir_u2u64(b, stride)));
}

static bool
lower_global_cps_stack_base(nir_builder *b, nir_intrinsic_instr *intrin,
                            void *data)
{
   if (intrin->intrinsic != nir_intrinsic_load_scratch_base_ptr)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   const struct global_lower_state *state = data;
   nir_def *frame_addr;
   if (state && state->resume_shader) {
      nir_def *ray_ref = callback_ray_ref(b);
      frame_addr = completion_cps_frame_addr(b, ray_ref);
   } else {
      nir_def *launch_id = nir_load_ray_launch_id(b);
      frame_addr = root_cps_frame_addr(b, launch_id);
   }
   nir_def_rewrite_uses(&intrin->def, frame_addr);
   nir_instr_remove(&intrin->instr);
   return true;
}

static nir_def *
global_worker_cps_frame_addr(nir_builder *b,
                             const struct global_lower_state *state)
{
   if (state && state->resume_shader) {
      nir_def *ray_ref = callback_ray_ref(b);
      return completion_cps_frame_addr(b, ray_ref);
   }

   return root_cps_frame_addr(b, nir_load_ray_launch_id(b));
}

/* Payload lowering deliberately leaves abstract RT slots so the PDS and
 * global profiles can share its type/layout logic.  Bind those slots to the
 * CPS arena here; letting LLVM lower them directly would recreate
 * addrspace(5) private/PDS accesses. */
static bool
lower_global_worker_slot(nir_builder *b, nir_intrinsic_instr *intrin,
                         void *data)
{
   if (intrin->intrinsic != nir_intrinsic_ventus_rt_load_slot &&
       intrin->intrinsic != nir_intrinsic_ventus_rt_store_slot)
      return false;

   const bool load = intrin->intrinsic == nir_intrinsic_ventus_rt_load_slot;
   nir_def *slot_def = intrin->src[load ? 0 : 1].ssa;
   uint32_t slot = 0;
   if (!constant_def_u32(slot_def, &slot))
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   const struct global_lower_state *state = data;
   nir_def *addr = NULL;
   if (state && state->resume_shader)
      addr = global_callback_slot_addr(b, state->config, slot,
                                      callback_ray_ref(b));
   else
      addr = nir_iadd_imm(b, global_worker_cps_frame_addr(b, state), slot);
   if (!addr)
      return false;
   if (load) {
      nir_def_rewrite_uses(&intrin->def,
         nir_load_global(b, intrin->def.num_components,
                         intrin->def.bit_size, addr));
   } else {
      nir_store_global(b, intrin->src[0].ssa, addr, .align_mul = 4);
   }
   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
lower_root_trace(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   if (intrin->intrinsic != nir_intrinsic_rt_trace_ray)
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   /* The dummy if gives the inserted halt a structured CFG home. */
   nir_push_if(b, nir_imm_true(b));
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

   const struct global_lower_state *state = data;
   nir_def *launch_id;
   nir_def *cps_frame_addr;
   if (state && state->resume_shader) {
      nir_def *ray_ref = callback_ray_ref(b);
      launch_id = nir_vec3(
         b, load_completion_field(b, ray_ref,
                                  VT_RT_GLOBAL_COMPLETION_LAUNCH_ID_X),
         load_completion_field(b, ray_ref,
                               VT_RT_GLOBAL_COMPLETION_LAUNCH_ID_Y),
         load_completion_field(b, ray_ref,
                               VT_RT_GLOBAL_COMPLETION_LAUNCH_ID_Z));
      cps_frame_addr = global_worker_cps_frame_addr(b, state);
   } else {
      launch_id = nir_load_ray_launch_id(b);
      cps_frame_addr = global_worker_cps_frame_addr(b, state);
   }
   /* Queue the same CPS-backed payload slot used by global worker lowering.
    * base_bytes is an ABI offset within a global CPS frame here, never a PDS
    * address.  The original trace operand names a function-local deref that
    * ceases to exist once this producer halts. */
   params.payload = nir_iadd_imm(
      b, cps_frame_addr,
      state->config->scratchpad[VT_RT_LAYOUT_RAYGEN].base_bytes +
      state->config->scratchpad[VT_RT_LAYOUT_RAYGEN].payload_base_bytes);
   nir_def *mailbox_pool = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_BASE, 64);
   nir_def *mailbox_stride = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_STRIDE, 32);
   nir_intrinsic_instr *worker_intrin = nir_intrinsic_instr_create(
      b->shader, nir_intrinsic_ventus_rt_worker_id);
   nir_def_init(&worker_intrin->instr, &worker_intrin->def, 1, 32);
   nir_builder_instr_insert(b, &worker_intrin->instr);
   /* The mailbox belongs to the physical compact dispatch lane.  A resume's
    * restored launch ID is logical state and can alias across compacted rays. */
   nir_def *lane_source = nir_channel(b, launch_id, 0);
   if (state && state->resume_shader) {
      nir_intrinsic_instr *dispatch_index = nir_intrinsic_instr_create(
         b->shader, nir_intrinsic_ventus_rt_dispatch_index);
      nir_def_init(&dispatch_index->instr, &dispatch_index->def, 1, 32);
      nir_builder_instr_insert(b, &dispatch_index->instr);
      lane_source = &dispatch_index->def;
   }
   nir_def *lane_id = nir_iand_imm(b, lane_source,
                                   VT_RT_GLOBAL_WARP_SIZE - 1u);
   nir_def *mailbox_addr = vt_nir_rt_submission_mailbox_addr(
      b, mailbox_pool, mailbox_stride, &worker_intrin->def, lane_id,
      VT_RT_SUBMISSION_MAILBOX_PRIMARY);
   struct vt_nir_rt_submission_mailbox_args mailbox = {
      .base_addr = mailbox_addr,
      .generation = load_dispatch_arg(
         b, VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_GENERATION, 32),
      .queue_base_addr = load_dispatch_arg(
         b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_QUEUE_BASE, 64),
   };
   vt_nir_rt_store_submission_mailbox(
      b, &mailbox, &params, nir_imm_int(b, 0), root_cps_frame(b, launch_id),
      nir_imm_int(b, nir_intrinsic_call_idx(intrin)),
      nir_imm_int(b, nir_intrinsic_stack_size(intrin)), launch_id,
      load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_QUEUE_GENERATION, 32),
      nir_imm_int(b, 0));

   /* RV32 instruction selection consumes the checked low half. */
   nir_ventus_rt_enqueue(b, nir_u2u32(b, mailbox_addr));
   nir_jump(b, nir_jump_halt);
   nir_pop_if(b, NULL);
   return true;
}

static bool
remove_resume_marker(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   (void)b;
   (void)data;
   if (intrin->intrinsic != nir_intrinsic_rt_resume)
      return false;
   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
is_ray_launch_id_load(nir_intrinsic_instr *intrin)
{
   if (intrin->intrinsic == nir_intrinsic_load_ray_launch_id)
      return true;
   if (intrin->intrinsic != nir_intrinsic_load_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   return var && var->data.mode == nir_var_system_value &&
          var->data.location == SYSTEM_VALUE_RAY_LAUNCH_ID;
}

static bool
lower_global_resume_launch_id(nir_builder *b, nir_intrinsic_instr *intrin,
                              void *data)
{
   (void)data;
   if (!is_ray_launch_id_load(intrin))
      return false;
   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *ray_ref = callback_ray_ref(b);
   nir_def_rewrite_uses(&intrin->def, nir_vec3(
      b, load_completion_field(b, ray_ref,
                               VT_RT_GLOBAL_COMPLETION_LAUNCH_ID_X),
      load_completion_field(b, ray_ref,
                            VT_RT_GLOBAL_COMPLETION_LAUNCH_ID_Y),
      load_completion_field(b, ray_ref,
                            VT_RT_GLOBAL_COMPLETION_LAUNCH_ID_Z)));
   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
lower_global_shader(nir_shader *shader, const struct vt_nir_rt_config *config,
                    bool resume_shader)
{
   if (!shader || !config ||
       config->execution_model != VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER)
      return false;

   /* Callback and child trace dispatch deliberately remain unsupported. */
   if (config->enable_monolithic_shader_callbacks || config->emit_closest_hit ||
       config->emit_miss || config->emit_any_hit || config->emit_intersection)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   const struct global_lower_state state = {
      .config = config,
      .resume_shader = resume_shader,
   };
   bool progress = nir_shader_intrinsics_pass(
      shader, lower_global_cps_stack_base, nir_metadata_none, (void *)&state);
   progress |= nir_shader_intrinsics_pass(shader, lower_global_push_constants,
                                          nir_metadata_none, NULL);
   progress |= nir_shader_intrinsics_pass(shader, lower_global_worker_slot,
                                          nir_metadata_none, (void *)&state);
   if (resume_shader)
      progress |= nir_shader_intrinsics_pass(shader, remove_resume_marker,
                                             nir_metadata_none, NULL);
   progress |= nir_shader_intrinsics_pass(shader, lower_root_trace,
                                          nir_metadata_none, (void *)&state);
   if (resume_shader)
      progress |= nir_shader_intrinsics_pass(shader, lower_global_resume_launch_id,
                                             nir_metadata_none, NULL);
   if (!progress)
      return false;

   nir_opt_dead_cf(shader);
   nir_index_blocks(impl);
   nir_index_ssa_defs(impl);
   return true;
}

bool
vt_nir_lower_rt_global_raygen(nir_shader *shader,
                              const struct vt_nir_rt_config *config)
{
   if (!shader || shader->info.stage != MESA_SHADER_RAYGEN)
      return false;
   return lower_global_shader(shader, config, false);
}

bool
vt_nir_lower_rt_global_resume(nir_shader *shader,
                               const struct vt_nir_rt_config *config)
{
   if (!shader || shader->info.stage != MESA_SHADER_RAYGEN)
      return false;
   return lower_global_shader(shader, config, true);
}

/* Terminal callback kernels run over a compact list whose logical identity is
 * ray_ref, not the physical launch lane.  The queue remains the immutable ray
 * input plane while the completion plane owns the visible traversal result. */
static nir_def *
callback_ray_ref(nir_builder *b)
{
   nir_intrinsic_instr *index = nir_intrinsic_instr_create(
      b->shader, nir_intrinsic_ventus_rt_dispatch_index);
   nir_def_init(&index->instr, &index->def, 1, 32);
   nir_builder_instr_insert(b, &index->instr);
   nir_def *list = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_RAY_REF_LIST_BASE, 64);
   nir_def *offset = nir_imul_imm(b, nir_u2u64(b, &index->def), 4);
   return nir_load_global(b, 1, 32, nir_iadd(b, list, offset));
}

nir_def *
vt_nir_rt_global_callback_ray_ref(nir_builder *b)
{
   return callback_ray_ref(b);
}

static nir_def *
completion_field_addr(nir_builder *b, nir_def *ray_ref,
                      enum vt_rt_global_completion_field field)
{
   nir_def *base = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_BASE, 64);
   nir_def *stride = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_FIELD_STRIDE, 32);
   nir_def *field_offset = nir_imul_imm(b, nir_u2u64(b, stride), field);
   nir_def *ray_offset = nir_imul_imm(b, nir_u2u64(b, ray_ref), 4);
   return nir_iadd_imm(b, nir_iadd(b, base,
                                   nir_iadd(b, field_offset, ray_offset)),
                       VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES);
}

static nir_def *
queue_field_addr(nir_builder *b, nir_def *ray_ref,
                 enum vt_rt_global_trace_field field)
{
   nir_def *base = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_QUEUE_BASE, 64);
   nir_def *capacity = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_QUEUE_CAPACITY, 32);
   nir_def *stride = nir_iand_imm(b,
      nir_iadd_imm(b, nir_imul_imm(b, nir_u2u64(b, capacity), 4),
                   VT_RT_GLOBAL_ALIGNMENT_BYTES - 1),
      ~(uint64_t)(VT_RT_GLOBAL_ALIGNMENT_BYTES - 1));
   nir_def *field_offset = nir_imul_imm(b, stride, field);
   nir_def *ray_offset = nir_imul_imm(b, nir_u2u64(b, ray_ref), 4);
   return nir_iadd_imm(b, nir_iadd(b, base,
                                   nir_iadd(b, field_offset, ray_offset)),
                       VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES);
}

static nir_def *
load_completion_field(nir_builder *b, nir_def *ray_ref,
                      enum vt_rt_global_completion_field field)
{
   return nir_load_global(b, 1, 32, completion_field_addr(b, ray_ref, field));
}

static nir_def *
load_completion_addr(nir_builder *b, nir_def *ray_ref,
                     enum vt_rt_global_completion_field lo)
{
   return nir_pack_64_2x32_split(
      b, load_completion_field(b, ray_ref, lo),
      load_completion_field(b, ray_ref, lo + 1));
}

nir_def *
vt_nir_rt_global_load_completion_addr(
   nir_builder *b, nir_def *ray_ref, enum vt_rt_global_completion_field lo)
{
   return load_completion_addr(b, ray_ref, lo);
}

static bool
constant_def_u32(nir_def *def, uint32_t *value)
{
   if (!def)
      return false;
   nir_instr *instr = nir_def_instr(def);
   if (!instr)
      return false;
   if (instr->type == nir_instr_type_load_const) {
      *value = nir_instr_as_load_const(instr)->value[0].u32;
      return true;
   }
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_iadd && alu->op != nir_op_u2u64)
      return false;
   if (alu->op == nir_op_u2u64)
      return constant_def_u32(alu->src[0].src.ssa, value);
   uint32_t a = 0, b = 0;
   if (!constant_def_u32(alu->src[0].src.ssa, &a) ||
       !constant_def_u32(alu->src[1].src.ssa, &b))
      return false;
   *value = a + b;
   return true;
}

static enum vt_rt_global_trace_field
trace_field_for_slot(enum vt_rt_slot_word word)
{
   static const enum vt_rt_global_trace_field fields[VT_RT_SLOT_WORD_COUNT] = {
      [VT_RT_SLOT_ACCEL_LO] = VT_RT_GLOBAL_TRACE_TLAS_ADDR_LO,
      [VT_RT_SLOT_ACCEL_HI] = VT_RT_GLOBAL_TRACE_TLAS_ADDR_HI,
      [VT_RT_SLOT_TRACE_META0] = VT_RT_GLOBAL_TRACE_FLAGS,
      [VT_RT_SLOT_TRACE_META1] = VT_RT_GLOBAL_TRACE_MISS_INDEX,
      [VT_RT_SLOT_ORIGIN_X] = VT_RT_GLOBAL_TRACE_ORIGIN_X,
      [VT_RT_SLOT_ORIGIN_Y] = VT_RT_GLOBAL_TRACE_ORIGIN_Y,
      [VT_RT_SLOT_ORIGIN_Z] = VT_RT_GLOBAL_TRACE_ORIGIN_Z,
      [VT_RT_SLOT_TMIN] = VT_RT_GLOBAL_TRACE_TMIN,
      [VT_RT_SLOT_DIRECTION_X] = VT_RT_GLOBAL_TRACE_DIRECTION_X,
      [VT_RT_SLOT_DIRECTION_Y] = VT_RT_GLOBAL_TRACE_DIRECTION_Y,
      [VT_RT_SLOT_DIRECTION_Z] = VT_RT_GLOBAL_TRACE_DIRECTION_Z,
      [VT_RT_SLOT_TMAX] = VT_RT_GLOBAL_TRACE_TMAX,
      [VT_RT_SLOT_PAYLOAD_PTR_LO] = VT_RT_GLOBAL_TRACE_PAYLOAD_LO,
      [VT_RT_SLOT_PAYLOAD_PTR_HI] = VT_RT_GLOBAL_TRACE_PAYLOAD_HI,
   };
   return word < VT_RT_SLOT_WORD_COUNT ? fields[word] :
                                         VT_RT_GLOBAL_TRACE_READY;
}

static nir_def *
global_callback_slot_addr(nir_builder *b, const struct vt_nir_rt_config *config,
                          uint32_t slot, nir_def *ray_ref)
{
   const struct vt_nir_scratchpad_layout *sp =
      &config->scratchpad[vt_nir_layout_for_stage(b->shader->info.stage)];
   if (slot < sp->base_bytes)
      return NULL;
   uint32_t offset = slot - sp->base_bytes;
   if (offset >= sp->ray_state_base_bytes &&
       offset < sp->ray_state_base_bytes + VT_RT_SLOT_WORD_COUNT * 4) {
      enum vt_rt_slot_word word =
         (enum vt_rt_slot_word)((offset - sp->ray_state_base_bytes) / 4);
      if (word == VT_RT_SLOT_STATUS)
         return completion_field_addr(b, ray_ref, VT_RT_GLOBAL_COMPLETION_STATUS);
      return queue_field_addr(b, ray_ref, trace_field_for_slot(word));
   }
   if (offset >= sp->hit_record_base_bytes &&
       offset < sp->hit_record_base_bytes + VT_RT_HIT_RECORD_WORD_COUNT * 4) {
      uint32_t word = (offset - sp->hit_record_base_bytes) / 4;
      return completion_field_addr(
         b, ray_ref,
         (enum vt_rt_global_completion_field)(
            VT_RT_GLOBAL_COMPLETION_COMMITTED_HIT_BASE + word));
   }
   if (offset >= sp->candidate_hit_record_base_bytes &&
       offset < sp->candidate_hit_record_base_bytes +
                    VT_RT_HIT_RECORD_WORD_COUNT * 4) {
      uint32_t word =
         (offset - sp->candidate_hit_record_base_bytes) / 4;
      return completion_field_addr(
         b, ray_ref,
         (enum vt_rt_global_completion_field)(
            VT_RT_GLOBAL_COMPLETION_CANDIDATE_HIT_BASE + word));
   }
   if (offset >= sp->control_base_bytes &&
       offset < sp->control_base_bytes + VT_RT_CONTROL_WORD_COUNT * 4) {
      uint32_t word = (offset - sp->control_base_bytes) / 4;
      return completion_field_addr(
         b, ray_ref,
         (enum vt_rt_global_completion_field)(
            VT_RT_GLOBAL_COMPLETION_CANDIDATE_CONTROL_BASE + word));
   }
   if (offset >= sp->payload_base_bytes &&
       offset < sp->total_size_bytes) {
      uint32_t payload_offset = offset - sp->payload_base_bytes;
      const struct vt_nir_scratchpad_layout *raygen_sp =
         &config->scratchpad[VT_RT_LAYOUT_RAYGEN];
      return nir_iadd_imm(
         b, completion_cps_frame_addr(b, ray_ref),
         raygen_sp->base_bytes + raygen_sp->payload_base_bytes +
            payload_offset);
   }
   const bool candidate_hit_attrib =
      offset >= sp->hit_attrib_base_bytes &&
      offset < sp->hit_attrib_base_bytes + sp->hit_attrib_size_bytes;
   const bool committed_hit_attrib =
      offset >= sp->committed_hit_attrib_base_bytes &&
      offset < sp->committed_hit_attrib_base_bytes + sp->hit_attrib_size_bytes;
   if (candidate_hit_attrib || committed_hit_attrib) {
      uint32_t attrib_offset = offset -
         (candidate_hit_attrib ? sp->hit_attrib_base_bytes :
                                sp->committed_hit_attrib_base_bytes);
      nir_def *status = load_completion_field(
         b, ray_ref, VT_RT_GLOBAL_COMPLETION_STATUS);
      nir_def *candidate = nir_ior(
         b, nir_ieq_imm(b, status,
                         VT_RT_TRAVERSAL_CANDIDATE_NON_OPAQUE_TRIANGLE),
         nir_ieq_imm(b, status,
                         VT_RT_TRAVERSAL_CANDIDATE_PROCEDURAL_AABB));
      nir_def *committed_addr = load_completion_addr(
         b, ray_ref, VT_RT_GLOBAL_COMPLETION_HIT_ATTRIBUTE_ADDR_LO);
      nir_def *candidate_addr = load_completion_addr(
         b, ray_ref,
         VT_RT_GLOBAL_COMPLETION_CANDIDATE_HIT_ATTRIBUTE_ADDR_LO);
      return nir_iadd_imm(b,
                          committed_hit_attrib ? committed_addr :
                          nir_bcsel(b, candidate, candidate_addr, committed_addr),
                          attrib_offset);
   }
   return NULL;
}

/* Returns a lane-private byte offset in the callback LDS slice.  Candidate
 * records and hit attributes are read-only in an any-hit shader, therefore
 * no epilogue writeback is required.  Control words deliberately remain in
 * global completion: RTcore consumes them after this compact launch ends. */
static bool
callback_lds_slot_offset(const struct callback_lds_state *state,
                         uint32_t slot, uint32_t *offset)
{
   if (!state || !state->lds_size_bytes || !offset)
      return false;

   const struct vt_nir_scratchpad_layout *sp =
      &state->config->scratchpad[VT_RT_LAYOUT_ANY_HIT];
   if (slot < sp->base_bytes)
      return false;
   const uint32_t relative = slot - sp->base_bytes;

   if (relative >= sp->candidate_hit_record_base_bytes &&
       relative < sp->candidate_hit_record_base_bytes +
                     VT_RT_HIT_RECORD_WORD_COUNT * 4u) {
      const uint32_t word =
         (relative - sp->candidate_hit_record_base_bytes) / 4u;
      if (!state->candidate_word[word])
         return false;
      *offset = VT_RT_CALLBACK_LDS_CANDIDATE_BASE + word * 4u;
      return true;
   }

   if (relative >= sp->hit_attrib_base_bytes &&
       relative < sp->hit_attrib_base_bytes + sp->hit_attrib_size_bytes) {
      const uint32_t word = (relative - sp->hit_attrib_base_bytes) / 4u;
      if (word >= ARRAY_SIZE(state->hit_attrib_word) ||
          !state->hit_attrib_word[word])
         return false;
      *offset = VT_RT_CALLBACK_LDS_HIT_ATTRIB_BASE + word * 4u;
      return true;
   }

   return false;
}

static nir_def *
callback_lds_addr(nir_builder *b, uint32_t offset)
{
   nir_intrinsic_instr *index = nir_intrinsic_instr_create(
      b->shader, nir_intrinsic_ventus_rt_dispatch_index);
   nir_def_init(&index->instr, &index->def, 1, 32);
   nir_builder_instr_insert(b, &index->instr);
   /* Compact callbacks always use 32 lanes per workgroup.  The physical
    * dispatch index is global, so reduce it to the workgroup-local lane. */
   nir_def *lane = nir_iand_imm(b, &index->def,
                                VT_RT_GLOBAL_WARP_SIZE - 1u);
   return nir_iadd_imm(b, nir_imul_imm(b, lane,
                                       VT_RT_CALLBACK_LDS_LANE_STRIDE),
                       offset);
}

static bool
callback_lds_slot_index(const struct vt_nir_rt_config *config, uint32_t slot,
                        uint32_t *index)
{
   const struct vt_nir_scratchpad_layout *sp =
      &config->scratchpad[VT_RT_LAYOUT_ANY_HIT];
   if (slot < sp->base_bytes)
      return false;
   const uint32_t relative = slot - sp->base_bytes;
   if (relative >= sp->candidate_hit_record_base_bytes &&
       relative < sp->candidate_hit_record_base_bytes +
                     VT_RT_HIT_RECORD_WORD_COUNT * 4u) {
      *index = (relative - sp->candidate_hit_record_base_bytes) / 4u;
      return true;
   }
   if (relative >= sp->hit_attrib_base_bytes &&
       relative < sp->hit_attrib_base_bytes + sp->hit_attrib_size_bytes) {
      *index = VT_RT_HIT_RECORD_WORD_COUNT +
         (relative - sp->hit_attrib_base_bytes) / 4u;
      return true;
   }
   return false;
}

/* Cache only repeated, read-only accesses.  This avoids turning a one-shot
 * global load into a global-to-LDS copy and leaves all mutable ABI state in
 * its cross-launch global owner. */
static void
prepare_callback_lds(nir_shader *shader, struct callback_lds_state *state)
{
   uint8_t loads[VT_RT_HIT_RECORD_WORD_COUNT +
                 VT_RT_ABI_HIT_ATTRIB_SIZE_BYTES / 4] = {};
   bool writes[ARRAY_SIZE(loads)] = {};

   if (shader->info.stage != MESA_SHADER_ANY_HIT)
      return;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            const bool load = intrin->intrinsic ==
               nir_intrinsic_ventus_rt_load_slot;
            const bool store = intrin->intrinsic ==
               nir_intrinsic_ventus_rt_store_slot;
            if (!load && !store)
               continue;
            uint32_t slot = 0, index = 0;
            if (!constant_def_u32(intrin->src[load ? 0 : 1].ssa, &slot) ||
                !callback_lds_slot_index(state->config, slot, &index))
               continue;
            if (load && loads[index] != UINT8_MAX)
               loads[index]++;
            if (store)
               writes[index] = true;
         }
      }
   }

   for (uint32_t word = 0; word < VT_RT_HIT_RECORD_WORD_COUNT; word++)
      state->candidate_word[word] = loads[word] > 1 && !writes[word];
   for (uint32_t word = 0; word < ARRAY_SIZE(state->hit_attrib_word); word++)
      state->hit_attrib_word[word] =
         loads[VT_RT_HIT_RECORD_WORD_COUNT + word] > 1 &&
         !writes[VT_RT_HIT_RECORD_WORD_COUNT + word];

   for (uint32_t word = 0; word < VT_RT_HIT_RECORD_WORD_COUNT; word++)
      state->lds_size_bytes |= state->candidate_word[word];
   for (uint32_t word = 0; word < ARRAY_SIZE(state->hit_attrib_word); word++)
      state->lds_size_bytes |= state->hit_attrib_word[word];
   if (state->lds_size_bytes)
      state->lds_size_bytes = VT_RT_CALLBACK_LDS_SIZE_BYTES;
}

static void
emit_callback_lds_prologue(nir_shader *shader,
                           const struct callback_lds_state *state)
{
   if (!state->lds_size_bytes)
      return;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *ray_ref = callback_ray_ref(&b);
   for (uint32_t word = 0; word < VT_RT_HIT_RECORD_WORD_COUNT; word++) {
      if (!state->candidate_word[word])
         continue;
      nir_def *addr = completion_field_addr(
         &b, ray_ref, (enum vt_rt_global_completion_field)(
            VT_RT_GLOBAL_COMPLETION_CANDIDATE_HIT_BASE + word));
      nir_store_shared(&b, nir_load_global(&b, 1, 32, addr),
                       callback_lds_addr(&b,
                          VT_RT_CALLBACK_LDS_CANDIDATE_BASE + word * 4u),
                       .align_mul = 4);
   }

   nir_def *attribute_base = load_completion_addr(
      &b, ray_ref,
      VT_RT_GLOBAL_COMPLETION_CANDIDATE_HIT_ATTRIBUTE_ADDR_LO);
   for (uint32_t word = 0; word < ARRAY_SIZE(state->hit_attrib_word); word++) {
      if (!state->hit_attrib_word[word])
         continue;
      nir_store_shared(&b,
                       nir_load_global(&b, 1, 32,
                                       nir_iadd_imm(&b, attribute_base, word * 4u)),
                       callback_lds_addr(&b,
                          VT_RT_CALLBACK_LDS_HIT_ATTRIB_BASE + word * 4u),
                       .align_mul = 4);
   }
}

static bool
lower_global_callback_slot(nir_builder *b, nir_intrinsic_instr *intrin,
                           void *data)
{
   if (intrin->intrinsic != nir_intrinsic_ventus_rt_load_slot &&
       intrin->intrinsic != nir_intrinsic_ventus_rt_store_slot)
      return false;

   const bool load = intrin->intrinsic == nir_intrinsic_ventus_rt_load_slot;
   nir_def *slot_def = intrin->src[load ? 0 : 1].ssa;
   uint32_t slot = 0;
   if (!constant_def_u32(slot_def, &slot))
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   const struct callback_lds_state *state = data;
   uint32_t lds_offset = 0;
   if (load && callback_lds_slot_offset(state, slot, &lds_offset)) {
      nir_def_rewrite_uses(&intrin->def,
                           nir_load_shared(b, intrin->def.num_components,
                                           intrin->def.bit_size,
                                           callback_lds_addr(b, lds_offset),
                                           .align_mul = 4));
      nir_instr_remove(&intrin->instr);
      return true;
   }

   nir_def *ray_ref = callback_ray_ref(b);
   nir_def *addr = global_callback_slot_addr(b, state->config, slot, ray_ref);
   if (!addr)
      return false;
   if (load)
      nir_def_rewrite_uses(&intrin->def,
                           nir_load_global(b, intrin->def.num_components,
                                           intrin->def.bit_size, addr));
   else
      nir_store_global(b, intrin->src[0].ssa, addr, .align_mul = 4);
   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
lower_global_callback_launch_id(nir_builder *b, nir_intrinsic_instr *intrin,
                                void *data)
{
   (void)data;
   if (!is_ray_launch_id_load(intrin))
      return false;
   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *ray_ref = callback_ray_ref(b);
   nir_def *id = nir_vec3(
      b, load_completion_field(b, ray_ref, VT_RT_GLOBAL_COMPLETION_LAUNCH_ID_X),
      load_completion_field(b, ray_ref, VT_RT_GLOBAL_COMPLETION_LAUNCH_ID_Y),
      load_completion_field(b, ray_ref, VT_RT_GLOBAL_COMPLETION_LAUNCH_ID_Z));
   nir_def_rewrite_uses(&intrin->def, id);
   nir_instr_remove(&intrin->instr);
   return true;
}

bool
vt_nir_lower_rt_global_callback(nir_shader *shader,
                                const struct vt_nir_rt_config *config,
                                uint32_t *lds_size_bytes)
{
   if (lds_size_bytes)
      *lds_size_bytes = 0;
   if (!shader || !config ||
       config->execution_model != VT_RT_EXECUTION_WAVEFRONT_GLOBAL_CALLBACK ||
       (shader->info.stage != MESA_SHADER_MISS &&
        shader->info.stage != MESA_SHADER_CLOSEST_HIT &&
        shader->info.stage != MESA_SHADER_ANY_HIT &&
        shader->info.stage != MESA_SHADER_INTERSECTION))
      return false;

   NIR_PASS(_, shader, nir_opt_constant_folding);
   struct callback_lds_state lds = {
      .config = config,
   };
   prepare_callback_lds(shader, &lds);
   emit_callback_lds_prologue(shader, &lds);
   bool progress = false;
   progress |= nir_shader_intrinsics_pass(shader, lower_global_push_constants,
                                          nir_metadata_none, NULL);
   progress |= nir_shader_intrinsics_pass(shader, lower_global_callback_slot,
                                          nir_metadata_none, &lds);
   progress |= nir_shader_intrinsics_pass(shader, lower_global_callback_launch_id,
                                          nir_metadata_none, NULL);
   if (progress) {
      nir_opt_dead_cf(shader);
      nir_shader_gather_info(shader, nir_shader_get_entrypoint(shader));
   }
   if (lds_size_bytes)
      *lds_size_bytes = lds.lds_size_bytes;
   return progress;
}
