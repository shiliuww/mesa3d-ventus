/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"
#include "compiler/glsl_types.h"
#include "ventus/common/ventus_rt_v5.h"

#define VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT 0x8u

struct lower_rt_io_state {
   const struct vt_nir_rt_config *config;
   uint32_t trace_depth;
};

struct lower_global_root_continuation_state {
   const struct vt_nir_rt_config *config;
   struct set *lowered_traces;
};

struct lower_resident_continuation_state {
   const struct vt_nir_rt_config *config;
   struct set *lowered_traces;
   uint32_t next_site;
   bool failed;
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

static nir_def *
trace_ray_has_flag(nir_builder *b, nir_def *flags, uint32_t flag)
{
   return nir_ine_imm(b, nir_iand_imm(b, flags, flag), 0);
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

/* The remote V4 layout reserves one complete payload frame per active trace
 * level.  The local synchronous MegaKernel executes an inlined child trace in
 * that next frame, so marshal the frame at the call/return boundary. */
static void
copy_raygen_payload_level(nir_builder *b,
                          const struct vt_nir_rt_config *config,
                          uint32_t source_level, uint32_t dest_level)
{
   const struct vt_nir_scratchpad_layout *sp =
      &config->scratchpad[VT_RT_LAYOUT_RAYGEN];
   const uint32_t payload_size =
      sp->total_size_bytes - sp->payload_spill_size_bytes -
      sp->payload_base_bytes;

   for (uint32_t offset = 0; offset < payload_size;
        offset += VT_RT_ABI_WORD_BYTES) {
      nir_def *source = vt_nir_scratchpad_addr(
         b, config, VT_RT_LAYOUT_RAYGEN,
         sp->payload_base_bytes + source_level * payload_size + offset);
      nir_def *dest = vt_nir_scratchpad_addr(
         b, config, VT_RT_LAYOUT_RAYGEN,
         sp->payload_base_bytes + dest_level * payload_size + offset);
      nir_def *value = nir_ventus_rt_load_slot(b, 1, 32, source);
      nir_ventus_rt_store_slot(b, value, dest);
   }
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
                     nir_def *dynamic_skip_closest_hit,
                     bool constant_skip_closest_hit)
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
   if (config->emit_closest_hit && !constant_skip_closest_hit) {
      if (dynamic_skip_closest_hit)
         nir_push_if(b, nir_inot(b, dynamic_skip_closest_hit));
      config->emit_closest_hit(b, params, result, config->callback_data);
      if (dynamic_skip_closest_hit)
         nir_pop_if(b, NULL);
   }

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
         nir_ieq_imm(b, load_raygen_callback_decision(b, config),
                     VT_RT_CALLBACK_ACCEPT);
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

enum resident_frame_state {
   RESIDENT_FRAME_INACTIVE = 0,
   RESIDENT_FRAME_ROOT = 1,
   RESIDENT_FRAME_NESTED = 2,
};

static nir_def *
resident_frame_address(nir_builder *b, const struct vt_nir_rt_config *config,
                       uint32_t call_site)
{
   const uint64_t offset =
      (uint64_t)VENTUS_RT_V5_PDS_SIZE_BYTES +
      config->v5_continuation_scratch_bytes +
      (uint64_t)call_site *
         config->v5_continuation_state_frame_stride_bytes;
   assert(offset <= UINT32_MAX);
   return nir_ventus_rt_hot_pds_address(b, nir_imm_int(b, (uint32_t)offset));
}

static nir_def *
resident_global_load_u32(nir_builder *b, nir_def *base, uint32_t offset)
{
   return nir_load_global(
      b, 1, 32, nir_iadd_imm(b, nir_u2u64(b, base), offset),
      .align_mul = 4);
}

static void
resident_global_store_u32(nir_builder *b, nir_def *base, uint32_t offset,
                          nir_def *value)
{
   nir_store_global(b, nir_u2u32(b, value),
                    nir_iadd_imm(b, nir_u2u64(b, base), offset),
                    .write_mask = 0x1, .align_mul = 4);
}

static void
resident_copy_words(nir_builder *b, nir_def *src, uint32_t src_offset,
                    nir_def *dst, uint32_t dst_offset, uint32_t bytes)
{
   assert((bytes & 3) == 0);
   for (uint32_t offset = 0; offset < bytes; offset += sizeof(uint32_t)) {
      resident_global_store_u32(
         b, dst, dst_offset + offset,
         resident_global_load_u32(b, src, src_offset + offset));
   }
}

static void
resident_save_parent_pds(nir_builder *b,
                         const struct vt_nir_rt_config *config,
                         nir_def *pds, nir_def *frame)
{
   const uint32_t header = VENTUS_RT_V5_CONTINUATION_HEADER_BYTES;
   resident_copy_words(b, pds, 0, frame, header,
                       VENTUS_RT_V5_TRACE_PDS_PREFIX_BYTES);
   resident_copy_words(b, pds, VENTUS_RT_V5_PDS_COMMITTED_BASE, frame,
                       header + VENTUS_RT_V5_TRACE_PDS_PREFIX_BYTES,
                       VENTUS_RT_V5_TRACE_COMMITTED_BYTES);
   resident_copy_words(
      b, pds, VENTUS_RT_V5_PDS_CALL_DATA_BASE, frame,
      header + VENTUS_RT_V5_TRACE_PDS_PREFIX_BYTES +
         VENTUS_RT_V5_TRACE_COMMITTED_BYTES,
      config->v5_payload_size_bytes);
}

static bool
lower_trace_ray_v5(nir_builder *b, nir_intrinsic_instr *intrin,
                   struct lower_rt_io_state *state,
                   const struct vt_nir_rt_trace_params *params,
                   nir_def *dynamic_skip_closest_hit,
                   bool constant_skip_closest_hit)
{
   const struct vt_nir_rt_config *config = state->config;
   nir_def *launch_id = nir_load_ray_launch_id(b);
   nir_def *lane_id =
      nir_iand_imm(b, nir_channel(b, launch_id, 0),
                   VENTUS_RT_V5_LANES_PER_WARP - 1);
   nir_def *private_region_offset = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_PRIVATE_REGION_OFFSET, 32);
   nir_def *private_state_address = vt_nir_rt_v5_private_address(
      b, nir_imm_int(b, 0), private_region_offset, nir_imm_int(b, 0), lane_id);
   nir_def *pds_address =
      nir_ventus_rt_hot_pds_address(b, nir_imm_int(b, 0));

   const bool resident = config->use_v5_resident_continuations &&
                         intrin->intrinsic == nir_intrinsic_rt_trace_ray;
   struct vt_nir_rt_v5_submit_result submit = {
      .private_state_address = private_state_address,
   };
   if (resident) {
      const uint32_t call_site = nir_intrinsic_call_idx(intrin);
      nir_def *frame = resident_frame_address(b, config, call_site);
      nir_def *parent_depth = vt_nir_load_active_level(
         b, config, VT_RT_LAYOUT_RAYGEN);
      nir_def *max_depth = load_dispatch_arg(
         b, VT_NIR_RT_DISPATCH_ARG_MAX_RAY_RECURSION_DEPTH, 32);
      nir_def *depth_allowed = nir_ult(b, parent_depth, max_depth);
      resident_global_store_u32(
         b, frame, 0, nir_imm_int(b, RESIDENT_FRAME_INACTIVE));

      nir_push_if(b, depth_allowed);
      nir_def *has_parent = nir_ine_imm(b, parent_depth, 0);
      nir_push_if(b, has_parent);
      resident_save_parent_pds(b, config, pds_address, frame);
      resident_global_store_u32(
         b, frame, 0, nir_imm_int(b, RESIDENT_FRAME_NESTED));
      nir_push_else(b, NULL);
      resident_global_store_u32(
         b, frame, 0, nir_imm_int(b, RESIDENT_FRAME_ROOT));
      nir_pop_if(b, NULL);

      resident_global_store_u32(
         b, pds_address, VENTUS_RT_V5_PDS_CONTINUATION_FRAME_ADDRESS,
         nir_u2u32(b, frame));
      resident_global_store_u32(
         b, pds_address, VENTUS_RT_V5_PDS_CONTINUATION_FRAME_ADDRESS + 4,
         nir_unpack_64_2x32_split_y(b, nir_u2u64(b, frame)));
      vt_nir_store_active_level(
         b, config, VT_RT_LAYOUT_RAYGEN, nir_iadd_imm(b, parent_depth, 1));
      store_raygen_callback_decision(b, config, VT_RT_CALLBACK_PENDING);
      const struct vt_nir_rt_v5_submit_result admitted_submit =
         vt_nir_build_traversal_v5_submit(
            b, params, pds_address, private_state_address);
      nir_def *admitted_compact_result = admitted_submit.compact_result;
      nir_def *admitted_traverse = admitted_submit.traverse_admitted;
      nir_push_else(b, NULL);
      nir_def *rejected_compact_result = nir_imm_int(
         b, ventus_rt_v5_compact_result(VENTUS_RT_V5_RETURN_FAULT));
      nir_def *rejected_traverse = nir_imm_false(b);
      nir_pop_if(b, NULL);
      submit.compact_result =
         nir_if_phi(b, admitted_compact_result, rejected_compact_result);
      submit.traverse_admitted =
         nir_if_phi(b, admitted_traverse, rejected_traverse);
   } else {
      vt_nir_store_active_level(b, config, VT_RT_LAYOUT_RAYGEN,
                                nir_imm_int(b, state->trace_depth));
      store_raygen_callback_decision(b, config, VT_RT_CALLBACK_PENDING);
      submit = vt_nir_build_traversal_v5_submit(
         b, params, pds_address, private_state_address);
   }

   nir_variable *compact_result_var = nir_local_variable_create(
      b->impl, glsl_uint_type(), "ventus.v5.compact.result");
   nir_store_var(b, compact_result_var, submit.compact_result, 0x1);

   nir_loop *loop = nir_push_loop(b);
   {
      nir_def *compact_result = nir_load_var(b, compact_result_var);
      nir_def *reason = nir_iand_imm(b, compact_result, 0xff);
      nir_def *valid = nir_ine_imm(
         b, nir_iand_imm(b, compact_result, VENTUS_RT_V5_COMPLETION_VALID), 0);
      nir_def *complete_hit =
         nir_ieq_imm(b, reason, VENTUS_RT_V5_RETURN_CLOSEST_HIT_READY);
      nir_def *complete_miss =
         nir_ieq_imm(b, reason, VENTUS_RT_V5_RETURN_MISS);
      nir_def *hit_done =
         nir_ieq_imm(b, reason, VENTUS_RT_V5_RETURN_HIT_DONE_NO_SHADER);
      nir_def *candidate_non_opaque =
         nir_ieq_imm(b, reason, VENTUS_RT_V5_RETURN_ANY_HIT_REQUIRED);
      nir_def *candidate_procedural =
         nir_ieq_imm(b, reason, VENTUS_RT_V5_RETURN_INTERSECTION_REQUIRED);
      nir_def *candidate =
         nir_ior(b, candidate_non_opaque, candidate_procedural);
      nir_def *fault = nir_ior(
         b, nir_ieq_imm(b, reason, VENTUS_RT_V5_RETURN_FAULT),
         nir_ieq_imm(b, reason, VENTUS_RT_V5_RETURN_UNSUPPORTED));
      nir_def *terminal = nir_ior(
         b, nir_inot(b, valid),
         nir_ior(b, fault,
                 nir_ior(b, hit_done,
                         nir_ior(b, complete_hit, complete_miss))));

      nir_def *candidate_instance_sbt =
         vt_nir_load_candidate_hit_record_word(
            b, config, VT_RT_LAYOUT_RAYGEN,
            VT_RT_HIT_RECORD_INSTANCE_SBT_RECORD_OFFSET, 32);
      nir_def *committed_instance_sbt = vt_nir_load_hit_record_word(
         b, config, VT_RT_LAYOUT_RAYGEN,
         VT_RT_HIT_RECORD_INSTANCE_SBT_RECORD_OFFSET, 32);
      nir_def *instance_sbt = nir_bcsel(
         b, candidate, candidate_instance_sbt, committed_instance_sbt);
      nir_def *candidate_geometry = vt_nir_load_candidate_hit_record_word(
         b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_GEOMETRY_ID, 32);
      nir_def *committed_geometry = vt_nir_load_hit_record_word(
         b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_GEOMETRY_ID, 32);
      nir_def *geometry =
         nir_bcsel(b, candidate, candidate_geometry, committed_geometry);
      nir_def *sbt_index = nir_iadd(
         b, nir_iadd(b, instance_sbt,
                     nir_imul(b, geometry, nir_u2u32(b, params->sbt_stride))),
         nir_u2u32(b, params->sbt_offset));
      nir_def *hit_record_ptr = vt_nir_build_hit_shader_record_ptr(
         b, config, params, params->sbt_base, instance_sbt, geometry);
      nir_def *miss_record_ptr =
         vt_nir_build_miss_shader_record_ptr(b, config, params);
      nir_def *candidate_t = vt_nir_load_candidate_hit_record_word(
         b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_HIT_T, 32);
      nir_def *committed_t = vt_nir_load_hit_record_word(
         b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_HIT_T, 32);

      struct vt_nir_rt_result result = {
         .slot_base = pds_address,
         .status = reason,
         .hit = nir_ior(b, complete_hit, hit_done),
         .hit_t = nir_bcsel(b, candidate, candidate_t, committed_t),
         .sbt_index = sbt_index,
         .instance_sbt_record_offset = instance_sbt,
         .geometry_index = geometry,
         .hit_shader_record_ptr = hit_record_ptr,
         .miss_shader_record_ptr = miss_record_ptr,
         .shader_record_ptr = nir_bcsel(
            b, complete_miss, miss_record_ptr,
            nir_bcsel(b, nir_ior(b, complete_hit, candidate), hit_record_ptr,
                      nir_imm_int64(b, 0))),
      };

      emit_trace_callbacks(b, config, params, &result,
                           resident ? nir_imm_false(b) : complete_hit,
                           resident ? nir_imm_false(b) : complete_miss,
                           candidate_non_opaque,
                           candidate_procedural, dynamic_skip_closest_hit,
                           constant_skip_closest_hit);

      nir_push_if(b, terminal);
      nir_jump(b, nir_jump_break);
      nir_push_else(b, NULL);
      nir_push_if(b, nir_inot(b, candidate));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);

      nir_store_var(
         b, compact_result_var,
         vt_nir_build_traversal_v5_resume(
            b, pds_address, submit.private_state_address),
         0x1);
   }
   nir_pop_loop(b, loop);

   if (resident) {
      nir_def *final_result = nir_load_var(b, compact_result_var);
      resident_global_store_u32(
         b, pds_address, VENTUS_RT_V5_PDS_TRACE_RETURN_CONTROL,
         nir_iand_imm(b, final_result, 0xff));
   }

   nir_push_if(b, submit.traverse_admitted);
   nir_ventus_rt_release(b, pds_address);
   nir_pop_if(b, NULL);

   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
lower_trace_ray(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   if (intrin->intrinsic != nir_intrinsic_trace_ray &&
       intrin->intrinsic != nir_intrinsic_rt_trace_ray)
      return false;

   struct lower_rt_io_state *state = data;
   const struct vt_nir_rt_config *config = state->config;
   if (intrin->intrinsic == nir_intrinsic_rt_trace_ray &&
       (!config->use_v5_pds || !config->use_v5_resident_continuations))
      return false;
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
   };

   const bool constant_flags = nir_src_is_const(intrin->src[1]);
   const bool constant_skip_closest_hit =
      constant_flags &&
      (nir_src_as_uint(intrin->src[1]) & VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT);
   nir_def *dynamic_skip_closest_hit =
      constant_flags ? NULL :
                          trace_ray_has_flag(
                          b, params.flags, VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT);

   if (config->use_v5_pds)
      return lower_trace_ray_v5(
         b, intrin, state, &params, dynamic_skip_closest_hit,
         constant_skip_closest_hit);

   if (state->trace_depth > 0)
      copy_raygen_payload_level(b, config, state->trace_depth - 1,
                                state->trace_depth);
   vt_nir_store_active_level(b, config, VT_RT_LAYOUT_RAYGEN,
                             nir_imm_int(b, state->trace_depth));
   store_raygen_callback_decision(b, config, VT_RT_CALLBACK_PENDING);

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
                           candidate_procedural, dynamic_skip_closest_hit,
                           constant_skip_closest_hit);

      nir_push_if(b, terminal);
      nir_jump(b, nir_jump_break);
      nir_push_else(b, NULL);

      nir_push_if(b, nir_inot(b, candidate));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, loop);

   if (state->trace_depth > 0)
      copy_raygen_payload_level(b, config, state->trace_depth,
                                state->trace_depth - 1);
   vt_nir_store_active_level(
      b, config, VT_RT_LAYOUT_RAYGEN,
      nir_imm_int(b, state->trace_depth > 0 ? state->trace_depth - 1 : 0));

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
   const bool constant_flags = nir_src_is_const(intrin->src[1]);
   const bool constant_skip_closest_hit =
      constant_flags &&
      (nir_src_as_uint(intrin->src[1]) & VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT);
   nir_def *dynamic_skip_closest_hit =
      constant_flags ? NULL :
                       trace_ray_has_flag(
                          b, params.flags, VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT);
   emit_trace_callbacks(b, config, &params, &result, complete_hit,
                        complete_miss, nir_imm_false(b), nir_imm_false(b),
                        dynamic_skip_closest_hit, constant_skip_closest_hit);
   return true;
}

static bool
lower_resident_continuation(nir_builder *b, nir_intrinsic_instr *intrin,
                            void *data)
{
   if (intrin->intrinsic != nir_intrinsic_trace_ray)
      return false;

   struct lower_resident_continuation_state *state = data;
   if (_mesa_set_search(state->lowered_traces, intrin))
      return false;
   _mesa_set_add(state->lowered_traces, intrin);

   if (!nir_src_is_deref(intrin->src[10])) {
      state->failed = true;
      return false;
   }
   nir_variable *payload =
      nir_deref_instr_get_variable(nir_src_as_deref(intrin->src[10]));
   if (!payload || !payload->data.explicit_location ||
       payload->data.location < 0) {
      state->failed = true;
      return false;
   }
   unsigned payload_size = 0;
   unsigned payload_align = 0;
   glsl_get_natural_size_align_bytes(payload->type, &payload_size,
                                     &payload_align);
   const uint64_t payload_offset =
      (uint64_t)(uint32_t)payload->data.location * VT_RT_ABI_ALIGNMENT_BYTES;
   const uint64_t payload_end = payload_offset + ALIGN_POT(payload_size, 4);
   if (payload_size == 0 || payload_align == 0 ||
       payload_end > VENTUS_RT_V5_CALL_DATA_CAPACITY_BYTES) {
      state->failed = true;
      return false;
   }

   const uint32_t continuation_token = state->next_site++;
   b->cursor = nir_before_instr(&intrin->instr);
   nir_ventus_rt_continuation_begin(
      b, nir_imm_int(b, continuation_token));

   const struct vt_nir_rt_config *config = state->config;
   b->cursor = nir_after_instr(&intrin->instr);
   struct vt_nir_rt_trace_params params = {
      .accel_struct = intrin->src[0].ssa,
      .flags = intrin->src[1].ssa,
      .cull_mask = intrin->src[2].ssa,
      .sbt_base = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HIT_SBT_ADDR, 64),
      .sbt_stride_bytes =
         load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HIT_SBT_STRIDE, 64),
      .miss_sbt_base =
         load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_MISS_SBT_ADDR, 64),
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

   nir_def *pds_address =
      nir_ventus_rt_hot_pds_address(b, nir_imm_int(b, 0));
   nir_def *return_control = nir_load_global(
      b, 1, 32,
      nir_iadd_imm(b, nir_u2u64(b, pds_address),
                   VENTUS_RT_V5_PDS_TRACE_RETURN_CONTROL),
      .align_mul = 4);
   nir_def *reason = nir_iand_imm(b, return_control, 0xff);
   nir_def *complete_hit =
      nir_ieq_imm(b, reason, VENTUS_RT_V5_RETURN_CLOSEST_HIT_READY);
   nir_def *complete_miss =
      nir_ieq_imm(b, reason, VENTUS_RT_V5_RETURN_MISS);
   nir_def *instance_sbt = vt_nir_load_hit_record_word(
      b, config, VT_RT_LAYOUT_RAYGEN,
      VT_RT_HIT_RECORD_INSTANCE_SBT_RECORD_OFFSET, 32);
   nir_def *geometry = vt_nir_load_hit_record_word(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_GEOMETRY_ID, 32);
   nir_def *hit_record_ptr = vt_nir_build_hit_shader_record_ptr(
      b, config, &params, params.sbt_base, instance_sbt, geometry);
   nir_def *miss_record_ptr =
      vt_nir_build_miss_shader_record_ptr(b, config, &params);
   struct vt_nir_rt_result result = {
      .slot_base = pds_address,
      .status = reason,
      .hit = complete_hit,
      .hit_t = vt_nir_load_hit_record_word(
         b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_HIT_T, 32),
      .sbt_index = nir_iadd(
         b, nir_iadd(b, instance_sbt,
                     nir_imul(b, geometry, nir_u2u32(b, params.sbt_stride))),
         nir_u2u32(b, params.sbt_offset)),
      .instance_sbt_record_offset = instance_sbt,
      .geometry_index = geometry,
      .hit_shader_record_ptr = hit_record_ptr,
      .miss_shader_record_ptr = miss_record_ptr,
      .shader_record_ptr =
         nir_bcsel(b, complete_miss, miss_record_ptr, hit_record_ptr),
   };
   const bool constant_flags = nir_src_is_const(intrin->src[1]);
   const bool constant_skip_closest_hit =
      constant_flags &&
      (nir_src_as_uint(intrin->src[1]) & VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT);
   nir_def *dynamic_skip_closest_hit =
      constant_flags ? NULL :
                       trace_ray_has_flag(
                          b, params.flags, VT_SPV_RAY_FLAGS_SKIP_CLOSEST_HIT);
   emit_trace_callbacks(b, config, &params, &result, complete_hit,
                        complete_miss, nir_imm_false(b), nir_imm_false(b),
                        dynamic_skip_closest_hit,
                        constant_skip_closest_hit);

   nir_ventus_rt_continuation_return(
      b, nir_imm_int(b, continuation_token),
      nir_imm_int(b, (uint32_t)payload_offset),
      nir_imm_int(b, ALIGN_POT(payload_size, 4)));
   return true;
}

bool
vt_nir_prepare_rt_resident_continuations(
   nir_shader *shader, const struct vt_nir_rt_config *config)
{
   if (!shader || !config || !config->use_v5_pds ||
       config->execution_model != VT_RT_EXECUTION_ITERATIVE_RAYGEN_LOOP ||
       !config->enable_monolithic_shader_callbacks)
      return false;

   struct lower_resident_continuation_state state = {
      .config = config,
      .lowered_traces = _mesa_set_create(NULL, _mesa_hash_pointer,
                                         _mesa_key_pointer_equal),
   };
   if (!state.lowered_traces)
      return false;

   bool progress = false;
   for (uint32_t i = 0; i < 8; i++) {
      const bool pass_progress = nir_shader_intrinsics_pass(
         shader, lower_resident_continuation, nir_metadata_none, &state);
      progress |= pass_progress;
      if (state.failed || !pass_progress)
         break;
   }
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_trace_ray &&
             !_mesa_set_search(state.lowered_traces,
                               nir_instr_as_intrinsic(instr))) {
            state.failed = true;
            break;
         }
      }
      if (state.failed)
         break;
   }
   _mesa_set_destroy(state.lowered_traces, NULL);
   if (!progress || state.failed)
      return false;

   nir_index_blocks(impl);
   nir_index_ssa_defs(impl);
   nir_shader_gather_info(shader, impl);
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

struct materialize_resident_state {
   const struct vt_nir_rt_config *config;
   const struct vt_nir_rt_resident_continuation_info *info;
   uint32_t *token_to_call;
   bool failed;
};

static nir_def *
resident_scratch_address(nir_builder *b, nir_def *offset)
{
   nir_def *logical_offset = nir_iadd_imm(
      b, nir_u2u32(b, offset), VENTUS_RT_V5_PDS_SIZE_BYTES);
   return nir_u2u64(
      b, nir_ventus_rt_hot_pds_address(b, logical_offset));
}

static bool
materialize_resident_intrinsic(nir_builder *b,
                               nir_intrinsic_instr *intrin, void *data)
{
   struct materialize_resident_state *state = data;
   switch (intrin->intrinsic) {
   case nir_intrinsic_ventus_rt_continuation_begin:
      nir_instr_remove(&intrin->instr);
      return true;
   case nir_intrinsic_load_scratch: {
      b->cursor = nir_before_instr(&intrin->instr);
      nir_def *value = nir_load_global(
         b, intrin->def.num_components, intrin->def.bit_size,
         resident_scratch_address(b, intrin->src[0].ssa),
         .align_mul = nir_intrinsic_align_mul(intrin),
         .align_offset = nir_intrinsic_align_offset(intrin));
      nir_def_rewrite_uses(&intrin->def, value);
      nir_instr_remove(&intrin->instr);
      return true;
   }
   case nir_intrinsic_store_scratch:
      b->cursor = nir_before_instr(&intrin->instr);
      nir_store_global(
         b, intrin->src[0].ssa,
         resident_scratch_address(b, intrin->src[1].ssa),
         .write_mask = nir_intrinsic_write_mask(intrin),
         .align_mul = nir_intrinsic_align_mul(intrin),
         .align_offset = nir_intrinsic_align_offset(intrin));
      nir_instr_remove(&intrin->instr);
      return true;
   case nir_intrinsic_rt_resume:
      nir_instr_remove(&intrin->instr);
      return true;
   case nir_intrinsic_ventus_rt_continuation_return:
      break;
   default:
      return false;
   }

   if (!nir_src_is_const(intrin->src[0]) ||
       !nir_src_is_const(intrin->src[1]) ||
       !nir_src_is_const(intrin->src[2])) {
      state->failed = true;
      return false;
   }
   const uint32_t token = nir_src_as_uint(intrin->src[0]);
   const uint32_t payload_offset = nir_src_as_uint(intrin->src[1]);
   const uint32_t payload_bytes = nir_src_as_uint(intrin->src[2]);
   if (token >= state->info->site_count || (payload_offset & 3) != 0 ||
       (payload_bytes & 3) != 0 || payload_bytes == 0 ||
       payload_offset + payload_bytes > state->config->v5_payload_size_bytes ||
       payload_bytes > VENTUS_RT_V5_CALL_DATA_CAPACITY_BYTES) {
      state->failed = true;
      return false;
   }
   const uint32_t call_site = state->token_to_call[token];
   if (call_site >= state->info->site_count) {
      state->failed = true;
      return false;
   }

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *pds = nir_ventus_rt_hot_pds_address(b, nir_imm_int(b, 0));
   nir_def *frame = resident_frame_address(b, state->config, call_site);
   nir_def *frame_state = resident_global_load_u32(b, frame, 0);
   nir_def *child_payload[VENTUS_RT_V5_CALL_DATA_CAPACITY_BYTES / 4];
   const uint32_t payload_words = payload_bytes / sizeof(uint32_t);
   for (uint32_t word = 0; word < payload_words; word++) {
      child_payload[word] = resident_global_load_u32(
         b, pds, VENTUS_RT_V5_PDS_CALL_DATA_BASE + payload_offset + word * 4);
   }

   nir_push_if(b, nir_ieq_imm(b, frame_state, RESIDENT_FRAME_NESTED));
   const uint32_t header = VENTUS_RT_V5_CONTINUATION_HEADER_BYTES;
   resident_copy_words(b, frame, header, pds, 0,
                       VENTUS_RT_V5_TRACE_PDS_PREFIX_BYTES);
   resident_copy_words(b, frame,
                       header + VENTUS_RT_V5_TRACE_PDS_PREFIX_BYTES,
                       pds, VENTUS_RT_V5_PDS_COMMITTED_BASE,
                       VENTUS_RT_V5_TRACE_COMMITTED_BYTES);
   resident_copy_words(
      b, frame,
      header + VENTUS_RT_V5_TRACE_PDS_PREFIX_BYTES +
         VENTUS_RT_V5_TRACE_COMMITTED_BYTES,
      pds, VENTUS_RT_V5_PDS_CALL_DATA_BASE,
      state->config->v5_payload_size_bytes);
   for (uint32_t word = 0; word < payload_words; word++) {
      resident_global_store_u32(
         b, pds,
         VENTUS_RT_V5_PDS_CALL_DATA_BASE + payload_offset + word * 4,
         child_payload[word]);
   }
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, frame_state, RESIDENT_FRAME_ROOT));
   resident_global_store_u32(
      b, pds, VENTUS_RT_V5_PDS_CONTINUATION_FRAME_ADDRESS,
      nir_imm_int(b, 0));
   resident_global_store_u32(
      b, pds, VENTUS_RT_V5_PDS_CONTINUATION_FRAME_ADDRESS + 4,
      nir_imm_int(b, 0));
   vt_nir_store_active_level(
      b, state->config, VT_RT_LAYOUT_RAYGEN, nir_imm_int(b, 0));
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);

   nir_instr_remove(&intrin->instr);
   return true;
}

bool
vt_nir_materialize_rt_resident_continuations(
   nir_shader *shader, const struct vt_nir_rt_config *config,
   const struct vt_nir_rt_resident_continuation_info *info)
{
   if (!shader || !config || !info || !config->use_v5_pds ||
       !config->use_v5_resident_continuations ||
       info->state_frame_stride_bytes == 0 || !info->token_to_call)
      return false;
   struct materialize_resident_state state = {
      .config = config,
      .info = info,
      .token_to_call = info->token_to_call,
   };
   const bool progress = nir_shader_intrinsics_pass(
      shader, materialize_resident_intrinsic, nir_metadata_none, &state);
   return progress && !state.failed;
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
         nir_ventus_rt_store_slot(
            &b, nir_imm_int(&b, VT_RT_SLOT_STATUS_TERMINATED), slot_base);
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
      state.trace_depth = i;
      bool pass_progress =
         nir_shader_intrinsics_pass(shader, lower_trace_ray, nir_metadata_none,
                                    &state);
      progress |= pass_progress;
      if (!pass_progress)
         break;

   }

   if (progress && !config->use_v5_pds)
      progress |= insert_raygen_release_before_returns(shader, config);

   if (progress && config->use_v5_pds)
      nir_lower_vars_to_ssa(shader);

   if (progress) {
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      if (impl) {
         nir_index_blocks(impl);
         nir_index_ssa_defs(impl);
      }
   }

   return progress;
}
