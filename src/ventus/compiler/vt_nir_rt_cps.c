/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

#include <limits.h>
#include <string.h>

#include "compiler/glsl_types.h"
#include "nir/nir_deref.h"
#include "util/ralloc.h"
#include "ventus/common/ventus_kernel_resource.h"
#include "ventus/common/ventus_rt_v5.h"

static nir_def *
stack_load_u32(nir_builder *b, nir_def *base, uint32_t offset)
{
   return nir_load_global(b, 1, 32,
                          nir_iadd_imm(b, nir_u2u64(b, base), offset),
                          .align_mul = 4);
}

static void
stack_store_u32(nir_builder *b, nir_def *base, uint32_t offset,
                nir_def *value)
{
   nir_store_global(b, nir_u2u32(b, value),
                    nir_iadd_imm(b, nir_u2u64(b, base), offset),
                    .write_mask = 0x1, .align_mul = 4);
}

nir_def *
vt_nir_rt_v5_continuation_stack_base(nir_builder *b)
{
   return nir_ventus_rt_continuation_stack_address(b);
}

nir_def *
vt_nir_rt_v5_current_continuation_frame(nir_builder *b)
{
   nir_def *lo = nir_ventus_rt_load_pds_field(
      b, .base = VT_NIR_RT_PDS_FIELD_CONTINUATION_FRAME, .range_base = 0);
   nir_def *hi = nir_ventus_rt_load_pds_field(
      b, .base = VT_NIR_RT_PDS_FIELD_CONTINUATION_FRAME, .range_base = 4);
   return nir_pack_64_2x32_split(b, lo, hi);
}

static void
store_current_continuation_frame(nir_builder *b, nir_def *frame)
{
   nir_ventus_rt_store_pds_field(
      b, nir_unpack_64_2x32_split_x(b, frame),
      .base = VT_NIR_RT_PDS_FIELD_CONTINUATION_FRAME, .range_base = 0);
   nir_ventus_rt_store_pds_field(
      b, nir_unpack_64_2x32_split_y(b, frame),
      .base = VT_NIR_RT_PDS_FIELD_CONTINUATION_FRAME, .range_base = 4);
}

void
vt_nir_rt_v5_load_stack_cursor(nir_builder *b,
                               struct vt_nir_rt_stack_cursor *cursor)
{
   memset(cursor, 0, sizeof(*cursor));
   cursor->base = vt_nir_rt_v5_continuation_stack_base(b);
   cursor->stack_pointer = stack_load_u32(
      b, cursor->base, VT_NIR_RT_STACK_SP_OFFSET);
   cursor->callable_depth = stack_load_u32(
      b, cursor->base, VT_NIR_RT_STACK_CALLABLE_DEPTH_OFFSET);
   cursor->frame_stride = stack_load_u32(
      b, cursor->base, VT_NIR_RT_STACK_FRAME_STRIDE_OFFSET);
   cursor->frame_capacity = stack_load_u32(
      b, cursor->base, VT_NIR_RT_STACK_FRAME_CAPACITY_OFFSET);
   cursor->callable_capacity = stack_load_u32(
      b, cursor->base, VT_NIR_RT_STACK_CALLABLE_CAPACITY_OFFSET);
   cursor->next_frame = nir_iadd(
      b, nir_iadd_imm(b, cursor->base, VT_NIR_RT_STACK_HEADER_BYTES),
      nir_u2u64(b, nir_imul(b, cursor->stack_pointer,
                            cursor->frame_stride)));
}

void
vt_nir_rt_v5_publish_stack_push(
   nir_builder *b, const struct vt_nir_rt_stack_cursor *cursor,
   nir_def *frame_kind, nir_def *site_id,
   nir_def *record_ptr, bool callable)
{
   stack_store_u32(b, cursor->next_frame, VT_NIR_RT_FRAME_KIND_OFFSET,
                   frame_kind);
   stack_store_u32(b, cursor->next_frame, VT_NIR_RT_FRAME_SITE_OFFSET,
                   site_id ? site_id : nir_imm_int(b, 0));
   nir_store_global(
      b, record_ptr ? nir_u2u64(b, record_ptr) : nir_imm_int64(b, 0),
      nir_iadd_imm(b, cursor->next_frame,
                   VT_NIR_RT_FRAME_RECORD_PTR_OFFSET),
      .write_mask = 0x1, .align_mul = 8);
   stack_store_u32(b, cursor->base, VT_NIR_RT_STACK_SP_OFFSET,
                   nir_iadd_imm(b, cursor->stack_pointer, 1));
   if (callable) {
      stack_store_u32(b, cursor->base,
                      VT_NIR_RT_STACK_CALLABLE_DEPTH_OFFSET,
                      nir_iadd_imm(b, cursor->callable_depth, 1));
   }

   store_current_continuation_frame(b, cursor->next_frame);
}

void
vt_nir_rt_v5_publish_stack_pop(nir_builder *b, bool callable)
{
   struct vt_nir_rt_stack_cursor cursor;
   vt_nir_rt_v5_load_stack_cursor(b, &cursor);
   nir_def *new_sp = nir_iadd_imm(b, cursor.stack_pointer, -1);
   stack_store_u32(b, cursor.base, VT_NIR_RT_STACK_SP_OFFSET, new_sp);
   if (callable) {
      stack_store_u32(b, cursor.base,
                      VT_NIR_RT_STACK_CALLABLE_DEPTH_OFFSET,
                      nir_iadd_imm(b, cursor.callable_depth, -1));
   }

   nir_def *previous = nir_iadd(
      b, nir_iadd_imm(b, cursor.base, VT_NIR_RT_STACK_HEADER_BYTES),
      nir_u2u64(b, nir_imul(
                      b, nir_iadd_imm(b, new_sp, -1), cursor.frame_stride)));
   previous = nir_bcsel(b, nir_ine_imm(b, new_sp, 0), previous,
                        nir_imm_int64(b, 0));
   store_current_continuation_frame(b, previous);
}

bool
vt_nir_rt_v5_initialize_continuation_stack(
   nir_shader *shader, uint32_t frame_stride, uint32_t frame_capacity,
   uint32_t callable_capacity)
{
   if (!shader || frame_stride < VT_NIR_RT_CALLABLE_FRAME_BYTES ||
       (frame_stride & (VT_NIR_RT_STACK_FRAME_ALIGNMENT_BYTES - 1)) ||
       frame_capacity == 0 || callable_capacity > frame_capacity)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;
   nir_builder b = nir_builder_at(nir_before_cf_list(&impl->body));
   nir_def *base = vt_nir_rt_v5_continuation_stack_base(&b);
   stack_store_u32(&b, base, VT_NIR_RT_STACK_SP_OFFSET, nir_imm_int(&b, 0));
   stack_store_u32(&b, base, VT_NIR_RT_STACK_CALLABLE_DEPTH_OFFSET,
                   nir_imm_int(&b, 0));
   stack_store_u32(&b, base, VT_NIR_RT_STACK_FRAME_STRIDE_OFFSET,
                   nir_imm_int(&b, frame_stride));
   stack_store_u32(&b, base, VT_NIR_RT_STACK_FRAME_CAPACITY_OFFSET,
                   nir_imm_int(&b, frame_capacity));
   stack_store_u32(&b, base, VT_NIR_RT_STACK_CALLABLE_CAPACITY_OFFSET,
                   nir_imm_int(&b, callable_capacity));

   store_current_continuation_frame(&b, nir_imm_int64(&b, 0));
   return true;
}

static bool
remove_nonresident_call_for_analysis(nir_builder *b,
                                     nir_intrinsic_instr *intrin, void *data)
{
   (void)data;
   if (intrin->intrinsic != nir_intrinsic_report_ray_intersection)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def_rewrite_uses(&intrin->def, nir_imm_false(b));
   nir_instr_remove(&intrin->instr);
   return true;
}

static bool
align_u32(uint32_t value, uint32_t alignment, uint32_t *result)
{
   uint64_t aligned = 0;
   if (!ventus_rt_v5_align_u64(value, alignment, &aligned) ||
       aligned > UINT32_MAX)
      return false;
   *result = (uint32_t)aligned;
   return true;
}

bool
vt_nir_rt_v5_compute_continuation_stack_layout(
   uint32_t frame_stride, uint32_t max_active_trace_frames,
   uint32_t max_active_callable_frames, bool reserve_trace_guard_frame,
   uint32_t *physical_frame_capacity, uint32_t *stack_bytes_per_thread)
{
   if (!physical_frame_capacity || !stack_bytes_per_thread ||
       frame_stride == 0 ||
       (frame_stride & (VT_NIR_RT_STACK_FRAME_ALIGNMENT_BYTES - 1)))
      return false;

   const uint64_t guard_frames =
      reserve_trace_guard_frame && max_active_trace_frames ? 1 : 0;
   const uint64_t capacity = (uint64_t)max_active_trace_frames +
                             max_active_callable_frames + guard_frames;
   const uint64_t bytes = VT_NIR_RT_STACK_HEADER_BYTES +
                          capacity * frame_stride;
   if (capacity == 0 || capacity > UINT32_MAX || bytes > UINT32_MAX)
      return false;

   uint32_t aligned_bytes = 0;
   if (!align_u32((uint32_t)bytes,
                  VT_NIR_RT_STACK_FRAME_ALIGNMENT_BYTES, &aligned_bytes))
      return false;
   *physical_frame_capacity = (uint32_t)capacity;
   *stack_bytes_per_thread = aligned_bytes;
   return true;
}

static bool
resident_state_frame_stride(uint32_t parent_pds_save_bytes,
                            uint32_t payload_size_bytes, uint32_t *stride)
{
   const uint64_t bytes = VENTUS_RT_V5_CONTINUATION_HEADER_BYTES +
                          parent_pds_save_bytes +
                          payload_size_bytes;
   return bytes <= UINT32_MAX &&
          align_u32((uint32_t)bytes,
                    VT_NIR_RT_STACK_FRAME_ALIGNMENT_BYTES, stride);
}

static bool
is_rt_parent_state_system_value(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_object_origin:
   case nir_intrinsic_load_ray_object_direction:
   case nir_intrinsic_load_ray_t_min:
   case nir_intrinsic_load_ray_t_max:
   case nir_intrinsic_load_ray_object_to_world:
   case nir_intrinsic_load_ray_world_to_object:
   case nir_intrinsic_load_ray_hit_kind:
   case nir_intrinsic_load_ray_flags:
   case nir_intrinsic_load_ray_geometry_index:
   case nir_intrinsic_load_ray_instance_custom_index:
   case nir_intrinsic_load_shader_record_ptr:
   case nir_intrinsic_load_cull_mask:
   case nir_intrinsic_load_ray_triangle_vertex_positions:
   case nir_intrinsic_load_primitive_id:
   case nir_intrinsic_load_instance_id:
   case nir_intrinsic_load_front_face:
   case nir_intrinsic_load_front_face_fsign:
   case nir_intrinsic_ventus_rt_load_hit_attrib:
      return true;
   default:
      return false;
   }
}

struct parent_hit_attrib_state {
   const struct vt_nir_rt_config *config;
};

static bool
is_parent_barycentric_var(const struct parent_hit_attrib_state *state,
                          const nir_variable *var)
{
   if (!state->config->scratchpad[VT_RT_LAYOUT_ANY_HIT]
           .hit_attrib_barycentrics ||
       var->data.driver_location != 0)
      return false;

   unsigned size = 0, align = 0;
   glsl_get_natural_size_align_bytes(var->type, &size, &align);
   return size >= 8 && align >= 4;
}

static nir_def *
load_parent_hit_attrib_word(nir_builder *b, uint32_t byte_offset,
                            bool barycentrics)
{
   if (barycentrics && byte_offset >= 8)
      return nir_imm_int(b, 0);

   return nir_ventus_rt_load_hit_attrib(
      b, .base = byte_offset, .component = barycentrics);
}

static bool
lower_parent_hit_attrib_load(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   if (intrin->intrinsic != nir_intrinsic_load_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   if (!deref || !nir_deref_mode_is(deref, nir_var_ray_hit_attrib) ||
       deref->deref_type == nir_deref_type_cast)
      return false;

   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (!var)
      return false;

   struct parent_hit_attrib_state *state = data;
   const uint32_t location =
      var->data.driver_location +
      nir_deref_instr_get_const_offset(deref,
                                       glsl_get_natural_size_align_bytes);
   const bool barycentrics = is_parent_barycentric_var(state, var);

   b->cursor = nir_after_instr(instr);
   nir_def *components[NIR_MAX_VEC_COMPONENTS];
   for (uint32_t comp = 0; comp < intrin->def.num_components; comp++) {
      const uint32_t component_offset =
         location + comp * DIV_ROUND_UP(intrin->def.bit_size, 8);
      const uint32_t word_offset = component_offset & ~3u;
      const uint32_t byte_in_word = component_offset & 3u;

      if (intrin->def.bit_size == 64) {
         components[comp] = nir_pack_64_2x32_split(
            b, load_parent_hit_attrib_word(b, word_offset, barycentrics),
            load_parent_hit_attrib_word(b, word_offset + 4,
                                        barycentrics));
      } else if (intrin->def.bit_size == 32) {
         components[comp] =
            load_parent_hit_attrib_word(b, word_offset, barycentrics);
      } else if (intrin->def.bit_size == 16) {
         components[comp] = nir_channel(
            b, nir_unpack_32_2x16(
                  b, load_parent_hit_attrib_word(b, word_offset,
                                                 barycentrics)),
            byte_in_word / 2);
      } else if (intrin->def.bit_size == 8) {
         components[comp] = nir_channel(
            b, nir_unpack_bits(
                  b, load_parent_hit_attrib_word(b, word_offset,
                                                 barycentrics),
                  8),
            byte_in_word);
      } else {
         assert(intrin->def.bit_size == 1);
         components[comp] = nir_i2b(
            b, load_parent_hit_attrib_word(b, word_offset, barycentrics));
      }
   }

   nir_def_rewrite_uses(
      &intrin->def,
      nir_vec(b, components, intrin->def.num_components));
   nir_instr_remove(instr);
   return true;
}

static bool
prepare_parent_hit_attrib_ssa(nir_shader *shader,
                              const struct vt_nir_rt_config *config,
                              bool *complete)
{
   bool progress = false;
   progress |= nir_lower_vars_to_explicit_types(
      shader, nir_var_ray_hit_attrib, glsl_get_natural_size_align_bytes);
   progress |= nir_lower_indirect_derefs_to_if_else_trees(
      shader, nir_var_ray_hit_attrib, UINT32_MAX);

   struct parent_hit_attrib_state state = {.config = config};
   progress |= nir_shader_instructions_pass(
      shader, lower_parent_hit_attrib_load, nir_metadata_control_flow,
      &state);
   if (progress)
      nir_remove_dead_derefs(shader);

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (nir_deref_mode_is(deref, nir_var_ray_hit_attrib))
               *complete = false;
         }
      }
   }
   return progress;
}

bool
vt_nir_opt_rt_parent_state_ssa(nir_shader *shader,
                               const struct vt_nir_rt_config *config,
                               bool *complete)
{
   if (!shader || !config || !complete)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   bool has_trace = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_trace_ray) {
            has_trace = true;
            break;
         }
      }
      if (has_trace)
         break;
   }

   *complete = true;
   if (!has_trace)
      return true;

   bool progress = prepare_parent_hit_attrib_ssa(shader, config, complete);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (is_rt_parent_state_system_value(intrin->intrinsic)) {
            /* These values describe the current shader invocation.  Moving
             * them before its first child trace makes Mesa's existing CPS
             * liveness pass spill only values used after that trace. */
            nir_instr_remove(instr);
            nir_instr_insert(nir_before_impl(impl), instr);
            progress = true;
            continue;
         }

         if (intrin->intrinsic == nir_intrinsic_load_deref &&
             nir_src_is_deref(intrin->src[0])) {
            nir_variable *var = nir_deref_instr_get_variable(
               nir_src_as_deref(intrin->src[0]));
            if (var && var->data.mode == nir_var_ray_hit_attrib)
               *complete = false;
         }

         /* Direct PDS access is outside the system-value proof.  Preserve the
          * old full snapshot unless a later compiler pass gives it an
          * explicit SSA lifetime. */
         if (intrin->intrinsic ==
                nir_intrinsic_ventus_rt_hot_pds_address ||
             intrin->intrinsic == nir_intrinsic_ventus_rt_load_slot ||
             intrin->intrinsic == nir_intrinsic_ventus_rt_store_slot)
            *complete = false;
      }
   }

   if (progress) {
      nir_progress(true, impl, nir_metadata_control_flow);
      NIR_PASS(_, shader, nir_opt_cse);
   }
   return true;
}

static bool
count_trace_calls(nir_shader *shader, uint32_t *count)
{
   uint32_t calls = 0;
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         switch (nir_instr_as_intrinsic(instr)->intrinsic) {
         case nir_intrinsic_trace_ray:
            calls++;
            break;
         case nir_intrinsic_execute_callable:
         case nir_intrinsic_report_ray_intersection:
            return false;
         default:
            break;
         }
      }
   }
   *count = calls;
   return true;
}

static bool
resident_site_frame_size(enum vt_nir_rt_continuation_kind kind,
                         uint32_t ssa_spill_bytes,
                         uint32_t parent_pds_save_bytes,
                         uint32_t payload_size_bytes,
                         uint32_t *frame_size_bytes)
{
   uint64_t fixed_bytes = VENTUS_RT_V5_CONTINUATION_HEADER_BYTES;
   if (kind == VT_NIR_RT_TRACE_CONTINUATION)
      fixed_bytes += parent_pds_save_bytes;
   fixed_bytes += payload_size_bytes;
   if (fixed_bytes > UINT32_MAX - ssa_spill_bytes)
      return false;
   return align_u32((uint32_t)fixed_bytes + ssa_spill_bytes,
                    VT_NIR_RT_STACK_FRAME_ALIGNMENT_BYTES,
                    frame_size_bytes);
}

static bool
collect_resident_sites(
   nir_shader *shader, uint32_t ordinary_scratch_bytes,
   uint32_t payload_size_bytes, uint32_t callable_data_size_bytes,
   uint32_t parent_pds_save_bytes,
   struct vt_nir_rt_resident_continuation_info *info, bool *seen)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         enum vt_nir_rt_continuation_kind kind;
         switch (intrin->intrinsic) {
         case nir_intrinsic_rt_trace_ray:
            kind = VT_NIR_RT_TRACE_CONTINUATION;
            break;
         case nir_intrinsic_rt_execute_callable:
            kind = VT_NIR_RT_CALLABLE_CONTINUATION;
            break;
         default:
            continue;
         }

         const uint32_t call_index = nir_intrinsic_call_idx(intrin);
         const uint32_t stack_size = nir_intrinsic_stack_size(intrin);
         if (call_index >= info->site_count ||
             stack_size < ordinary_scratch_bytes)
            return false;
         const uint32_t ssa_spill_bytes =
            stack_size - ordinary_scratch_bytes;
         uint32_t frame_size_bytes = 0;
         if (!resident_site_frame_size(kind, ssa_spill_bytes,
                                       kind == VT_NIR_RT_TRACE_CONTINUATION ?
                                          parent_pds_save_bytes : 0,
                                       payload_size_bytes,
                                       &frame_size_bytes))
            return false;

         struct vt_nir_rt_continuation_site *site = &info->sites[call_index];
         if (seen[call_index]) {
            if (site->kind != kind ||
                site->ssa_spill_bytes != ssa_spill_bytes ||
                site->frame_size_bytes != frame_size_bytes)
               return false;
            continue;
         }
         seen[call_index] = true;
         *site = (struct vt_nir_rt_continuation_site) {
            .kind = kind,
            .call_index = call_index,
            .ssa_spill_bytes = ssa_spill_bytes,
            .parent_pds_save_bytes =
               kind == VT_NIR_RT_TRACE_CONTINUATION ?
                  parent_pds_save_bytes : 0,
            .parent_payload_save_bytes = payload_size_bytes,
            .return_data_bytes =
               kind == VT_NIR_RT_TRACE_CONTINUATION ?
                  payload_size_bytes : callable_data_size_bytes,
            .frame_size_bytes = frame_size_bytes,
         };
         if (kind == VT_NIR_RT_TRACE_CONTINUATION) {
            info->feature_flags |=
               VENTUS_RT_RESOURCE_USES_TRAVERSAL |
               VENTUS_RT_RESOURCE_USES_TRACE_CONTINUATION;
            info->max_trace_frame_bytes =
               MAX2(info->max_trace_frame_bytes, frame_size_bytes);
         } else {
            info->feature_flags |=
               VENTUS_RT_RESOURCE_USES_CALLABLE_CONTINUATION;
            info->max_callable_frame_bytes =
               MAX2(info->max_callable_frame_bytes, frame_size_bytes);
         }
      }
   }
   return true;
}

static bool
map_resident_tokens(nir_shader *shader,
                    struct vt_nir_rt_resident_continuation_info *info,
                    void *mem_ctx)
{
   uint32_t *tokens = ralloc_array(mem_ctx, uint32_t, info->site_count);
   uint32_t *calls = ralloc_array(mem_ctx, uint32_t, info->site_count);
   bool *seen = rzalloc_array(mem_ctx, bool, info->site_count);
   info->token_to_call = ralloc_array(mem_ctx, uint32_t, info->site_count);
   if (!tokens || !calls || !seen || !info->token_to_call)
      return false;

   uint32_t token_count = 0;
   uint32_t call_count = 0;
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic ==
             nir_intrinsic_ventus_rt_continuation_begin) {
            if (token_count >= info->site_count ||
                !nir_src_is_const(intrin->src[0]))
               return false;
            const uint32_t token = nir_src_as_uint(intrin->src[0]);
            if (token >= info->site_count || seen[token])
               return false;
            tokens[token_count++] = token;
            seen[token] = true;
         } else if (intrin->intrinsic == nir_intrinsic_rt_trace_ray) {
            if (call_count >= info->site_count)
               return false;
            const uint32_t call = nir_intrinsic_call_idx(intrin);
            if (call >= info->site_count)
               return false;
            calls[call_count++] = call;
         }
      }
   }
   if (token_count != info->site_count || call_count != info->site_count)
      return false;
   for (uint32_t i = 0; i < info->site_count; i++) {
      if (!seen[i])
         return false;
      info->token_to_call[tokens[i]] = calls[i];
   }
   return true;
}

bool
vt_nir_plan_rt_resident_continuations(
   const nir_shader *shader, uint32_t max_ray_recursion_depth,
   uint32_t payload_size_bytes, uint32_t callable_data_size_bytes,
   struct vt_nir_rt_resident_continuation_info *info, void *mem_ctx)
{
   if (!shader || !info || !mem_ctx ||
       payload_size_bytes > VENTUS_RT_MAX_PAYLOAD_BYTES_V5 ||
       callable_data_size_bytes > VENTUS_RT_MAX_CALLABLE_DATA_BYTES_V5)
      return false;
   *info = (struct vt_nir_rt_resident_continuation_info) {0};

   nir_shader *analysis = nir_shader_clone(mem_ctx, shader);
   if (!analysis)
      return false;
   info->lowered_initial_shader = analysis;
   NIR_PASS(_, analysis, nir_shader_intrinsics_pass,
            remove_nonresident_call_for_analysis, nir_metadata_none, NULL);
   const uint32_t ordinary_scratch_bytes = analysis->scratch_size;
   nir_shader **resume_shaders = NULL;
   uint32_t resume_shader_count = 0;
   const nir_lower_shader_calls_options options = {
      .address_format = nir_address_format_32bit_offset,
      .stack_alignment = VENTUS_RT_V5_CONTINUATION_ALIGNMENT_BYTES,
      .localized_loads = true,
   };
   if (!nir_lower_shader_calls(analysis, &options, &resume_shaders,
                               &resume_shader_count, mem_ctx))
      return true;

   info->resume_shaders = resume_shaders;
   info->resume_shader_count = resume_shader_count;
   info->site_count = resume_shader_count;
   info->sites = rzalloc_array(mem_ctx, struct vt_nir_rt_continuation_site,
                               info->site_count);
   bool *seen = rzalloc_array(mem_ctx, bool, info->site_count);
   if (!info->sites || !seen)
      return false;
   if (!collect_resident_sites(analysis, ordinary_scratch_bytes,
                               payload_size_bytes, callable_data_size_bytes,
                               VENTUS_RT_V5_TRACE_PDS_PREFIX_BYTES +
                                  VENTUS_RT_V5_TRACE_COMMITTED_BYTES,
                               info, seen))
      return false;
   for (uint32_t i = 0; i < resume_shader_count; i++) {
      if (!collect_resident_sites(resume_shaders[i], ordinary_scratch_bytes,
                                  payload_size_bytes,
                                  callable_data_size_bytes,
                                  VENTUS_RT_V5_TRACE_PDS_PREFIX_BYTES +
                                     VENTUS_RT_V5_TRACE_COMMITTED_BYTES,
                                  info, seen))
         return false;
   }
   for (uint32_t i = 0; i < info->site_count; i++) {
      if (!seen[i])
         return false;
   }

   if (info->max_trace_frame_bytes && max_ray_recursion_depth == 0)
      return false;
   if (info->max_callable_frame_bytes && callable_data_size_bytes == 0)
      return false;
   const uint32_t frame_stride = MAX2(
      MAX2(info->max_trace_frame_bytes, info->max_callable_frame_bytes),
      info->max_callable_frame_bytes ? VT_NIR_RT_CALLABLE_FRAME_BYTES : 0);
   const uint32_t callable_capacity =
      info->max_callable_frame_bytes ? VT_NIR_RT_BASE_CALLABLE_DEPTH : 0;
   const uint32_t trace_capacity =
      info->max_trace_frame_bytes ? max_ray_recursion_depth : 0;
   uint32_t physical_capacity = 0;
   if (!vt_nir_rt_v5_compute_continuation_stack_layout(
          frame_stride, trace_capacity, callable_capacity,
          trace_capacity != 0, &physical_capacity,
          &info->stack_bytes_per_thread))
      return false;
   return true;
}

bool
vt_nir_lower_rt_resident_continuations(
   nir_shader *shader, uint32_t max_ray_recursion_depth,
   uint32_t payload_size_bytes, uint32_t callable_data_size_bytes,
   bool parent_state_ssa_complete,
   struct vt_nir_rt_resident_continuation_info *info, void *mem_ctx)
{
   if (!shader || !info || !mem_ctx ||
       payload_size_bytes > VENTUS_RT_MAX_PAYLOAD_BYTES_V5 ||
       callable_data_size_bytes > VENTUS_RT_MAX_CALLABLE_DATA_BYTES_V5)
      return false;
   *info = (struct vt_nir_rt_resident_continuation_info) {0};
   info->lowered_initial_shader = shader;
   info->parent_state_ssa_complete = parent_state_ssa_complete;

   uint32_t site_count = 0;
   if (!count_trace_calls(shader, &site_count))
      return false;
   if (site_count == 0)
      return true;
   if (max_ray_recursion_depth == 0)
      return false;

   NIR_PASS(_, shader, nir_lower_vars_to_ssa);
   const uint32_t ordinary_scratch_bytes = shader->scratch_size;
   nir_shader **unused_resume_shaders = NULL;
   uint32_t unused_resume_shader_count = 0;
   const nir_lower_shader_calls_options options = {
      .address_format = nir_address_format_32bit_offset,
      .stack_alignment = VENTUS_RT_V5_CONTINUATION_ALIGNMENT_BYTES,
      .localized_loads = true,
      .preserve_inline_resume = true,
   };
   if (!nir_lower_shader_calls(shader, &options, &unused_resume_shaders,
                               &unused_resume_shader_count, mem_ctx) ||
       unused_resume_shader_count != 0 || unused_resume_shaders != NULL)
      return false;

   info->site_count = site_count;
   info->sites = rzalloc_array(mem_ctx, struct vt_nir_rt_continuation_site,
                               site_count);
   bool *seen = rzalloc_array(mem_ctx, bool, site_count);
   const uint32_t parent_pds_save_bytes = parent_state_ssa_complete ? 0 :
      VENTUS_RT_V5_TRACE_PDS_PREFIX_BYTES +
         VENTUS_RT_V5_TRACE_COMMITTED_BYTES;
   if (!info->sites || !seen ||
       !collect_resident_sites(shader, ordinary_scratch_bytes,
                               payload_size_bytes, callable_data_size_bytes,
                               parent_pds_save_bytes,
                               info, seen))
      return false;
   for (uint32_t i = 0; i < site_count; i++) {
      if (!seen[i] || info->sites[i].kind != VT_NIR_RT_TRACE_CONTINUATION)
         return false;
   }
   if (!map_resident_tokens(shader, info, mem_ctx))
      return false;

   if (!align_u32(shader->scratch_size,
                  VENTUS_RT_V5_CONTINUATION_ALIGNMENT_BYTES,
                  &info->scratch_bytes_per_thread) ||
       !resident_state_frame_stride(parent_pds_save_bytes,
                                    payload_size_bytes,
                                    &info->scratch_offset_bytes))
      return false;
   const uint64_t frame_bytes =
      (uint64_t)info->scratch_offset_bytes + info->scratch_bytes_per_thread;
   if (frame_bytes > UINT32_MAX ||
       !align_u32((uint32_t)frame_bytes,
                  VT_NIR_RT_STACK_FRAME_ALIGNMENT_BYTES,
                  &info->state_frame_stride_bytes))
      return false;
   info->state_frame_stride_bytes = MAX2(
      info->state_frame_stride_bytes, VT_NIR_RT_CALLABLE_FRAME_BYTES);
   info->max_trace_frame_bytes = MAX2(
      info->max_trace_frame_bytes, info->state_frame_stride_bytes);
   const uint32_t callable_capacity =
      callable_data_size_bytes ? VT_NIR_RT_BASE_CALLABLE_DEPTH : 0;
   uint32_t physical_capacity = 0;
   if (!vt_nir_rt_v5_compute_continuation_stack_layout(
          info->state_frame_stride_bytes, max_ray_recursion_depth,
          callable_capacity, true, &physical_capacity,
          &info->stack_bytes_per_thread))
      return false;

   return true;
}

bool
vt_nir_lower_rt_cps(nir_shader *shader,
                    struct vt_nir_rt_cps_info *info,
                    void *mem_ctx)
{
   if (!shader || !info || !mem_ctx)
      return false;

   *info = (struct vt_nir_rt_cps_info) {0};

   const nir_lower_shader_calls_options options = {
      /* CPS frames live in the RV64 global arena, never in PDS scratch. */
      .address_format = nir_address_format_64bit_global,
      .stack_alignment = 16,
      .localized_loads = true,
   };

   NIR_PASS(_, shader, nir_lower_vars_to_ssa);

   if (!nir_lower_shader_calls(shader, &options, &info->resume_shaders,
                               &info->num_resume_shaders, mem_ctx))
      return false;

   /*
    * This is a per-application frame size, not a PDS allocation.  The
    * wavefront profile will place frames in its global CPS arena, indexed by
    * the queue record's cps_frame ordinal.  The legacy PDS profile does not
    * invoke this lowering.
    */
   info->frame_size_bytes = ALIGN_POT(shader->scratch_size, 16);
   return true;
}
