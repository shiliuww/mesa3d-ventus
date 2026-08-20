/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

#include <limits.h>

#include "util/ralloc.h"
#include "ventus/common/ventus_kernel_resource.h"
#include "ventus/common/ventus_rt_v5.h"

#define VT_RT_V5_CONTINUATION_HEADER_BYTES 16u
#define VT_RT_V5_TRACE_PDS_PREFIX_BYTES 96u
#define VT_RT_V5_TRACE_COMMITTED_BYTES 96u

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

static bool
resident_site_frame_size(enum vt_nir_rt_continuation_kind kind,
                         uint32_t ssa_spill_bytes,
                         uint32_t payload_size_bytes,
                         uint32_t *frame_size_bytes)
{
   uint64_t fixed_bytes = VT_RT_V5_CONTINUATION_HEADER_BYTES;
   if (kind == VT_NIR_RT_TRACE_CONTINUATION)
      fixed_bytes += VT_RT_V5_TRACE_PDS_PREFIX_BYTES +
                     VT_RT_V5_TRACE_COMMITTED_BYTES;
   fixed_bytes += payload_size_bytes;
   if (fixed_bytes > UINT32_MAX - ssa_spill_bytes)
      return false;
   return align_u32((uint32_t)fixed_bytes + ssa_spill_bytes,
                    VENTUS_RT_V5_CONTINUATION_ALIGNMENT_BYTES,
                    frame_size_bytes);
}

static bool
collect_resident_sites(
   nir_shader *shader, uint32_t ordinary_scratch_bytes,
   uint32_t payload_size_bytes, uint32_t callable_data_size_bytes,
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
                  VT_RT_V5_TRACE_PDS_PREFIX_BYTES +
                     VT_RT_V5_TRACE_COMMITTED_BYTES : 0,
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
                               info, seen))
      return false;
   for (uint32_t i = 0; i < resume_shader_count; i++) {
      if (!collect_resident_sites(resume_shaders[i], ordinary_scratch_bytes,
                                  payload_size_bytes,
                                  callable_data_size_bytes, info, seen))
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
   uint64_t stack_bytes =
      (uint64_t)info->max_trace_frame_bytes * max_ray_recursion_depth +
      info->max_callable_frame_bytes;
   if (stack_bytes > UINT32_MAX ||
       !align_u32((uint32_t)stack_bytes,
                  VENTUS_RT_V5_CONTINUATION_ALIGNMENT_BYTES,
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
