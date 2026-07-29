/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

#include "util/ralloc.h"

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
