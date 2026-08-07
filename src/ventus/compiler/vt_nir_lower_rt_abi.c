/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

bool
vt_nir_lower_rt_abi(nir_shader *shader,
                    const struct vt_nir_rt_config *config)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   struct vt_nir_rt_config default_config;
   if (!config) {
      default_config = vt_nir_rt_default_config();
      config = &default_config;
   }

   /* Global workers use a separate global-record ABI.  Falling through here
    * would silently materialize the legacy PDS slot ABI. */
   if (config->execution_model == VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER)
      return false;

   bool progress = false;
   if (shader->info.stage == MESA_SHADER_RAYGEN) {
      nir_builder b = nir_builder_create(impl);
      b.cursor = nir_before_impl(impl);
      vt_nir_store_active_level(&b, config, VT_RT_LAYOUT_RAYGEN,
                                nir_imm_int(&b, 0));
      progress = true;
   }
   progress |= vt_nir_lower_rt_payload(shader, config);
   progress |= vt_nir_lower_rt_hit_record(shader, config);
   progress |= vt_nir_lower_rt_control(shader, config);
   progress |= vt_nir_lower_shader_record_ptr(shader, config);
   progress |= vt_nir_lower_shader_scratch(shader, config);

   enum vt_rt_shader_layout layout = vt_nir_layout_for_stage(shader->info.stage);
   shader->scratch_size =
      MAX2(shader->scratch_size, config->scratchpad[layout].total_size_bytes);

   nir_shader_gather_info(shader, impl);
   nir_progress(progress, impl, progress ? nir_metadata_none : nir_metadata_all);
   return progress;
}
