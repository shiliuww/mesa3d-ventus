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

   bool progress = false;
   progress |= vt_nir_lower_rt_payload(shader, config);
   progress |= vt_nir_lower_rt_hit_record(shader, config);
   progress |= vt_nir_lower_shader_record_ptr(shader, config);
   progress |= vt_nir_lower_shader_scratch(shader, config);

   enum vt_rt_shader_layout layout = vt_nir_layout_for_stage(shader->info.stage);
   shader->scratch_size =
      MAX2(shader->scratch_size, config->scratchpad[layout].total_size_bytes);

   nir_shader_gather_info(shader, impl);
   nir_progress(progress, impl, progress ? nir_metadata_none : nir_metadata_all);
   return progress;
}
