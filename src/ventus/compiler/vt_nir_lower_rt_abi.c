/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

bool
vt_nir_lower_rt_abi(nir_shader *shader,
                    const struct vt_nir_rt_config *config)
{
   (void)config;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   struct vt_nir_rt_config default_config;
   if (!config) {
      default_config = vt_nir_rt_default_config();
      config = &default_config;
   }

   enum vt_rt_shader_layout layout = vt_nir_layout_for_stage(shader->info.stage);
   if (shader->scratch_size < config->scratchpad[layout].stride_bytes)
      shader->scratch_size = config->scratchpad[layout].stride_bytes;

   nir_shader_gather_info(shader, impl);
   nir_progress(true, impl, nir_metadata_none);
   return true;
}
