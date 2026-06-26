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

   nir_shader_gather_info(shader, impl);
   nir_progress(true, impl, nir_metadata_none);
   return true;
}
