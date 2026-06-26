/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

#include "compiler/glsl_types.h"

bool
vt_nir_lower_shader_scratch(nir_shader *shader,
                            const struct vt_nir_rt_config *config)
{
   (void)config;

   unsigned before = shader->scratch_size;
   bool progress = nir_lower_vars_to_scratch(shader, 0,
                                             glsl_get_cl_type_size_align,
                                             glsl_get_cl_type_size_align);

   return progress || shader->scratch_size != before;
}

void
vt_nir_pipeline_scratch_init(struct vt_nir_pipeline_scratch_info *info)
{
   info->per_lane_scratch_size = 0;
   for (unsigned i = 0; i < VT_RT_LAYOUT_COUNT; ++i)
      info->shader_scratch_size[i] = 0;
}

void
vt_nir_pipeline_scratch_include_shader(struct vt_nir_pipeline_scratch_info *info,
                                       const nir_shader *shader)
{
   if (!shader)
      return;

   enum vt_rt_shader_layout layout = vt_nir_layout_for_stage(shader->info.stage);
   info->shader_scratch_size[layout] =
      MAX2(info->shader_scratch_size[layout], shader->scratch_size);
   info->per_lane_scratch_size =
      MAX2(info->per_lane_scratch_size, shader->scratch_size);
}

uint32_t
vt_nir_pipeline_compute_scratch(nir_shader **shaders,
                                unsigned shader_count,
                                struct vt_nir_pipeline_scratch_info *info)
{
   struct vt_nir_pipeline_scratch_info local;
   if (!info)
      info = &local;

   vt_nir_pipeline_scratch_init(info);

   for (unsigned i = 0; i < shader_count; ++i)
      vt_nir_pipeline_scratch_include_shader(info, shaders[i]);

   return vt_nir_pipeline_scratch_size(info);
}

uint32_t
vt_nir_pipeline_scratch_size(const struct vt_nir_pipeline_scratch_info *info)
{
   return info->per_lane_scratch_size;
}
