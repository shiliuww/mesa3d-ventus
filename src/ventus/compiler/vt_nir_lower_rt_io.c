/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

struct lower_rt_io_state {
   const struct vt_nir_rt_config *config;
};

static bool
lower_trace_ray(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   if (intrin->intrinsic != nir_intrinsic_trace_ray)
      return false;

   struct lower_rt_io_state *state = data;
   const struct vt_nir_rt_config *config = state->config;

   b->cursor = nir_after_instr(&intrin->instr);

   struct vt_nir_rt_trace_params params = {
      .accel_struct = intrin->src[0].ssa,
      .flags = intrin->src[1].ssa,
      .cull_mask = intrin->src[2].ssa,
      .sbt_offset = nir_iand_imm(b, intrin->src[3].ssa, 0xf),
      .sbt_stride = nir_iand_imm(b, intrin->src[4].ssa, 0xf),
      .miss_index = nir_iand_imm(b, intrin->src[5].ssa, 0xffff),
      .origin = intrin->src[6].ssa,
      .tmin = intrin->src[7].ssa,
      .direction = intrin->src[8].ssa,
      .tmax = intrin->src[9].ssa,
      .payload = intrin->src[10].ssa,
   };

   struct vt_nir_rt_result result = vt_nir_build_traversal(b, &params, config);

   if (config && config->enable_monolithic_shader_callbacks &&
       (config->emit_closest_hit || config->emit_miss)) {
      nir_push_if(b, result.hit);
      if (config->emit_closest_hit)
         config->emit_closest_hit(b, &params, &result, config->callback_data);

      nir_push_else(b, NULL);
      if (config->emit_miss)
         config->emit_miss(b, &params, &result, config->callback_data);

      nir_pop_if(b, NULL);
   }

   nir_instr_remove(&intrin->instr);
   return true;
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

   return nir_shader_intrinsics_pass(shader, lower_trace_ray,
                                     nir_metadata_none, &state);
}
