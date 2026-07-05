/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

struct lower_rt_io_state {
   const struct vt_nir_rt_config *config;
};

static nir_def *
load_dispatch_arg(nir_builder *b, uint32_t offset, unsigned bit_size)
{
   nir_intrinsic_instr *intrin =
      nir_intrinsic_instr_create(b->shader, nir_intrinsic_ventus_rt_dispatch_arg);
   intrin->src[0] = nir_src_for_ssa(nir_imm_int(b, offset));

   nir_def_init(&intrin->instr, &intrin->def, 1, bit_size);
   intrin->num_components = 1;
   nir_builder_instr_insert(b, &intrin->instr);
   return &intrin->def;
}

static bool
lower_trace_ray(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   if (intrin->intrinsic != nir_intrinsic_trace_ray)
      return false;

   struct lower_rt_io_state *state = data;
   const struct vt_nir_rt_config *config = state->config;
   if (config->execution_model != VT_RT_EXECUTION_ITERATIVE_RAYGEN_LOOP)
      return false;

   b->cursor = nir_after_instr(&intrin->instr);

   struct vt_nir_rt_trace_params params = {
      .accel_struct = intrin->src[0].ssa,
      .flags = intrin->src[1].ssa,
      .cull_mask = intrin->src[2].ssa,
      .sbt_base = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HIT_SBT_ADDR, 64),
      .miss_sbt_base = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_MISS_SBT_ADDR, 64),
      .sbt_offset = intrin->src[3].ssa,
      .sbt_stride = intrin->src[4].ssa,
      .miss_index = intrin->src[5].ssa,
      .origin = intrin->src[6].ssa,
      .tmin = intrin->src[7].ssa,
      .direction = intrin->src[8].ssa,
      .tmax = intrin->src[9].ssa,
      .payload = intrin->src[10].ssa,
   };

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
      nir_ieq_imm(b, result.status, VT_RT_TRAVERSAL_CANDIDATE_NON_OPAQUE_TRIANGLE);
   nir_def *candidate_procedural =
      nir_ieq_imm(b, result.status, VT_RT_TRAVERSAL_CANDIDATE_PROCEDURAL_AABB);

   if (config && config->enable_monolithic_shader_callbacks &&
       (config->emit_closest_hit || config->emit_miss ||
        config->emit_any_hit || config->emit_intersection)) {
      nir_push_if(b, complete_hit);
      if (config->emit_closest_hit)
         config->emit_closest_hit(b, &params, &result, config->callback_data);

      nir_push_else(b, NULL);
      nir_push_if(b, complete_miss);
      if (config->emit_miss)
         config->emit_miss(b, &params, &result, config->callback_data);
      nir_push_else(b, NULL);

      nir_push_if(b, candidate_non_opaque);
      if (config->emit_any_hit)
         config->emit_any_hit(b, &params, &result, config->callback_data);
      nir_push_else(b, NULL);

      nir_push_if(b, candidate_procedural);
      if (config->emit_intersection)
         config->emit_intersection(b, &params, &result, config->callback_data);

      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);

      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
   }

   nir_push_if(b, terminal);
   nir_ventus_rt_release(b, result.slot_base);
   nir_pop_if(b, NULL);

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
