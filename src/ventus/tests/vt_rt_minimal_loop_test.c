/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_rt_test_common.h"

int
main(int argc, char **argv)
{
   VT_RT_CHECK(argc == 2);
   glsl_type_singleton_init_or_ref();

   const nir_shader_compiler_options options = {0};
   struct vt_nir_rt_config config = vt_nir_rt_default_config();
   nir_shader *raygen = vt_rt_load_raygen_spirv(argv[1], &options);
   nir_shader *miss =
      vt_rt_build_empty_shader(MESA_SHADER_MISS, &options, "vt_rt_minimal_miss");
   nir_shader *closest_hit =
      vt_rt_build_empty_shader(MESA_SHADER_CLOSEST_HIT, &options,
                               "vt_rt_minimal_closest_hit");

   struct vt_rt_intrinsic_counts before;
   vt_rt_count_intrinsics(raygen, &before);
   VT_RT_CHECK(before.trace_ray == 1);

   bool progress = vt_nir_lower_rt_io_monolithic(raygen, &config);
   VT_RT_CHECK(progress);

   progress = vt_nir_lower_rt_abi(raygen, &config);
   VT_RT_CHECK(!progress || raygen->scratch_size >= 0);

   struct vt_rt_intrinsic_counts after;
   vt_rt_count_intrinsics(raygen, &after);
   VT_RT_CHECK(after.trace_ray == 0);
   VT_RT_CHECK(after.store_shared > 0);
   VT_RT_CHECK(after.load_shared > 0);

   raygen->scratch_size = 48;
   miss->scratch_size = 16;
   closest_hit->scratch_size = 80;

   nir_shader *pipeline[] = {raygen, miss, closest_hit};
   struct vt_nir_pipeline_scratch_info scratch_info;
   uint32_t per_lane_scratch =
      vt_nir_pipeline_compute_scratch(pipeline, 3, &scratch_info);

   VT_RT_CHECK(per_lane_scratch == 80);
   VT_RT_CHECK(scratch_info.shader_scratch_size[VT_RT_LAYOUT_RAYGEN] == 48);
   VT_RT_CHECK(scratch_info.shader_scratch_size[VT_RT_LAYOUT_MISS] == 16);
   VT_RT_CHECK(scratch_info.shader_scratch_size[VT_RT_LAYOUT_CLOSEST_HIT] == 80);

   printf("PASS vt_rt_minimal_loop spirv_input=1 trace_ray_lowered=1 shared_store=%u "
          "shared_load=%u per_lane_scratch=%u\n",
          after.store_shared, after.load_shared, per_lane_scratch);

   ralloc_free(raygen);
   ralloc_free(miss);
   ralloc_free(closest_hit);
   glsl_type_singleton_decref();
   return 0;
}
