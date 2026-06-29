/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_rt_test_common.h"

static nir_shader *
build_abi_probe_closest_hit(const nir_shader_compiler_options *options)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_CLOSEST_HIT, options,
                                     "vt_rt_abi_probe_closest_hit");

   nir_variable *payload =
      nir_variable_create(b.shader, nir_var_shader_call_data,
                          glsl_vec4_type(), "payload_in");
   nir_def *payload_value = nir_load_var(&b, payload);
   nir_store_var(&b, payload, payload_value, 0xf);

   nir_def *record_ptr = nir_load_shader_record_ptr(&b);
   nir_def *hit_t = nir_load_ray_t_max(&b);
   nir_def *primitive_id = nir_load_primitive_id(&b);
   nir_def *instance_id = nir_load_instance_id(&b);
   nir_def *geometry_id = nir_load_ray_geometry_index(&b);
   nir_def *hit_kind = nir_load_ray_hit_kind(&b);

   nir_variable *sink =
      nir_variable_create(b.shader, nir_var_shader_temp,
                          glsl_uvec4_type(), "abi_probe_sink");
   nir_store_var(&b, sink,
                 nir_vec4(&b, nir_unpack_64_2x32_split_x(&b, record_ptr),
                          primitive_id, instance_id,
                          nir_iadd(&b, geometry_id, hit_kind)),
                 0xf);
   nir_store_var(&b, sink,
                 nir_vec4(&b, nir_unpack_64_2x32_split_y(&b, record_ptr),
                          hit_t, primitive_id, instance_id),
                 0xf);

   return b.shader;
}

static nir_shader *
build_abi_probe_any_hit(const nir_shader_compiler_options *options)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_ANY_HIT, options,
                                     "vt_rt_abi_probe_any_hit");

   nir_def *front_face = nir_load_front_face(&b, 1);
   nir_def *hit_kind = nir_load_ray_hit_kind(&b);

   nir_variable *sink =
      nir_variable_create(b.shader, nir_var_shader_temp,
                          glsl_uint_type(), "any_hit_sink");
   nir_store_var(&b, sink,
                 nir_iadd(&b, nir_b2i32(&b, front_face), hit_kind), 0x1);
   nir_ignore_ray_intersection(&b);

   return b.shader;
}

static nir_shader *
build_abi_probe_intersection(const nir_shader_compiler_options *options)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_INTERSECTION, options,
                                     "vt_rt_abi_probe_intersection");

   nir_def *accepted =
      nir_report_ray_intersection(&b, 1, nir_imm_float(&b, 1.0f),
                                  nir_imm_int(&b, 0xfe));

   nir_variable *sink =
      nir_variable_create(b.shader, nir_var_shader_temp,
                          glsl_bool_type(), "intersection_sink");
   nir_store_var(&b, sink, accepted, 0x1);

   return b.shader;
}

static nir_shader *
build_abi_probe_terminating_any_hit(const nir_shader_compiler_options *options)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_ANY_HIT, options,
                                     "vt_rt_abi_probe_terminating_any_hit");

   nir_terminate_ray(&b);
   return b.shader;
}

int
main(int argc, char **argv)
{
   VT_RT_CHECK(argc == 2);
   glsl_type_singleton_init_or_ref();

   const nir_shader_compiler_options options = {0};
   struct vt_nir_rt_config config = vt_nir_rt_default_config();
   config.hit_sbt_base_addr = 0x100000000ull;
   config.hit_sbt_record_stride = 96;
   config.shader_group_handle_size = 32;
   nir_shader *raygen = vt_rt_load_raygen_spirv(argv[1], &options);
   nir_shader *miss =
      vt_rt_build_empty_shader(MESA_SHADER_MISS, &options, "vt_rt_minimal_miss");
   nir_shader *closest_hit = build_abi_probe_closest_hit(&options);
   nir_shader *any_hit = build_abi_probe_any_hit(&options);
   nir_shader *intersection = build_abi_probe_intersection(&options);
   nir_shader *terminating_any_hit =
      build_abi_probe_terminating_any_hit(&options);

   struct vt_nir_rt_abi_layout abi = vt_nir_rt_default_abi_layout();
   VT_RT_CHECK(abi.payload_base_bytes >=
               abi.ray_state_base_bytes + abi.ray_state_size_bytes);
   VT_RT_CHECK(abi.control_base_bytes >=
               abi.payload_base_bytes + abi.payload_size_bytes);
   VT_RT_CHECK(abi.candidate_hit_record_base_bytes >=
               abi.control_base_bytes + abi.control_size_bytes);
   VT_RT_CHECK(abi.hit_record_base_bytes >=
               abi.candidate_hit_record_base_bytes +
               abi.candidate_hit_record_size_bytes);
   VT_RT_CHECK(abi.hit_attrib_base_bytes >=
               abi.hit_record_base_bytes + abi.hit_record_size_bytes);
   VT_RT_CHECK(abi.continuation_base_bytes >=
               abi.hit_attrib_base_bytes + abi.hit_attrib_size_bytes);
   VT_RT_CHECK(abi.control_size_bytes == VT_RT_CONTROL_WORD_COUNT * 4);
   VT_RT_CHECK(abi.candidate_hit_record_size_bytes ==
               VT_RT_HIT_RECORD_WORD_COUNT * 4);
   VT_RT_CHECK(abi.hit_record_size_bytes == VT_RT_HIT_RECORD_WORD_COUNT * 4);
   VT_RT_CHECK(abi.continuation_size_bytes == 0);

   struct vt_rt_intrinsic_counts before;
   vt_rt_count_intrinsics(raygen, &before);
   VT_RT_CHECK(before.trace_ray == 1);

   bool progress = vt_nir_lower_rt_io_monolithic(raygen, &config);
   VT_RT_CHECK(progress);

   progress = vt_nir_lower_rt_abi(raygen, &config);
   VT_RT_CHECK(!progress || raygen->scratch_size >= 0);
   VT_RT_CHECK(raygen->scratch_size >= abi.total_size_bytes);

   struct vt_rt_intrinsic_counts after;
   vt_rt_count_intrinsics(raygen, &after);
   VT_RT_CHECK(after.trace_ray == 0);
   VT_RT_CHECK(after.store_shared > 0);
   VT_RT_CHECK(after.load_shared > 0);
   VT_RT_CHECK(vt_rt_shader_has_imul_const(raygen,
                                           config.hit_sbt_record_stride));

   struct vt_rt_intrinsic_counts closest_before;
   vt_rt_count_intrinsics(closest_hit, &closest_before);
   VT_RT_CHECK(closest_before.load_deref > 0);
   VT_RT_CHECK(closest_before.store_deref > 0);
   VT_RT_CHECK(closest_before.load_shader_record_ptr == 1);
   VT_RT_CHECK(closest_before.load_ray_t_max == 1);
   VT_RT_CHECK(closest_before.load_primitive_id == 1);
   VT_RT_CHECK(closest_before.load_instance_id == 1);
   VT_RT_CHECK(closest_before.load_ray_geometry_index == 1);
   VT_RT_CHECK(closest_before.load_ray_hit_kind == 1);

   progress = vt_nir_lower_rt_abi(closest_hit, &config);
   VT_RT_CHECK(progress);
   VT_RT_CHECK(closest_hit->scratch_size >= abi.total_size_bytes);

   struct vt_rt_intrinsic_counts closest_after;
   vt_rt_count_intrinsics(closest_hit, &closest_after);
   VT_RT_CHECK(closest_after.load_shader_record_ptr == 0);
   VT_RT_CHECK(closest_after.load_ray_t_max == 0);
   VT_RT_CHECK(closest_after.load_primitive_id == 0);
   VT_RT_CHECK(closest_after.load_instance_id == 0);
   VT_RT_CHECK(closest_after.load_ray_geometry_index == 0);
   VT_RT_CHECK(closest_after.load_ray_hit_kind == 0);
   VT_RT_CHECK(closest_after.load_shared > closest_before.load_shared);
   VT_RT_CHECK(closest_after.store_shared > closest_before.store_shared);

   struct vt_rt_intrinsic_counts any_before;
   vt_rt_count_intrinsics(any_hit, &any_before);
   VT_RT_CHECK(any_before.load_front_face == 1);
   VT_RT_CHECK(any_before.load_ray_hit_kind == 1);
   VT_RT_CHECK(any_before.ignore_ray_intersection == 1);

   progress = vt_nir_lower_rt_abi(any_hit, &config);
   VT_RT_CHECK(progress);
   VT_RT_CHECK(any_hit->scratch_size >= abi.total_size_bytes);

   struct vt_rt_intrinsic_counts any_after;
   vt_rt_count_intrinsics(any_hit, &any_after);
   VT_RT_CHECK(any_after.load_front_face == 0);
   VT_RT_CHECK(any_after.load_ray_hit_kind == 0);
   VT_RT_CHECK(any_after.ignore_ray_intersection == 0);
   VT_RT_CHECK(any_after.load_shared > any_before.load_shared);
   VT_RT_CHECK(any_after.store_shared > any_before.store_shared);

   struct vt_rt_intrinsic_counts terminating_any_before;
   vt_rt_count_intrinsics(terminating_any_hit, &terminating_any_before);
   VT_RT_CHECK(terminating_any_before.terminate_ray == 1);

   progress = vt_nir_lower_rt_abi(terminating_any_hit, &config);
   VT_RT_CHECK(progress);
   VT_RT_CHECK(terminating_any_hit->scratch_size >= abi.total_size_bytes);

   struct vt_rt_intrinsic_counts terminating_any_after;
   vt_rt_count_intrinsics(terminating_any_hit, &terminating_any_after);
   VT_RT_CHECK(terminating_any_after.terminate_ray == 0);
   VT_RT_CHECK(terminating_any_after.store_shared >
               terminating_any_before.store_shared);

   struct vt_rt_intrinsic_counts intersection_before;
   vt_rt_count_intrinsics(intersection, &intersection_before);
   VT_RT_CHECK(intersection_before.report_ray_intersection == 1);

   progress = vt_nir_lower_rt_abi(intersection, &config);
   VT_RT_CHECK(progress);
   VT_RT_CHECK(intersection->scratch_size >= abi.total_size_bytes);

   struct vt_rt_intrinsic_counts intersection_after;
   vt_rt_count_intrinsics(intersection, &intersection_after);
   VT_RT_CHECK(intersection_after.report_ray_intersection == 0);
   VT_RT_CHECK(intersection_after.load_shared > intersection_before.load_shared);
   VT_RT_CHECK(intersection_after.store_shared > intersection_before.store_shared);

   nir_shader *pipeline[] = {
      raygen, miss, closest_hit, any_hit, intersection, terminating_any_hit,
   };
   struct vt_nir_pipeline_scratch_info scratch_info;
   uint32_t per_lane_scratch =
      vt_nir_pipeline_compute_scratch(pipeline, 6, &scratch_info);

   VT_RT_CHECK(per_lane_scratch >= abi.total_size_bytes);
   VT_RT_CHECK(scratch_info.shader_scratch_size[VT_RT_LAYOUT_RAYGEN] >=
               abi.total_size_bytes);
   VT_RT_CHECK(scratch_info.shader_scratch_size[VT_RT_LAYOUT_MISS] ==
               miss->scratch_size);
   VT_RT_CHECK(scratch_info.shader_scratch_size[VT_RT_LAYOUT_CLOSEST_HIT] >=
               abi.total_size_bytes);
   VT_RT_CHECK(scratch_info.shader_scratch_size[VT_RT_LAYOUT_ANY_HIT] >=
               abi.total_size_bytes);
   VT_RT_CHECK(scratch_info.shader_scratch_size[VT_RT_LAYOUT_INTERSECTION] >=
               abi.total_size_bytes);

   nir_shader *manual_raygen =
      vt_rt_build_empty_shader(MESA_SHADER_RAYGEN, &options,
                               "vt_rt_manual_raygen_scratch");
   nir_shader *manual_miss =
      vt_rt_build_empty_shader(MESA_SHADER_MISS, &options,
                               "vt_rt_manual_miss_scratch");
   nir_shader *manual_closest_hit =
      vt_rt_build_empty_shader(MESA_SHADER_CLOSEST_HIT, &options,
                               "vt_rt_manual_closest_hit_scratch");
   manual_raygen->scratch_size = 48;
   manual_miss->scratch_size = 16;
   manual_closest_hit->scratch_size = 80;
   nir_shader *manual_pipeline[] = {
      manual_raygen,
      manual_miss,
      manual_closest_hit,
   };
   struct vt_nir_pipeline_scratch_info manual_scratch_info;
   uint32_t manual_per_lane_scratch =
      vt_nir_pipeline_compute_scratch(manual_pipeline, 3,
                                      &manual_scratch_info);
   VT_RT_CHECK(manual_per_lane_scratch == 80);
   VT_RT_CHECK(manual_scratch_info.shader_scratch_size[VT_RT_LAYOUT_RAYGEN] == 48);
   VT_RT_CHECK(manual_scratch_info.shader_scratch_size[VT_RT_LAYOUT_MISS] == 16);
   VT_RT_CHECK(manual_scratch_info.shader_scratch_size[VT_RT_LAYOUT_CLOSEST_HIT] == 80);

   printf("PASS vt_rt_minimal_loop spirv_input=1 trace_ray_lowered=1 "
          "abi_candidate_committed=1 abi_control=1 sbt_entry_calc=1 "
          "hit_sbt_base=0x%llx "
          "hit_sbt_stride=%u shader_group_handle_size=%u "
          "shared_store=%u shared_load=%u per_lane_scratch=%u "
          "abi_total=%u control_base=%u candidate_base=%u committed_base=%u "
          "hit_attrib_base=%u manual_max_scratch=%u\n",
          (unsigned long long)config.hit_sbt_base_addr,
          config.hit_sbt_record_stride, config.shader_group_handle_size,
          after.store_shared + closest_after.store_shared +
             any_after.store_shared + terminating_any_after.store_shared +
             intersection_after.store_shared,
          after.load_shared + closest_after.load_shared +
             any_after.load_shared + intersection_after.load_shared,
          per_lane_scratch, abi.total_size_bytes, abi.control_base_bytes,
          abi.candidate_hit_record_base_bytes, abi.hit_record_base_bytes,
          abi.hit_attrib_base_bytes, manual_per_lane_scratch);

   ralloc_free(raygen);
   ralloc_free(miss);
   ralloc_free(closest_hit);
   ralloc_free(any_hit);
   ralloc_free(intersection);
   ralloc_free(terminating_any_hit);
   ralloc_free(manual_raygen);
   ralloc_free(manual_miss);
   ralloc_free(manual_closest_hit);
   glsl_type_singleton_decref();
   return 0;
}
