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
build_repeated_any_hit_probe(const nir_shader_compiler_options *options)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_ANY_HIT, options,
                                     "vt_rt_repeated_any_hit_probe");

   /* The same candidate system value twice exercises the read-only LDS
    * promotion used by compact any-hit callbacks. */
   nir_def *front_face_a = nir_load_front_face(&b, 1);
   nir_def *front_face_b = nir_load_front_face(&b, 1);
   nir_variable *sink =
      nir_variable_create(b.shader, nir_var_shader_temp,
                          glsl_uint_type(), "repeated_any_hit_sink");
   nir_store_var(&b, sink,
                 nir_iadd(&b, nir_b2i32(&b, front_face_a),
                          nir_b2i32(&b, front_face_b)), 0x1);
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

static nir_shader *
build_payload_location_probe(const nir_shader_compiler_options *options)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_CLOSEST_HIT, options,
                                     "vt_rt_payload_location_probe");

   nir_variable *hit_value =
      nir_local_variable_create(b.impl, glsl_vec_type(3), "hitValue");
   hit_value->data.explicit_location = true;
   hit_value->data.location = 0;

   nir_variable *shadowed =
      nir_local_variable_create(b.impl, glsl_bool_type(), "shadowed");
   shadowed->data.explicit_location = true;
   shadowed->data.location = 2;

   nir_store_var(&b, hit_value,
                 nir_vec3(&b, nir_imm_float(&b, 1.0f),
                          nir_imm_float(&b, 0.5f),
                          nir_imm_float(&b, 0.25f)),
                 0x7);
   nir_store_var(&b, shadowed, nir_imm_bool(&b, false), 0x1);

   return b.shader;
}

static nir_shader *
build_push_constant_probe(const nir_shader_compiler_options *options)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_RAYGEN, options,
                                     "vt_rt_push_constant_probe");

   nir_def *address = nir_load_push_constant(
      &b, 1, 64, nir_imm_int(&b, 4), .base = 8, .range = 16);
   nir_variable *sink =
      nir_variable_create(b.shader, nir_var_shader_temp,
                          glsl_uint_type(), "push_constant_sink");
   nir_store_var(&b, sink, nir_unpack_64_2x32_split_x(&b, address), 0x1);

   return b.shader;
}

static void
emit_test_trace_ray(nir_builder *b)
{
   nir_intrinsic_instr *trace =
      nir_intrinsic_instr_create(b->shader, nir_intrinsic_trace_ray);
   trace->src[0] = nir_src_for_ssa(nir_imm_int64(b, 0x1000));
   trace->src[1] = nir_src_for_ssa(nir_imm_int(b, 0));
   trace->src[2] = nir_src_for_ssa(nir_imm_int(b, 0xff));
   trace->src[3] = nir_src_for_ssa(nir_imm_int(b, 0));
   trace->src[4] = nir_src_for_ssa(nir_imm_int(b, 0));
   trace->src[5] = nir_src_for_ssa(nir_imm_int(b, 0));
   trace->src[6] = nir_src_for_ssa(nir_vec3(
      b, nir_imm_float(b, 0.0), nir_imm_float(b, 0.0), nir_imm_float(b, 0.0)));
   trace->src[7] = nir_src_for_ssa(nir_imm_float(b, 0.0));
   trace->src[8] = nir_src_for_ssa(nir_vec3(
      b, nir_imm_float(b, 0.0), nir_imm_float(b, 0.0), nir_imm_float(b, 1.0)));
   trace->src[9] = nir_src_for_ssa(nir_imm_float(b, 1.0));
   trace->src[10] = nir_src_for_ssa(nir_imm_int64(b, 0));
   nir_builder_instr_insert(b, &trace->instr);
}

/* The child is followed by a parent-only branch.  Its two loads are the
 * expected PDS memory live-out set; unrelated candidate/header words must not
 * be copied merely because RTcore can overwrite them. */
static void
emit_nested_liveness_callback(nir_builder *b,
                              const struct vt_nir_rt_trace_params *params,
                              const struct vt_nir_rt_result *result,
                              void *data)
{
   (void)params;
   (void)result;

   unsigned *remaining_children = data;
   if (*remaining_children) {
      (*remaining_children)--;
      emit_test_trace_ray(b);
   }
   nir_def *launch_x = nir_channel(b, nir_load_ray_launch_id(b), 0);
   nir_push_if(b, nir_ine_imm(b, launch_x, 0));
   nir_ventus_rt_load_slot(
      b, 1, 32, nir_imm_int(
         b, VT_RT_ABI_COMMITTED_HIT_RECORD_BASE_BYTES +
               VT_RT_HIT_RECORD_META0 * sizeof(uint32_t)));
   nir_push_else(b, NULL);
   nir_ventus_rt_load_slot(
      b, 1, 32, nir_imm_int(
         b, VT_RT_ABI_COMMITTED_HIT_RECORD_BASE_BYTES +
               VT_RT_HIT_RECORD_META1 * sizeof(uint32_t)));
   nir_pop_if(b, NULL);
}

static unsigned
count_store_slot_to_const_addr(nir_shader *shader, uint32_t addr)
{
   unsigned count = 0;

   NIR_PASS(_, shader, nir_opt_constant_folding);

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != nir_intrinsic_ventus_rt_store_slot)
               continue;

            if (nir_src_is_const(intrin->src[1]) &&
                nir_src_as_uint(intrin->src[1]) == addr)
               count++;
         }
      }
   }

   return count;
}

static unsigned
count_slot_ssa_restores_to_const_addr(nir_shader *shader, uint32_t addr)
{
   unsigned count = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
            if (store->intrinsic != nir_intrinsic_ventus_rt_store_slot ||
                !nir_src_is_const(store->src[1]) ||
                nir_src_as_uint(store->src[1]) != addr)
               continue;

            nir_instr *def = nir_def_instr(store->src[0].ssa);
            if (def->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *load = nir_instr_as_intrinsic(def);
            if (load->intrinsic == nir_intrinsic_ventus_rt_load_slot &&
                nir_src_is_const(load->src[0]) &&
                nir_src_as_uint(load->src[0]) == addr)
               count++;
         }
      }
   }

   return count;
}

static unsigned
count_intrinsic(nir_shader *shader, nir_intrinsic_op op)
{
   unsigned count = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               count++;
         }
      }
   }

   return count;
}

static unsigned
count_variables(nir_shader *shader, nir_variable_mode mode)
{
   unsigned count = 0;
   nir_foreach_variable_in_shader(var, shader) {
      if (var->data.mode == mode)
         count++;
   }
   return count;
}

int
main(int argc, char **argv)
{
   VT_RT_CHECK(argc == 3);
   glsl_type_singleton_init_or_ref();

   const nir_shader_compiler_options options = {0};
   struct vt_nir_rt_config config = vt_nir_rt_default_config();
   config.hit_sbt_base_addr = 0x100000000ull;
   config.hit_sbt_record_stride = 96;
   config.shader_group_handle_size = 32;
   nir_shader *raygen = vt_rt_load_raygen_spirv(argv[1], &options);
   nir_shader *nested_liveness_raygen =
      vt_rt_load_raygen_spirv(argv[1], &options);
   nir_shader *nested_liveness_baseline =
      vt_rt_load_raygen_spirv(argv[1], &options);
   nir_shader *cps_raygen = vt_rt_load_raygen_spirv(argv[1], &options);
   nir_shader *global_raygen = vt_rt_load_raygen_spirv(argv[1], &options);
   nir_shader *global_loop_raygen = vt_rt_load_raygen_spirv(argv[2], &options);
   nir_shader *miss =
      vt_rt_build_empty_shader(MESA_SHADER_MISS, &options, "vt_rt_minimal_miss");
   nir_shader *closest_hit = build_abi_probe_closest_hit(&options);
   nir_shader *global_callback_closest_hit =
      build_abi_probe_closest_hit(&options);
   nir_shader *any_hit = build_abi_probe_any_hit(&options);
   nir_shader *global_callback_any_hit =
      build_repeated_any_hit_probe(&options);
   nir_shader *intersection = build_abi_probe_intersection(&options);
   nir_shader *deferred_intersection =
      build_abi_probe_intersection(&options);
   nir_shader *terminating_any_hit =
      build_abi_probe_terminating_any_hit(&options);
   nir_shader *payload_location_probe =
      build_payload_location_probe(&options);
   nir_shader *push_constant_probe = build_push_constant_probe(&options);

   struct vt_nir_rt_config nested_liveness_config = config;
   unsigned nested_liveness_children = 1;
   nested_liveness_config.enable_monolithic_shader_callbacks = true;
   nested_liveness_config.emit_miss = emit_nested_liveness_callback;
   nested_liveness_config.callback_data = &nested_liveness_children;

   struct vt_nir_rt_config nested_liveness_baseline_config = config;
   unsigned nested_liveness_baseline_children = 0;
   nested_liveness_baseline_config.enable_monolithic_shader_callbacks = true;
   nested_liveness_baseline_config.emit_miss = emit_nested_liveness_callback;
   nested_liveness_baseline_config.callback_data =
      &nested_liveness_baseline_children;
   VT_RT_CHECK(vt_nir_lower_rt_io_monolithic(
      nested_liveness_baseline, &nested_liveness_baseline_config));
   VT_RT_CHECK(vt_nir_lower_rt_io_monolithic(nested_liveness_raygen,
                                              &nested_liveness_config));
   NIR_PASS(_, nested_liveness_raygen, nir_opt_constant_folding);
   const uint32_t committed_meta0 = VT_RT_ABI_COMMITTED_HIT_RECORD_BASE_BYTES;
   const uint32_t committed_meta1 = committed_meta0 + sizeof(uint32_t);
   const uint32_t candidate_meta0 = VT_RT_ABI_CANDIDATE_HIT_RECORD_BASE_BYTES;
   /* The two committed metadata words are restored from distinct parent SSA
    * defs. Candidate meta0 is not live after the child. */
   const unsigned baseline_restore_committed_meta0 =
      count_slot_ssa_restores_to_const_addr(nested_liveness_baseline, committed_meta0);
   const unsigned baseline_restore_committed_meta1 =
      count_slot_ssa_restores_to_const_addr(nested_liveness_baseline, committed_meta1);
   const unsigned baseline_restore_candidate_meta0 =
      count_slot_ssa_restores_to_const_addr(nested_liveness_baseline, candidate_meta0);
   const unsigned nested_restore_committed_meta0 =
      count_slot_ssa_restores_to_const_addr(nested_liveness_raygen, committed_meta0);
   const unsigned nested_restore_committed_meta1 =
      count_slot_ssa_restores_to_const_addr(nested_liveness_raygen, committed_meta1);
   const unsigned nested_restore_candidate_meta0 =
      count_slot_ssa_restores_to_const_addr(nested_liveness_raygen, candidate_meta0);
   VT_RT_CHECK(nested_restore_committed_meta0 == baseline_restore_committed_meta0 + 1);
   VT_RT_CHECK(nested_restore_committed_meta1 == baseline_restore_committed_meta1 + 1);
   /* Packed meta0 is read-modify-written, so it is live across the nested
    * trace just like the committed words above. */
   VT_RT_CHECK(nested_restore_candidate_meta0 == baseline_restore_candidate_meta0 + 1);
   nir_validate_shader(nested_liveness_raygen,
                       "after nested PDS liveness lowering");

   struct vt_nir_rt_abi_layout abi = vt_nir_rt_default_abi_layout();
   VT_RT_CHECK(abi.control_base_bytes >=
               abi.ray_state_base_bytes + abi.ray_state_size_bytes);
   VT_RT_CHECK(abi.candidate_hit_record_base_bytes >=
               abi.control_base_bytes + abi.control_size_bytes);
   VT_RT_CHECK(abi.hit_record_base_bytes >=
               abi.candidate_hit_record_base_bytes +
               abi.candidate_hit_record_size_bytes);
   VT_RT_CHECK(abi.hit_attrib_base_bytes >=
               abi.hit_record_base_bytes + abi.hit_record_size_bytes);
   VT_RT_CHECK(abi.committed_hit_attrib_base_bytes >=
               abi.hit_attrib_base_bytes + abi.hit_attrib_size_bytes);
   VT_RT_CHECK(abi.payload_base_bytes >=
               abi.committed_hit_attrib_base_bytes + abi.hit_attrib_size_bytes);
   VT_RT_CHECK(abi.payload_size_bytes == 0);
   VT_RT_CHECK(VT_RT_CONTROL_WORD_COUNT == 1);
   VT_RT_CHECK(abi.control_size_bytes == VT_RT_CONTROL_WORD_COUNT * 4);
   VT_RT_CHECK(abi.candidate_hit_record_size_bytes == 36);
   VT_RT_CHECK(abi.hit_record_size_bytes == 36);
   VT_RT_CHECK(abi.ray_state_size_bytes == VT_RT_SLOT_WORD_COUNT * 4);
   VT_RT_CHECK(VT_RT_SLOT_LAUNCH_ID_X * 4 == 60);
   VT_RT_CHECK(VT_RT_SLOT_LAUNCH_ID_Y * 4 == 64);
   VT_RT_CHECK(VT_RT_SLOT_LAUNCH_ID_Z * 4 == 68);
   VT_RT_CHECK(VT_RT_ABI_VERSION == 9);
   VT_RT_CHECK(VT_RT_ABI_PDS_FIELD_MAJOR_PREFIX_SIZE_BYTES == 76);
   VT_RT_CHECK(VT_RT_ABI_PDS_FIELD_MAJOR_PREFIX_WORD_COUNT == 19);
   VT_RT_CHECK(VT_RT_ABI_PDS_LANE_MAJOR_HIT_RECORD_WORD_COUNT ==
               VT_RT_HIT_RECORD_WORD_COUNT);
   VT_RT_CHECK(abi.control_base_bytes == 72);
   VT_RT_CHECK(abi.control_size_bytes == 4);
   VT_RT_CHECK(abi.candidate_hit_record_base_bytes == 76);
   VT_RT_CHECK(abi.hit_record_base_bytes == 112);
   VT_RT_CHECK(abi.hit_attrib_base_bytes == 148);
   VT_RT_CHECK(abi.committed_hit_attrib_base_bytes == 148);
   VT_RT_CHECK(abi.payload_base_bytes == 148);
   VT_RT_CHECK(abi.region_size_bytes == 148);
   VT_RT_CHECK(abi.total_size_bytes == 148);
   VT_RT_CHECK(abi.total_size_bytes == VT_RT_ABI_REGION_SIZE_BYTES);
   VT_RT_CHECK(VT_RT_HIT_RECORD_WORD_COUNT * 4 ==
               abi.candidate_hit_record_size_bytes);
   VT_RT_CHECK(VT_RT_HIT_RECORD_META1 * 4 == 4);

   struct vt_rt_intrinsic_counts before;
   vt_rt_count_intrinsics(raygen, &before);
   VT_RT_CHECK(before.trace_ray == 1);

   void *cps_mem_ctx = ralloc_context(NULL);
   fprintf(stderr, "stage cps-terminal\n");
   struct vt_nir_rt_cps_info cps = {};
   VT_RT_CHECK(vt_nir_lower_rt_cps(cps_raygen, &cps, cps_mem_ctx));
   VT_RT_CHECK(cps.num_resume_shaders == 1);
   VT_RT_CHECK(cps.resume_shaders[0] != NULL);
   /* This shader ends immediately after traceRayEXT, so it has no live CPS
    * state.  The pass must report the real zero-sized frame, not invent one. */
   VT_RT_CHECK(cps.frame_size_bytes == 0);
   struct vt_rt_intrinsic_counts cps_after;
   vt_rt_count_intrinsics(cps_raygen, &cps_after);
   VT_RT_CHECK(cps_after.trace_ray == 0);

   struct vt_nir_rt_config global_config = vt_nir_rt_default_config();
   fprintf(stderr, "stage global-terminal\n");
   global_config.execution_model = VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER;
   /* The profile gate must not accidentally invoke any PDS ABI lower. */
   VT_RT_CHECK(!vt_nir_lower_rt_abi(global_raygen, &global_config));
   struct vt_nir_rt_cps_info global_cps = {};
   VT_RT_CHECK(vt_nir_lower_rt_cps(global_raygen, &global_cps, cps_mem_ctx));
   fprintf(stderr, "stage global-terminal-cps-done\n");
   VT_RT_CHECK(global_cps.num_resume_shaders == 1);
   VT_RT_CHECK(vt_nir_lower_rt_global_raygen(global_raygen, &global_config));
   fprintf(stderr, "stage global-terminal-lowered\n");
   nir_validate_shader(global_raygen, "after Ventus global RT raygen lowering");
   struct vt_rt_intrinsic_counts global_after;
   vt_rt_count_intrinsics(global_raygen, &global_after);
   VT_RT_CHECK(global_after.trace_ray == 0);
   VT_RT_CHECK(global_after.ventus_rt_enqueue == 1);
   VT_RT_CHECK(global_after.ventus_rt_traverse == 0);
   VT_RT_CHECK(global_after.ventus_rt_release == 0);
   VT_RT_CHECK(global_after.ventus_rt_load_slot == 0);
   VT_RT_CHECK(global_after.ventus_rt_store_slot == 0);
   VT_RT_CHECK(global_after.load_shared == 0);
   VT_RT_CHECK(global_after.store_shared == 0);
   VT_RT_CHECK(count_intrinsic(global_raygen,
                               nir_intrinsic_ventus_rt_worker_id) == 1);
   VT_RT_CHECK(global_raygen->scratch_size == 0);
   fprintf(stderr, "stage cps-loop\n");

   /* A loop with code after traceRayEXT must become a CPS resume, not a
    * terminal producer.  Lower both parts and require the re-trace path to
    * publish a second mailbox record through the same root CPS frame ABI. */
   struct vt_nir_rt_cps_info loop_cps = {};
   VT_RT_CHECK(vt_nir_lower_rt_cps(global_loop_raygen, &loop_cps, cps_mem_ctx));
   VT_RT_CHECK(loop_cps.num_resume_shaders == 1);
   VT_RT_CHECK(loop_cps.frame_size_bytes > 0);
   VT_RT_CHECK(vt_nir_lower_rt_global_raygen(global_loop_raygen, &global_config));
   fprintf(stderr, "stage global-loop-resume\n");
   VT_RT_CHECK(count_intrinsic(global_loop_raygen,
                               nir_intrinsic_rt_trace_ray) == 0);
   VT_RT_CHECK(count_intrinsic(global_loop_raygen,
                               nir_intrinsic_ventus_rt_enqueue) == 1);
   VT_RT_CHECK(vt_nir_lower_rt_global_resume(loop_cps.resume_shaders[0],
                                               &global_config));
   nir_validate_shader(loop_cps.resume_shaders[0],
                       "after Ventus global CPS loop resume lowering");
   VT_RT_CHECK(count_intrinsic(loop_cps.resume_shaders[0],
                               nir_intrinsic_rt_resume) == 0);
   VT_RT_CHECK(count_intrinsic(loop_cps.resume_shaders[0],
                               nir_intrinsic_rt_trace_ray) == 0);
   VT_RT_CHECK(count_intrinsic(loop_cps.resume_shaders[0],
                               nir_intrinsic_ventus_rt_enqueue) >= 1);
   fprintf(stderr, "stage legacy\n");

   bool progress = vt_nir_lower_rt_io_monolithic(raygen, &config);
   VT_RT_CHECK(progress);

   progress = vt_nir_lower_rt_abi(raygen, &config);
   VT_RT_CHECK(!progress || raygen->scratch_size >= 0);
   VT_RT_CHECK(raygen->scratch_size >= abi.total_size_bytes);

   struct vt_rt_intrinsic_counts after;
   vt_rt_count_intrinsics(raygen, &after);
   VT_RT_CHECK(after.trace_ray == 0);
   VT_RT_CHECK(vt_rt_count_loops(raygen) >= 1);
   VT_RT_CHECK(after.jump_break >= 1);
   VT_RT_CHECK(after.ventus_rt_pds_warp_tid_base == 2);
   VT_RT_CHECK(after.ventus_rt_traverse == 1);
   VT_RT_CHECK(after.ventus_rt_release == 1);
   VT_RT_CHECK(vt_rt_all_releases_are_outside_loops(raygen));
   VT_RT_CHECK(after.ventus_rt_store_slot > 0);
   VT_RT_CHECK(after.ventus_rt_load_slot > 0);
   VT_RT_CHECK(vt_rt_shader_has_imul_any(raygen));

   VT_RT_CHECK(count_intrinsic(push_constant_probe,
                               nir_intrinsic_load_push_constant) == 1);
   progress = vt_nir_lower_rt_io_monolithic(push_constant_probe, &config);
   VT_RT_CHECK(progress);
   VT_RT_CHECK(count_intrinsic(push_constant_probe,
                               nir_intrinsic_load_push_constant) == 0);
   VT_RT_CHECK(count_intrinsic(push_constant_probe,
                               nir_intrinsic_ventus_rt_dispatch_arg) == 1);
   VT_RT_CHECK(count_intrinsic(push_constant_probe,
                               nir_intrinsic_load_global) == 1);

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
   VT_RT_CHECK(closest_after.ventus_rt_load_slot >
               closest_before.ventus_rt_load_slot);
   VT_RT_CHECK(closest_after.ventus_rt_store_slot >
               closest_before.ventus_rt_store_slot);
   VT_RT_CHECK(count_variables(closest_hit, nir_var_shader_call_data) == 0);

   /* Terminal global closest-hit consumes a compact ray_ref list, reads the
    * queue/completion planes, and writes the original payload address. */
   progress = vt_nir_lower_rt_abi(global_callback_closest_hit, &config);
   VT_RT_CHECK(progress);
   struct vt_nir_rt_config callback_config = vt_nir_rt_default_config();
   callback_config.execution_model = VT_RT_EXECUTION_WAVEFRONT_GLOBAL_CALLBACK;
   uint32_t callback_lds_size = UINT32_MAX;
   VT_RT_CHECK(vt_nir_lower_rt_global_callback(global_callback_closest_hit,
                                                &callback_config,
                                                &callback_lds_size));
   /* In the Megakernel path this terminal closest-hit remains PDS-native and
    * does not reserve legacy callback LDS. */
   VT_RT_CHECK(callback_lds_size == 0);
   nir_validate_shader(global_callback_closest_hit,
                       "after Ventus global terminal closest-hit lowering");
   struct vt_rt_intrinsic_counts callback_after;
   vt_rt_count_intrinsics(global_callback_closest_hit, &callback_after);
   /* These dynamic addresses are per-lane PDS accesses.  They are not folded
    * to the legacy global completion plane. */
   VT_RT_CHECK(callback_after.ventus_rt_load_slot == 4);
   VT_RT_CHECK(callback_after.ventus_rt_store_slot == 4);
   VT_RT_CHECK(count_intrinsic(global_callback_closest_hit,
                               nir_intrinsic_ventus_rt_dispatch_index) > 0);
   VT_RT_CHECK(count_intrinsic(global_callback_closest_hit,
                               nir_intrinsic_load_global) > 0);
   VT_RT_CHECK(count_intrinsic(global_callback_closest_hit,
                               nir_intrinsic_store_global) == 0);

   progress = vt_nir_lower_rt_abi(global_callback_any_hit, &config);
   VT_RT_CHECK(progress);
   callback_lds_size = 0;
   VT_RT_CHECK(vt_nir_lower_rt_global_callback(global_callback_any_hit,
                                                &callback_config,
                                                &callback_lds_size));
   VT_RT_CHECK(callback_lds_size == 32 * 144);
   nir_validate_shader(global_callback_any_hit,
                       "after Ventus global any-hit LDS lowering");
   VT_RT_CHECK(count_intrinsic(global_callback_any_hit,
                               nir_intrinsic_load_shared) > 0);
   VT_RT_CHECK(count_intrinsic(global_callback_any_hit,
                               nir_intrinsic_store_shared) > 0);

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
   VT_RT_CHECK(any_after.ventus_rt_load_slot > any_before.ventus_rt_load_slot);
   VT_RT_CHECK(any_after.ventus_rt_store_slot > any_before.ventus_rt_store_slot);

   struct vt_rt_intrinsic_counts terminating_any_before;
   vt_rt_count_intrinsics(terminating_any_hit, &terminating_any_before);
   VT_RT_CHECK(terminating_any_before.terminate_ray == 1);

   progress = vt_nir_lower_rt_abi(terminating_any_hit, &config);
   VT_RT_CHECK(progress);
   VT_RT_CHECK(terminating_any_hit->scratch_size >= abi.total_size_bytes);

   struct vt_rt_intrinsic_counts terminating_any_after;
   vt_rt_count_intrinsics(terminating_any_hit, &terminating_any_after);
   VT_RT_CHECK(terminating_any_after.terminate_ray == 0);
   VT_RT_CHECK(terminating_any_after.ventus_rt_store_slot >
               terminating_any_before.ventus_rt_store_slot);

   struct vt_rt_intrinsic_counts intersection_before;
   vt_rt_count_intrinsics(intersection, &intersection_before);
   VT_RT_CHECK(intersection_before.report_ray_intersection == 1);

   progress = vt_nir_lower_rt_abi(intersection, &config);
   VT_RT_CHECK(progress);
   VT_RT_CHECK(intersection->scratch_size >= abi.total_size_bytes);

   struct vt_rt_intrinsic_counts intersection_after;
   vt_rt_count_intrinsics(intersection, &intersection_after);
   VT_RT_CHECK(intersection_after.report_ray_intersection == 0);
   VT_RT_CHECK(intersection_after.ventus_rt_load_slot >
               intersection_before.ventus_rt_load_slot);
   VT_RT_CHECK(intersection_after.ventus_rt_store_slot >
               intersection_before.ventus_rt_store_slot);

   struct vt_nir_rt_config deferred_config = config;
   deferred_config.execution_model = VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER;
   deferred_config.defer_intersection_reports = true;
   struct vt_rt_intrinsic_counts deferred_intersection_before;
   vt_rt_count_intrinsics(deferred_intersection, &deferred_intersection_before);
   progress = vt_nir_lower_rt_abi(deferred_intersection, &deferred_config);
   VT_RT_CHECK(!progress);
   struct vt_rt_intrinsic_counts deferred_intersection_after;
   vt_rt_count_intrinsics(deferred_intersection, &deferred_intersection_after);
   /* The per-group callback builder must still see the original report so it
    * can inline the matching any-hit shader at precisely this program point. */
   VT_RT_CHECK(deferred_intersection_after.report_ray_intersection == 1);
   VT_RT_CHECK(deferred_intersection_after.ventus_rt_load_slot ==
               deferred_intersection_before.ventus_rt_load_slot);

   progress = vt_nir_lower_rt_payload(payload_location_probe, &config);
   VT_RT_CHECK(progress);
   uint32_t absolute_payload_base = abi.payload_base_bytes;
   VT_RT_CHECK(count_store_slot_to_const_addr(payload_location_probe,
                                             absolute_payload_base + 0) == 1);
   VT_RT_CHECK(count_store_slot_to_const_addr(payload_location_probe,
                                             absolute_payload_base + 4) == 1);
   VT_RT_CHECK(count_store_slot_to_const_addr(payload_location_probe,
                                             absolute_payload_base + 8) == 1);
   VT_RT_CHECK(count_store_slot_to_const_addr(payload_location_probe,
                                             absolute_payload_base + 32) == 1);

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
          "rt_traverse=1 rt_release=1 traversal_resumable=1 "
          "cps_resume_shaders=%u cps_frame=%u "
          "abi_candidate_committed=1 abi_control=1 sbt_entry_calc=1 "
          "hit_sbt_base=0x%llx "
          "hit_sbt_stride=%u shader_group_handle_size=%u "
          "rt_slot_store=%u rt_slot_load=%u shared_store=%u shared_load=%u "
          "per_lane_scratch=%u "
          "abi_total=%u control_base=%u candidate_base=%u committed_base=%u "
          "hit_attrib_base=%u payload_base=%u manual_max_scratch=%u\n",
          cps.num_resume_shaders, cps.frame_size_bytes,
          (unsigned long long)config.hit_sbt_base_addr,
          config.hit_sbt_record_stride, config.shader_group_handle_size,
          after.ventus_rt_store_slot + closest_after.ventus_rt_store_slot +
             any_after.ventus_rt_store_slot +
             terminating_any_after.ventus_rt_store_slot +
             intersection_after.ventus_rt_store_slot,
          after.ventus_rt_load_slot + closest_after.ventus_rt_load_slot +
             any_after.ventus_rt_load_slot +
             intersection_after.ventus_rt_load_slot,
          after.store_shared + closest_after.store_shared +
             any_after.store_shared + terminating_any_after.store_shared +
             intersection_after.store_shared,
          after.load_shared + closest_after.load_shared +
             any_after.load_shared + intersection_after.load_shared,
          per_lane_scratch, abi.total_size_bytes, abi.control_base_bytes,
          abi.candidate_hit_record_base_bytes, abi.hit_record_base_bytes,
          abi.hit_attrib_base_bytes, abi.payload_base_bytes,
          manual_per_lane_scratch);

   ralloc_free(raygen);
   ralloc_free(nested_liveness_raygen);
   ralloc_free(nested_liveness_baseline);
   ralloc_free(cps_mem_ctx);
   ralloc_free(global_raygen);
   ralloc_free(global_callback_closest_hit);
   ralloc_free(global_callback_any_hit);
   ralloc_free(global_loop_raygen);
   ralloc_free(miss);
   ralloc_free(closest_hit);
   ralloc_free(any_hit);
   ralloc_free(intersection);
   ralloc_free(terminating_any_hit);
   ralloc_free(push_constant_probe);
   ralloc_free(manual_raygen);
   ralloc_free(manual_miss);
   ralloc_free(manual_closest_hit);
   glsl_type_singleton_decref();
   return 0;
}
