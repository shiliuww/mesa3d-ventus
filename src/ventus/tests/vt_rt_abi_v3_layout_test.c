/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 *
 * Keep the Megakernel PDS ABI test separate from the global queue/CPS tests:
 * v5 reserves a fixed RTcore header at offset zero and sizes the payload from
 * the linked Vulkan RT shader interfaces before LLVM private spilling.
 */

#include "vt_rt_test_common.h"

static nir_shader *
build_payload_shader(mesa_shader_stage stage, nir_variable_mode mode,
                     const struct glsl_type *type, int location,
                     const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(stage, NULL, "%s", name);
   nir_variable *payload = nir_variable_create(b.shader, mode, type, name);
   payload->data.explicit_location = true;
   payload->data.location = location;
   return b.shader;
}

static nir_shader *
build_payload_store_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_CLOSEST_HIT,
                                                   NULL,
                                                   "vt_rt_v4_payload_store");
   nir_variable *payload =
      nir_variable_create(b.shader, nir_var_shader_call_data, glsl_uvec4_type(),
                          "payload_in");
   payload->data.explicit_location = true;
   payload->data.location = 2;
   nir_store_var(&b, payload,
                 nir_vec4(&b, nir_imm_int(&b, 1), nir_imm_int(&b, 2),
                          nir_imm_int(&b, 3), nir_imm_int(&b, 4)),
                 0xf);
   return b.shader;
}

static nir_shader *
build_report_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_INTERSECTION,
                                                   NULL,
                                                   "vt_rt_report_semantics");
   nir_report_ray_intersection(&b, 1, nir_imm_float(&b, 1.0f),
                               nir_imm_int(&b, 0xff));
   return b.shader;
}

static nir_shader *
build_two_report_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_INTERSECTION,
                                                   NULL,
                                                   "vt_rt_two_report_semantics");
   nir_report_ray_intersection(&b, 1, nir_imm_float(&b, 1.0f),
                               nir_imm_int(&b, 0xff));
   nir_report_ray_intersection(&b, 1, nir_imm_float(&b, 2.0f),
                               nir_imm_int(&b, 0xfe));
   return b.shader;
}

static unsigned
count_store_to_const_addr(nir_shader *shader, nir_intrinsic_op op,
                          uint32_t addr)
{
   unsigned count = 0;

   NIR_PASS(_, shader, nir_opt_constant_folding);
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != op)
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
count_store_slot_to_const_addr(nir_shader *shader, uint32_t addr)
{
   return count_store_to_const_addr(shader, nir_intrinsic_ventus_rt_store_slot,
                                    addr);
}

static unsigned
count_store_slots(nir_shader *shader)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic == nir_intrinsic_ventus_rt_store_slot)
               count++;
         }
      }
   }
   return count;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   struct vt_nir_rt_abi_layout layout = vt_nir_rt_default_abi_layout();
   VT_RT_CHECK(VT_RT_ABI_VERSION == 11);
   VT_RT_CHECK(VT_RT_LOCAL_FIELD_MAJOR_FIXED_HEADER_WORD_COUNT == 37);
   VT_RT_CHECK(layout.ray_state_base_bytes == 0);
   VT_RT_CHECK(layout.payload_base_bytes == 0);
   VT_RT_CHECK(layout.region_size_bytes == 0);
   VT_RT_CHECK(layout.private_base_bytes == 0);
   VT_RT_CHECK(layout.payload_size_bytes == 0);
   VT_RT_CHECK(layout.total_size_bytes == 0);

   struct vt_nir_rt_abi_layout direct_barycentrics =
      vt_nir_rt_default_abi_layout();
   VT_RT_CHECK(vt_nir_rt_abi_layout_set_hit_attrib_size(
      &direct_barycentrics, 0, true));
   /* Triangle barycentrics are read from the committed hit record. */
   VT_RT_CHECK(direct_barycentrics.hit_attrib_size_bytes == 0);
   VT_RT_CHECK(direct_barycentrics.payload_base_bytes == 0);

   struct vt_nir_rt_abi_layout custom_hit_attribs =
      vt_nir_rt_default_abi_layout();
   VT_RT_CHECK(vt_nir_rt_abi_layout_set_hit_attrib_size(
      &custom_hit_attribs, 12, false));
   VT_RT_CHECK(custom_hit_attribs.payload_base_bytes ==
               VT_RT_PDS_PAYLOAD_BASE_BYTES(12));
   VT_RT_CHECK(custom_hit_attribs.payload_base_bytes == 2 * 12);

   uint32_t payload_size = UINT32_MAX;
   VT_RT_CHECK(vt_nir_rt_compute_pipeline_payload_size(NULL, 0,
                                                        &payload_size));
   VT_RT_CHECK(payload_size == 0);

   nir_shader *raygen =
      build_payload_shader(MESA_SHADER_RAYGEN, nir_var_shader_temp,
                           glsl_vec_type(3), 0, "raygen_payload");
   nir_shader *closest_hit =
      build_payload_shader(MESA_SHADER_CLOSEST_HIT, nir_var_shader_call_data,
                           glsl_uvec4_type(), 2, "closest_hit_payload");
   nir_shader *payload_pipeline[] = { raygen, closest_hit };
   VT_RT_CHECK(vt_nir_rt_compute_pipeline_payload_size(payload_pipeline, 2,
                                                        &payload_size));
   /* max(vec3@0: 12, uvec4@2: 32 + 16), rounded up to the ABI alignment. */
   VT_RT_CHECK(payload_size == 48);

   nir_shader *sparse =
      build_payload_shader(MESA_SHADER_MISS, nir_var_shader_call_data,
                           glsl_uvec4_type(), 4, "sparse_payload");
   VT_RT_CHECK(vt_nir_rt_compute_pipeline_payload_size(&sparse, 1,
                                                        &payload_size));
   /* A sparse payload location must reserve its complete location range. */
   VT_RT_CHECK(payload_size == 80);

   VT_RT_CHECK(vt_nir_rt_abi_layout_set_payload_size(&layout, payload_size));
   struct vt_nir_rt_config config = vt_nir_rt_default_config();
   vt_nir_rt_config_apply_abi_layout(&config, &layout);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].base_bytes == 0);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].payload_base_bytes == 0);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].total_size_bytes == 80);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].private_base_bytes == 80);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_CLOSEST_HIT].total_size_bytes == 80);

   nir_shader *report_shader = build_report_shader();
   VT_RT_CHECK(vt_nir_lower_rt_control(report_shader, &config));
   /* reportIntersectionEXT may update only report-specific fields; it cannot
    * overwrite RTcore-provided raw OPAQUE or effective-opacity metadata. */
   VT_RT_CHECK(count_store_slot_to_const_addr(report_shader, 180) == 0);
   VT_RT_CHECK(count_store_slot_to_const_addr(report_shader, 184) == 0);

   nir_shader *two_report_shader = build_two_report_shader();
   VT_RT_CHECK(vt_nir_lower_rt_control(two_report_shader, &config));
   /* Every report site remains independently lowered.  Committing the accepted
    * report and running any-hit is deliberately tested later in the pipeline,
   * outside this stack-free ABI test. */
   VT_RT_CHECK(count_store_to_const_addr(
      two_report_shader, nir_intrinsic_ventus_rt_local_store,
      VT_RT_LOCAL_CANDIDATE_HIT_RECORD_BASE_BYTES / 4 +
      VT_RT_HIT_RECORD_META0) > 0);
   VT_RT_CHECK(count_store_to_const_addr(
      two_report_shader, nir_intrinsic_ventus_rt_local_store,
      VT_RT_LOCAL_CANDIDATE_HIT_RECORD_BASE_BYTES / 4 +
      VT_RT_HIT_RECORD_HIT_T) > 0);
   VT_RT_CHECK(count_store_to_const_addr(
      two_report_shader, nir_intrinsic_ventus_rt_local_store,
      VT_RT_LOCAL_CANDIDATE_HIT_RECORD_BASE_BYTES / 4 +
      VT_RT_HIT_RECORD_HIT_KIND) > 0);

   nir_shader *deferred_report_shader = build_report_shader();
   struct vt_nir_rt_config deferred_config = config;
   deferred_config.defer_intersection_reports = true;
   VT_RT_CHECK(!vt_nir_lower_rt_control(deferred_report_shader,
                                        &deferred_config));
   struct vt_rt_intrinsic_counts deferred_counts;
   vt_rt_count_intrinsics(deferred_report_shader, &deferred_counts);
   VT_RT_CHECK(deferred_counts.report_ray_intersection == 1);

   nir_shader *deferred_two_report_shader = build_two_report_shader();
   VT_RT_CHECK(!vt_nir_lower_rt_control(deferred_two_report_shader,
                                        &deferred_config));
   vt_rt_count_intrinsics(deferred_two_report_shader, &deferred_counts);
   VT_RT_CHECK(deferred_counts.report_ray_intersection == 2);

   VT_RT_CHECK(vt_nir_rt_abi_layout_set_hit_attrib_size(&layout, 32, true));
   VT_RT_CHECK(vt_nir_rt_abi_layout_set_payload_size(&layout, 48));
   VT_RT_CHECK(vt_nir_rt_abi_layout_set_recursion_depth(&layout, 1));
   /* Depth zero uses the active payload slot in region_size_bytes. */
   VT_RT_CHECK(layout.payload_spill_size_bytes == 0);
   VT_RT_CHECK(layout.private_base_bytes == 112);
   VT_RT_CHECK(vt_nir_rt_abi_layout_set_recursion_depth(&layout, 2));
   VT_RT_CHECK(!vt_nir_rt_abi_layout_set_recursion_depth(
      &layout, VT_RT_TRACE_META1_MAX_RECURSION_DEPTH + 1));
   VT_RT_CHECK(layout.committed_hit_attrib_base_bytes == 32);
   VT_RT_CHECK(layout.payload_spill_base_bytes == 112);
   VT_RT_CHECK(layout.payload_spill_size_bytes == 48);
   VT_RT_CHECK(layout.private_base_bytes == 160);
   VT_RT_CHECK(layout.total_size_bytes == 160);
   vt_nir_rt_config_apply_abi_layout(&config, &layout);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].payload_spill_base_bytes == 112);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].payload_spill_size_bytes == 48);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].max_ray_recursion_depth == 2);

   nir_shader *store_shader = build_payload_store_shader();
   VT_RT_CHECK(vt_nir_rt_abi_layout_set_payload_size(&layout, 48));
   VT_RT_CHECK(vt_nir_rt_abi_layout_set_recursion_depth(&layout, 2));
   vt_nir_rt_config_apply_abi_layout(&config, &layout);
   VT_RT_CHECK(vt_nir_lower_rt_payload(store_shader, &config));
   /* Interface payload uses payload_base + active_depth * P + location * 16,
    * so these stores must not collapse to the depth-zero addresses 284..296. */
   VT_RT_CHECK(count_store_slot_to_const_addr(store_shader, 284) == 0);
   VT_RT_CHECK(count_store_slot_to_const_addr(store_shader, 288) == 0);
   VT_RT_CHECK(count_store_slot_to_const_addr(store_shader, 292) == 0);
   VT_RT_CHECK(count_store_slot_to_const_addr(store_shader, 296) == 0);
   VT_RT_CHECK(count_store_slots(store_shader) == 4);
   VT_RT_CHECK(vt_rt_shader_has_imul_any(store_shader));

   printf("PASS vt_rt_abi_v10_layout rt_local_header=148 payload_union=48 "
          "payload_sparse=80 pds_private_base=160\n");

   ralloc_free(raygen);
   ralloc_free(closest_hit);
   ralloc_free(sparse);
   ralloc_free(store_shader);
   ralloc_free(report_shader);
   ralloc_free(two_report_shader);
   ralloc_free(deferred_report_shader);
   ralloc_free(deferred_two_report_shader);
   glsl_type_singleton_decref();
   return 0;
}
