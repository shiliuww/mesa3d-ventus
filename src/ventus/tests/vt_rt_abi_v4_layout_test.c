/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 *
 * Keep the MegaKernel PDS ABI test separate from the global queue/CPS tests:
 * V4 reserves a fixed RTcore header at offset zero and sizes the payload from
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
build_split_call_data_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_RAYGEN, NULL, "vt_rt_split_call_data");
   nir_variable *payload = nir_variable_create(
      b.shader, nir_var_shader_temp, glsl_uvec4_type(), "ray_payload");
   payload->data.explicit_location = true;
   payload->data.location = 1;
   nir_variable *call_data = nir_variable_create(
      b.shader, nir_var_shader_temp, glsl_vec2_type(), "callable_data");
   call_data->data.explicit_location = true;
   call_data->data.location = 3;

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *zero64 = nir_imm_int64(&b, 0);
   nir_def *zero3 = nir_replicate(&b, zero, 3);
   nir_trace_ray(&b, zero64, zero, zero, zero, zero, zero,
                 zero3, zero, zero3, zero,
                 &nir_build_deref_var(&b, payload)->def);
   nir_execute_callable(&b, zero,
                        &nir_build_deref_var(&b, call_data)->def);
   return b.shader;
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
   VT_RT_CHECK(VT_RT_ABI_VERSION == 4);
   VT_RT_CHECK(layout.ray_state_base_bytes == 0);
   VT_RT_CHECK(layout.payload_base_bytes == 276);
   VT_RT_CHECK(layout.region_size_bytes == 276);
   VT_RT_CHECK(layout.private_base_bytes == 276);
   VT_RT_CHECK(layout.payload_size_bytes == 0);
   VT_RT_CHECK(layout.total_size_bytes == 276);

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
   VT_RT_CHECK(payload_size == 48);

   nir_shader *split_call_data = build_split_call_data_shader();
   nir_shader *callable_input =
      build_payload_shader(MESA_SHADER_CALLABLE, nir_var_shader_call_data,
                           glsl_uvec4_type(), 3, "callable_input");
   nir_shader *split_pipeline[] = {split_call_data, closest_hit,
                                   callable_input};
   uint32_t split_payload_size = 0;
   uint32_t callable_data_size = 0;
   VT_RT_CHECK(vt_nir_rt_compute_pipeline_call_data_sizes(
      split_pipeline, 3, &split_payload_size, &callable_data_size));
   VT_RT_CHECK(split_payload_size == 32);
   VT_RT_CHECK(callable_data_size == 64);

   nir_shader *sparse =
      build_payload_shader(MESA_SHADER_MISS, nir_var_shader_call_data,
                           glsl_uvec4_type(), 4, "sparse_payload");
   VT_RT_CHECK(vt_nir_rt_compute_pipeline_payload_size(&sparse, 1,
                                                        &payload_size));
   VT_RT_CHECK(payload_size == 80);

   VT_RT_CHECK(vt_nir_rt_abi_layout_set_payload_size(&layout, payload_size));
   struct vt_nir_rt_config config = vt_nir_rt_default_config();
   vt_nir_rt_config_apply_abi_layout(&config, &layout);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].base_bytes == 0);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].payload_base_bytes == 276);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].total_size_bytes == 356);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].private_base_bytes == 356);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_CLOSEST_HIT].total_size_bytes == 356);

   VT_RT_CHECK(vt_nir_rt_abi_layout_set_hit_attrib_size(&layout, 32, true));
   VT_RT_CHECK(vt_nir_rt_abi_layout_set_payload_size(&layout, 48));
   VT_RT_CHECK(vt_nir_rt_abi_layout_set_recursion_depth(&layout, 2));
   VT_RT_CHECK(layout.payload_spill_base_bytes == 356);
   VT_RT_CHECK(layout.payload_spill_size_bytes == 96);
   VT_RT_CHECK(layout.private_base_bytes == 452);
   VT_RT_CHECK(layout.total_size_bytes == 452);
   vt_nir_rt_config_apply_abi_layout(&config, &layout);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].payload_spill_base_bytes == 356);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].payload_spill_size_bytes == 96);
   VT_RT_CHECK(config.scratchpad[VT_RT_LAYOUT_RAYGEN].max_ray_recursion_depth == 2);

   nir_shader *store_shader = build_payload_store_shader();
   VT_RT_CHECK(vt_nir_rt_abi_layout_set_payload_size(&layout, 48));
   vt_nir_rt_config_apply_abi_layout(&config, &layout);
   VT_RT_CHECK(vt_nir_lower_rt_payload(store_shader, &config));
   VT_RT_CHECK(count_store_slot_to_const_addr(store_shader, 340) == 0);
   VT_RT_CHECK(count_store_slot_to_const_addr(store_shader, 344) == 0);
   VT_RT_CHECK(count_store_slot_to_const_addr(store_shader, 348) == 0);
   VT_RT_CHECK(count_store_slot_to_const_addr(store_shader, 352) == 0);
   VT_RT_CHECK(count_store_slots(store_shader) == 4);

   printf("PASS vt_rt_abi_v4_layout header=276 hitattr=32 "
          "payload_union=48 payload_sparse=80 private_base=452\n");

   ralloc_free(raygen);
   ralloc_free(closest_hit);
   ralloc_free(sparse);
   ralloc_free(store_shader);
   ralloc_free(split_call_data);
   ralloc_free(callable_input);
   glsl_type_singleton_decref();
   return 0;
}
