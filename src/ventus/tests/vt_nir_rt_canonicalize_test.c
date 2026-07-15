/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_rt_test_common.h"

static unsigned
count_function_temp_derefs(nir_shader *shader)
{
   unsigned count = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            unsigned src_count = nir_intrinsic_infos[intrin->intrinsic].num_srcs;
            for (unsigned src = 0; src < src_count; ++src) {
               if (!nir_src_is_deref(intrin->src[src]))
                  continue;
               nir_variable *var =
                  nir_deref_instr_get_variable(nir_src_as_deref(intrin->src[src]));
               count += var && var->data.mode == nir_var_function_temp;
            }
         }
      }
   }

   return count;
}

static nir_shader *
build_short_circuit_shader(const nir_shader_compiler_options *options)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_RAYGEN, options,
                                                   "vt_short_circuit");
   nir_variable *result =
      nir_local_variable_create(b.impl, glsl_bool_type(), "short_circuit");
   nir_def *launch_id = nir_load_ray_launch_id(&b);
   nir_def *first = nir_ine_imm(&b, nir_channel(&b, launch_id, 0), 0);
   nir_def *second = nir_ine_imm(&b, nir_channel(&b, launch_id, 1), 0);

   nir_store_var(&b, result, nir_imm_false(&b), 0x1);
   nir_push_if(&b, first);
   nir_store_var(&b, result, second, 0x1);
   nir_pop_if(&b, NULL);

   nir_ventus_rt_store_slot(&b, nir_b2i32(&b, nir_load_var(&b, result)),
                             nir_imm_int(&b, 0));
   return b.shader;
}

static nir_shader *
build_indirect_local_shader(const nir_shader_compiler_options *options)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_RAYGEN, options,
                                                   "vt_indirect_local");
   nir_variable *values = nir_local_variable_create(
      b.impl, glsl_array_type(glsl_bool_type(), 4, 0), "dynamic_local");
   nir_def *launch_id = nir_load_ray_launch_id(&b);
   nir_def *index = nir_umin_imm(&b, nir_channel(&b, launch_id, 0), 3);
   nir_deref_instr *element = nir_build_deref_array(
      &b, nir_build_deref_var(&b, values), index);

   nir_store_deref(&b, element, nir_imm_true(&b), 0x1);
   nir_ventus_rt_store_slot(&b,
                             nir_b2i32(&b, nir_load_deref(&b, element)),
                             nir_imm_int(&b, 0));
   return b.shader;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   const nir_shader_compiler_options options = {0};
   struct vt_nir_rt_canonicalize_error error = {};

   nir_shader *short_circuit = build_short_circuit_shader(&options);
   VT_RT_CHECK(count_function_temp_derefs(short_circuit) > 0);
   VT_RT_CHECK(vt_nir_canonicalize_rt_for_llvm(short_circuit, &error) ==
               VT_NIR_RT_CANONICALIZE_SUCCESS);
   VT_RT_CHECK(error.variable_name == NULL);
   VT_RT_CHECK(count_function_temp_derefs(short_circuit) == 0);
   nir_validate_shader(short_circuit, "Ventus RT canonical short circuit");

   nir_shader *indirect_local = build_indirect_local_shader(&options);
   VT_RT_CHECK(vt_nir_canonicalize_rt_for_llvm(indirect_local, &error) ==
               VT_NIR_RT_CANONICALIZE_RESIDUAL_FUNCTION_TEMP);
   VT_RT_CHECK(error.variable_name != NULL);
   nir_validate_shader(indirect_local, "Ventus RT canonical residual local");

   ralloc_free(short_circuit);
   ralloc_free(indirect_local);
   glsl_type_singleton_decref();
   return 0;
}
