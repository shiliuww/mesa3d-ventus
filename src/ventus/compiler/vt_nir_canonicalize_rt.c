/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

static nir_variable *
residual_function_temp(nir_shader *shader)
{
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
               if (var && var->data.mode == nir_var_function_temp)
                  return var;
            }
         }
      }
   }

   return NULL;
}

enum vt_nir_rt_canonicalize_status
vt_nir_canonicalize_rt_for_llvm(
   nir_shader *shader,
   struct vt_nir_rt_canonicalize_error *error)
{
   if (error)
      error->variable_name = NULL;
   if (!shader)
      return VT_NIR_RT_CANONICALIZE_SUCCESS;

   /* Callback inlining and trace-ray lowering introduce short-circuit locals.
    * Keep them as NIR SSA/phi values instead of LLVM allocas. */
   NIR_PASS(_, shader, nir_lower_returns);
   NIR_PASS(_, shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, shader, nir_opt_copy_prop);
   NIR_PASS(_, shader, nir_opt_dce);
   NIR_PASS(_, shader, nir_opt_dead_cf);
   NIR_PASS(_, shader, nir_opt_cse);
   NIR_PASS(_, shader, nir_opt_copy_prop);
   NIR_PASS(_, shader, nir_opt_dce);

   nir_variable *residual = residual_function_temp(shader);
   if (!residual)
      return VT_NIR_RT_CANONICALIZE_SUCCESS;

   if (error)
      error->variable_name = residual->name;
   return VT_NIR_RT_CANONICALIZE_RESIDUAL_FUNCTION_TEMP;
}
