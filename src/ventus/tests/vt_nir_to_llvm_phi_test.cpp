/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstdlib>

#include "compiler/glsl_types.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"

#include "ventus/llvm/ventus_nir_to_llvm.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#define VT_RT_CHECK(condition)                                                \
   do {                                                                      \
      if (!(condition)) {                                                    \
         std::fprintf(stderr, "CHECK failed: %s at %s:%d\n", #condition,  \
                      __FILE__, __LINE__);                                   \
         std::abort();                                                       \
      }                                                                      \
   } while (0)

static nir_shader *
build_float_phi_shader(const nir_shader_compiler_options *options)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_RAYGEN, options,
                                                   "vt_float_phi");
   nir_def *launch_id = nir_load_ray_launch_id(&b);
   nir_if *nif =
      nir_push_if(&b, nir_ieq_imm(&b, nir_channel(&b, launch_id, 0), 0));
   nir_def *then_value = nir_imm_float(&b, -1.0f);
   nir_push_else(&b, nif);
   nir_def *else_value = nir_fsub(
      &b, nir_u2f32(&b, nir_channel(&b, launch_id, 0)),
      nir_imm_float(&b, 1.0f));
   nir_pop_if(&b, nif);

   nir_def *merged = nir_if_phi(&b, then_value, else_value);
   nir_ventus_rt_store_slot(&b, merged, nir_imm_int(&b, 0));
   return b.shader;
}

static nir_shader *
build_loop_exit_phi_shader(const nir_shader_compiler_options *options)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_RAYGEN, options,
                                                   "vt_loop_exit_phi");
   nir_variable *color =
      nir_local_variable_create(b.impl, glsl_vec_type(3), "color");
   nir_def *launch_id = nir_load_ray_launch_id(&b);

   nir_loop *loop = nir_push_loop(&b);
   nir_if *nif =
      nir_push_if(&b, nir_ieq_imm(&b, nir_channel(&b, launch_id, 0), 0));
   nir_store_var(&b, color,
                 nir_vec3(&b, nir_imm_float(&b, 1.0f),
                          nir_imm_float(&b, 0.5f),
                          nir_imm_float(&b, 0.25f)),
                 0x7);
   nir_jump(&b, nir_jump_break);
   nir_push_else(&b, nif);
   nir_store_var(&b, color,
                 nir_vec3(&b, nir_imm_float(&b, 0.25f),
                          nir_imm_float(&b, 0.5f),
                          nir_imm_float(&b, 1.0f)),
                 0x7);
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, nif);
   nir_pop_loop(&b, loop);

   nir_def *merged = nir_load_var(&b, color);
   nir_ventus_rt_store_slot(&b, merged, nir_imm_int(&b, 0));
   nir_lower_vars_to_ssa(b.shader);
   return b.shader;
}

static nir_shader *
build_storage_image_shader(const nir_shader_compiler_options *options,
                           uint32_t descriptor_set, uint32_t binding)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_RAYGEN, options, "vt_sparse_storage_image");
   const glsl_type *image_type =
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT);
   nir_variable *output =
      nir_variable_create(b.shader, nir_var_image, image_type, "output");
   output->data.descriptor_set = descriptor_set;
   output->data.binding = binding;

   nir_def *coord =
      nir_vec4(&b, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
               nir_imm_int(&b, 0), nir_imm_int(&b, 0));
   nir_def *rgba =
      nir_vec4(&b, nir_imm_float(&b, 1.0f), nir_imm_float(&b, 0.5f),
               nir_imm_float(&b, 0.25f), nir_imm_float(&b, 1.0f));
   nir_image_deref_store(
      &b, &nir_build_deref_var(&b, output)->def, coord,
      nir_undef(&b, 1, 32), rgba, nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_2D);
   return b.shader;
}

int
main()
{
   glsl_type_singleton_init_or_ref();
   const nir_shader_compiler_options nir_options = {};
   nir_shader *shader = build_float_phi_shader(&nir_options);
   nir_shader *shaders[] = {shader};
   const vt_nir_to_llvm_options llvm_options = {
      .module_name = "vt_float_phi",
      .target_triple = "riscv32-unknown-elf",
      .data_layout = "e-m:e-p:32:32-i64:64-n32-S128",
      .require_ssa_function_temps = true,
   };

   llvm::Module *module =
      vt_nir_to_llvm_translate(shaders, 1, &llvm_options);
   VT_RT_CHECK(module != nullptr);
   VT_RT_CHECK(!llvm::verifyModule(*module));

   bool found_phi = false;
   bool found_float_bits = false;
   for (llvm::Function &function : *module) {
      for (llvm::BasicBlock &block : function) {
         for (llvm::Instruction &instruction : block) {
            auto *phi = llvm::dyn_cast<llvm::PHINode>(&instruction);
            if (!phi)
               continue;

            found_phi = true;
            VT_RT_CHECK(phi->getType()->isIntegerTy(32));
            for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
               llvm::Value *incoming = phi->getIncomingValue(i);
               VT_RT_CHECK(!llvm::isa<llvm::UndefValue>(incoming));
               if (auto *bitcast = llvm::dyn_cast<llvm::BitCastInst>(incoming))
                  found_float_bits |= bitcast->getSrcTy()->isFloatTy();
            }
         }
      }
   }

   VT_RT_CHECK(found_phi);
   VT_RT_CHECK(found_float_bits);

   delete module;
   ralloc_free(shader);

   shader = build_loop_exit_phi_shader(&nir_options);
   shaders[0] = shader;
   module = vt_nir_to_llvm_translate(shaders, 1, &llvm_options);
   VT_RT_CHECK(module != nullptr);
   VT_RT_CHECK(!llvm::verifyModule(*module));

   bool found_loop_exit_phi = false;
   for (llvm::Function &function : *module) {
      for (llvm::BasicBlock &block : function) {
         for (llvm::Instruction &instruction : block) {
            auto *phi = llvm::dyn_cast<llvm::PHINode>(&instruction);
            if (!phi || !phi->getType()->isStructTy())
               continue;
            found_loop_exit_phi = true;
            VT_RT_CHECK(phi->getNumIncomingValues() == 2);
            for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i)
               VT_RT_CHECK(phi->getIncomingBlock(i) != nullptr);
         }
      }
   }
   VT_RT_CHECK(found_loop_exit_phi);

   delete module;
   ralloc_free(shader);

   shader = build_storage_image_shader(&nir_options, 0, 13);
   shaders[0] = shader;
   module = vt_nir_to_llvm_translate(shaders, 1, &llvm_options);
   VT_RT_CHECK(module != nullptr);
   VT_RT_CHECK(!llvm::verifyModule(*module));

   bool found_binding_13 = false;
   bool found_image_store = false;
   for (llvm::Function &function : *module) {
      for (llvm::BasicBlock &block : function) {
         for (llvm::Instruction &instruction : block) {
            auto *mul =
               llvm::dyn_cast<llvm::BinaryOperator>(&instruction);
            if (mul) {
               if (mul->getOpcode() == llvm::Instruction::Mul) {
                  for (unsigned operand = 0; operand < 2; operand++) {
                     auto *constant =
                        llvm::dyn_cast<llvm::ConstantInt>(
                           mul->getOperand(operand));
                     found_binding_13 |=
                        constant && constant->getZExtValue() == 13;
                  }
               }
            }

            auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
            llvm::Function *callee = call ? call->getCalledFunction() : nullptr;
            found_image_store |=
               callee &&
               callee->getName() == "llvm.riscv.ventus.image.store.rgba8";
         }
      }
   }
   VT_RT_CHECK(found_binding_13);
   VT_RT_CHECK(found_image_store);

   delete module;
   ralloc_free(shader);

   shader = build_storage_image_shader(&nir_options, 1, 13);
   shaders[0] = shader;
   module = vt_nir_to_llvm_translate(shaders, 1, &llvm_options);
   VT_RT_CHECK(module == nullptr);
   ralloc_free(shader);

   glsl_type_singleton_decref();
   return 0;
}
