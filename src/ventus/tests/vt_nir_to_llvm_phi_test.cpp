/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstdlib>

#include "compiler/glsl_types.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"

#include "ventus/compiler/vt_nir.h"
#include "ventus/compiler/vt_rt_descriptor_abi.h"
#include "ventus/llvm/ventus_nir_to_llvm.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
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
build_float_mod_shader(const nir_shader_compiler_options *options)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_RAYGEN, options,
                                                   "vt_float_mod");
   nir_def *launch_id = nir_load_ray_launch_id(&b);
   nir_def *dividend =
      nir_u2f32(&b, nir_channel(&b, launch_id, 0));
   nir_def *divisor =
      nir_fadd_imm(&b, nir_u2f32(&b, nir_channel(&b, launch_id, 1)), 2.0f);
   nir_def *result = nir_fmod(&b, dividend, divisor);
   nir_ventus_rt_store_slot(&b, result, nir_imm_int(&b, 0));
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

static nir_shader *
build_rt_traverse_v5_shader(const nir_shader_compiler_options *options)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_RAYGEN, options, "vt_rt_traverse_v5");
   const struct vt_nir_rt_trace_params params = {
      .accel_struct = nir_imm_int64(&b, 0x10000),
      .flags = nir_imm_int(&b, 0),
      .cull_mask = nir_imm_int(&b, 0xff),
      .sbt_offset = nir_imm_int(&b, 0),
      .sbt_stride = nir_imm_int(&b, 0),
      .miss_index = nir_imm_int(&b, 0),
      .origin = nir_vec3(&b, nir_imm_float(&b, 0.0f),
                         nir_imm_float(&b, 0.0f),
                         nir_imm_float(&b, 0.0f)),
      .tmin = nir_imm_float(&b, 0.001f),
      .direction = nir_vec3(&b, nir_imm_float(&b, 0.0f),
                            nir_imm_float(&b, 0.0f),
                            nir_imm_float(&b, -1.0f)),
      .tmax = nir_imm_float(&b, 1000.0f),
      .payload = nir_imm_int(&b, 0),
   };
   nir_def *private_address = vt_nir_rt_v5_private_address(
      &b, nir_imm_int(&b, 0), nir_imm_int(&b, 0x180),
      nir_imm_int(&b, 1), nir_imm_int(&b, 31));
   nir_def *pds_address =
      nir_ventus_rt_hot_pds_address(&b, nir_imm_int(&b, 0));
   const struct vt_nir_rt_v5_submit_result result =
      vt_nir_build_traversal_v5_submit(
         &b, &params, pds_address, private_address);
   nir_ventus_rt_store_slot(&b, result.compact_result, nir_imm_int(&b, 0));
   return b.shader;
}

static nir_shader *
build_external_store_shader(const nir_shader_compiler_options *options,
                            unsigned bits)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_RAYGEN, options,
                                                 "vt_external_store");
   const glsl_type *type = bits == 32 ? glsl_uvec4_type() :
                           bits == 64 ? glsl_uint64_t_type() : glsl_uint16_t_type();
   nir_deref_instr *target = nir_build_deref_cast(
      &b, nir_imm_int64(&b, 0x91000000), nir_var_mem_global, type, 0);
   nir_def *data = bits == 32 ?
      nir_vec4(&b, nir_imm_int(&b, 101), nir_imm_int(&b, 202),
                   nir_imm_int(&b, 303), nir_imm_int(&b, 404)) :
      nir_imm_intN_t(&b, 0x1122334455667788ull, bits);
   nir_store_deref_with_access(&b, target, data, bits == 32 ? 0x5 : 0x1,
                               ACCESS_VOLATILE);
   return b.shader;
}

static nir_shader *
build_quad_shader(const nir_shader_compiler_options *options,
                  nir_intrinsic_op operation, unsigned target = 0,
                  bool dynamic = false, mesa_shader_stage stage = MESA_SHADER_RAYGEN,
                  unsigned bits = 32)
{
   nir_builder b = nir_builder_init_simple_shader(stage, options, "vt_explicit_quad");
   nir_def *x = nir_channel(&b, nir_load_ray_launch_id(&b), 0);
   nir_def *source = nir_vec2(&b, x, nir_iadd_imm(&b, x, 7));
   if (bits == 16)
      source = nir_u2u16(&b, source);
   nir_intrinsic_instr *quad = nir_intrinsic_instr_create(b.shader, operation);
   quad->num_components = 2;
   quad->src[0] = nir_src_for_ssa(source);
   if (operation == nir_intrinsic_quad_broadcast)
      quad->src[1] = nir_src_for_ssa(dynamic ? x : nir_imm_int(&b, target));
   nir_def_init(&quad->instr, &quad->def, 2, bits);
   nir_builder_instr_insert(&b, &quad->instr);
   nir_ventus_rt_store_slot(&b, &quad->def, nir_imm_int(&b, 0));
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
      .data_layout = "e-m:e-p:32:32-i64:64-n32-S128-A5-G1",
      .require_ssa_function_temps = true,
      .private_base_bytes = 320,
      .emit_rt_resource_v4 = true,
      .rt_feature_flags = 3,
      .rt_pds_layout_id = 5,
      .rt_private_layout_id = 5,
      .rt_hot_pds_offset_per_thread = 128,
      .rt_hot_pds_stride_bytes = 384,
      .rt_continuation_offset_per_thread = 512,
      .rt_continuation_bytes_per_thread = 256,
      .rt_payload_size_bytes = 96,
      .rt_callable_data_size_bytes = 0,
   };

   llvm::Module *module =
      vt_nir_to_llvm_translate(shaders, 1, &llvm_options);
   VT_RT_CHECK(module != nullptr);
   VT_RT_CHECK(!llvm::verifyModule(*module));
   VT_RT_CHECK(module->getDataLayout().getAllocaAddrSpace() == 5);
   llvm::Function *entry = module->getFunction("main");
   VT_RT_CHECK(entry != nullptr);
   llvm::MDNode *rt_resource = entry->getMetadata("ventus.rt.resource");
   VT_RT_CHECK(rt_resource != nullptr);
   VT_RT_CHECK(rt_resource->getNumOperands() == 10);
   const auto *rt_stride = llvm::mdconst::dyn_extract<llvm::ConstantInt>(
      rt_resource->getOperand(4));
   VT_RT_CHECK(rt_stride != nullptr);
   VT_RT_CHECK(rt_stride->getZExtValue() == 384);
   llvm::MDNode *pds_static = entry->getMetadata("ventus.pds.static");
   VT_RT_CHECK(pds_static != nullptr);
   VT_RT_CHECK(pds_static->getNumOperands() == 1);
   const auto *pds_value = llvm::dyn_cast<llvm::ValueAsMetadata>(
      pds_static->getOperand(0).get());
   const auto *pds_allocation = pds_value ?
      llvm::dyn_cast<llvm::GlobalVariable>(pds_value->getValue()) : nullptr;
   VT_RT_CHECK(pds_allocation != nullptr);
   VT_RT_CHECK(pds_allocation->getAddressSpace() == 5);
   const auto *pds_type =
      llvm::dyn_cast<llvm::ArrayType>(pds_allocation->getValueType());
   VT_RT_CHECK(pds_type != nullptr);
   VT_RT_CHECK(pds_type->getNumElements() == 768);

   bool found_phi = false;
   bool found_float_bits = false;
   bool found_reserved_private_prefix = false;
   for (llvm::Function &function : *module) {
      for (llvm::BasicBlock &block : function) {
         for (llvm::Instruction &instruction : block) {
            auto *alloca = llvm::dyn_cast<llvm::AllocaInst>(&instruction);
            if (alloca && alloca->getName() == "ventus.private.frame") {
               VT_RT_CHECK(alloca->getAddressSpace() == 5);
               auto *frame = llvm::dyn_cast<llvm::ArrayType>(
                  alloca->getAllocatedType());
               found_reserved_private_prefix =
                  frame && frame->getNumElements() >= 80;
            }

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
   VT_RT_CHECK(found_reserved_private_prefix);

   delete module;
   ralloc_free(shader);

   shader = build_rt_traverse_v5_shader(&nir_options);
   shaders[0] = shader;
   module = vt_nir_to_llvm_translate(shaders, 1, &llvm_options);
   VT_RT_CHECK(module != nullptr);
   VT_RT_CHECK(!llvm::verifyModule(*module));

   llvm::Function *traverse_v5 =
      module->getFunction("llvm.riscv.ventus.rt.traverse.v5.i32");
   VT_RT_CHECK(traverse_v5 != nullptr);
   VT_RT_CHECK(traverse_v5->arg_size() == 2);
   VT_RT_CHECK(!traverse_v5->use_empty());

   bool found_v5_fence = false;
   bool found_v5_pds_csr = false;
   bool found_v5_lds_csr = false;
   bool found_v5_thread_id = false;
   bool found_v5_pds_static_stride = false;
   unsigned v5_global_pds_store_count = 0;
   unsigned v5_shared_store_count = 0;
   unsigned v5_canonical_shared_store_count = 0;
   unsigned v5_private_store_count = 0;
   for (llvm::Function &function : *module) {
      for (llvm::BasicBlock &block : function) {
         for (llvm::Instruction &instruction : block) {
            found_v5_fence |= llvm::isa<llvm::FenceInst>(&instruction);
            auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
            if (call && call->isInlineAsm()) {
               const auto *assembly = llvm::cast<llvm::InlineAsm>(
                  call->getCalledOperand());
               const llvm::StringRef text = assembly->getAsmString();
               found_v5_pds_csr |= text.contains("CSR_PDS");
               found_v5_lds_csr |= text.contains("CSR_LDS");
               found_v5_thread_id |= text.contains("CSR_TID");
            }
            auto *binary = llvm::dyn_cast<llvm::BinaryOperator>(&instruction);
            if (binary && binary->getOpcode() == llvm::Instruction::Mul) {
               for (const llvm::Use &operand : binary->operands()) {
                  const auto *constant =
                     llvm::dyn_cast<llvm::ConstantInt>(operand.get());
                  found_v5_pds_static_stride |=
                     constant && constant->getZExtValue() == 768;
               }
            }
            auto *store = llvm::dyn_cast<llvm::StoreInst>(&instruction);
            if (!store)
               continue;
            const unsigned address_space =
               store->getPointerAddressSpace();
            v5_global_pds_store_count += address_space == 1;
            v5_shared_store_count += address_space == 3;
            v5_private_store_count += address_space == 5;
            if (address_space == 3) {
               auto *pointer = llvm::dyn_cast<llvm::IntToPtrInst>(
                  store->getPointerOperand());
               auto *address = pointer
                  ? llvm::dyn_cast<llvm::BinaryOperator>(
                       pointer->getOperand(0))
                  : nullptr;
               bool has_lds_base = false;
               if (address && address->getOpcode() == llvm::Instruction::Add) {
                  for (const llvm::Use &operand : address->operands()) {
                     auto *call = llvm::dyn_cast<llvm::CallBase>(operand.get());
                     if (!call || !call->isInlineAsm())
                        continue;
                     const auto *assembly = llvm::cast<llvm::InlineAsm>(
                        call->getCalledOperand());
                     has_lds_base |=
                        llvm::StringRef(assembly->getAsmString())
                           .contains("CSR_LDS");
                  }
               }
               v5_canonical_shared_store_count += has_lds_base;
            }
         }
      }
   }
   VT_RT_CHECK(found_v5_fence);
   VT_RT_CHECK(found_v5_pds_csr);
   VT_RT_CHECK(found_v5_lds_csr);
   VT_RT_CHECK(found_v5_thread_id);
   VT_RT_CHECK(found_v5_pds_static_stride);
   VT_RT_CHECK(v5_global_pds_store_count == 24);
   VT_RT_CHECK(v5_shared_store_count == 17);
   VT_RT_CHECK(v5_canonical_shared_store_count == v5_shared_store_count);
   /* Only the explicit test sink uses legacy private storage.  V5 launch IDs
    * stay in SSA so field-major private slots cannot alias lane-major PDS. */
   VT_RT_CHECK(v5_private_store_count == 1);

   delete module;
   ralloc_free(shader);

   shader = build_float_mod_shader(&nir_options);
   shaders[0] = shader;
   module = vt_nir_to_llvm_translate(shaders, 1, &llvm_options);
   VT_RT_CHECK(module != nullptr);
   VT_RT_CHECK(!llvm::verifyModule(*module));

   bool found_fdiv = false;
   bool found_fptosi = false;
   bool found_sitofp = false;
   bool found_select = false;
   bool found_fsub = false;
   for (llvm::Function &function : *module) {
      for (llvm::BasicBlock &block : function) {
         for (llvm::Instruction &instruction : block) {
            found_fdiv |= instruction.getOpcode() == llvm::Instruction::FDiv;
            found_fptosi |=
               instruction.getOpcode() == llvm::Instruction::FPToSI;
            found_sitofp |=
               instruction.getOpcode() == llvm::Instruction::SIToFP;
            found_select |=
               instruction.getOpcode() == llvm::Instruction::Select;
            found_fsub |= instruction.getOpcode() == llvm::Instruction::FSub;
            const bool fmod_operation =
               instruction.getOpcode() == llvm::Instruction::FDiv ||
               instruction.getOpcode() == llvm::Instruction::FPToSI ||
               instruction.getOpcode() == llvm::Instruction::SIToFP ||
               instruction.getOpcode() == llvm::Instruction::Select ||
               instruction.getOpcode() == llvm::Instruction::FSub;
            if (fmod_operation) {
               for (llvm::Use &operand : instruction.operands())
                  VT_RT_CHECK(!llvm::isa<llvm::UndefValue>(operand.get()));
            }
         }
      }
   }
   VT_RT_CHECK(found_fdiv);
   VT_RT_CHECK(found_fptosi);
   VT_RT_CHECK(found_sitofp);
   VT_RT_CHECK(found_select);
   VT_RT_CHECK(found_fsub);

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
   VT_RT_CHECK(module != nullptr);
   VT_RT_CHECK(!llvm::verifyModule(*module));

   bool found_set_1_binding_13 = false;
   for (llvm::Function &function : *module) {
      for (llvm::BasicBlock &block : function) {
         for (llvm::Instruction &instruction : block) {
            auto *mul = llvm::dyn_cast<llvm::BinaryOperator>(&instruction);
            if (!mul || mul->getOpcode() != llvm::Instruction::Mul)
               continue;
            for (unsigned operand = 0; operand < 2; operand++) {
               auto *constant =
                  llvm::dyn_cast<llvm::ConstantInt>(mul->getOperand(operand));
               found_set_1_binding_13 |=
                  constant && constant->getZExtValue() ==
                     VT_RT_DESCRIPTOR_BINDINGS_PER_SET + 13;
            }
         }
      }
   }
   VT_RT_CHECK(found_set_1_binding_13);

   delete module;
   ralloc_free(shader);

   shader = build_storage_image_shader(
      &nir_options, VT_RT_DESCRIPTOR_SET_COUNT, 13);
   shaders[0] = shader;
   module = vt_nir_to_llvm_translate(shaders, 1, &llvm_options);
   VT_RT_CHECK(module == nullptr);
   ralloc_free(shader);

   for (unsigned bits : {32u, 64u, 16u}) {
      shader = build_external_store_shader(&nir_options, bits);
      shaders[0] = shader;
      module = vt_nir_to_llvm_translate(shaders, 1, &llvm_options);
      if (bits == 16) {
         VT_RT_CHECK(module == nullptr);
      } else {
         VT_RT_CHECK(module != nullptr);
         VT_RT_CHECK(!llvm::verifyModule(*module));
         unsigned stores = 0;
         for (llvm::Function &function : *module)
            for (llvm::BasicBlock &block : function)
               for (llvm::Instruction &instruction : block) {
                  auto *store = llvm::dyn_cast<llvm::StoreInst>(&instruction);
                  if (!store || store->getPointerAddressSpace() != 1)
                     continue;
                  auto *ptr = llvm::dyn_cast<llvm::ConstantExpr>(store->getPointerOperand());
                  VT_RT_CHECK(ptr && ptr->getOpcode() == llvm::Instruction::IntToPtr);
                  auto *addr = llvm::dyn_cast<llvm::ConstantInt>(ptr->getOperand(0));
                  auto *data = llvm::dyn_cast<llvm::ConstantInt>(store->getValueOperand());
                  VT_RT_CHECK(addr && data && store->isVolatile());
                  VT_RT_CHECK(data->getBitWidth() == bits);
                  if (bits == 32) {
                     const auto offset = addr->getZExtValue() - 0x91000000;
                     VT_RT_CHECK((offset == 0 && data->getZExtValue() == 101) ||
                                 (offset == 8 && data->getZExtValue() == 303));
                  } else {
                     VT_RT_CHECK(addr->getZExtValue() == 0x91000000);
                     VT_RT_CHECK(data->getZExtValue() == 0x1122334455667788ull);
                  }
                  ++stores;
               }
         VT_RT_CHECK(stores == (bits == 32 ? 2u : 1u));
         delete module;
      }
      ralloc_free(shader);
   }

   vt_nir_to_llvm_options quad_options = llvm_options;
   for (nir_intrinsic_op op : {nir_intrinsic_quad_swap_horizontal,
                              nir_intrinsic_quad_swap_vertical,
                              nir_intrinsic_quad_swap_diagonal,
                              nir_intrinsic_quad_broadcast}) {
      for (bool enabled : {false, true}) {
         quad_options.texture_explicit_quad = enabled;
         shader = build_quad_shader(&nir_options, op);
         shaders[0] = shader;
         module = vt_nir_to_llvm_translate(shaders, 1, &quad_options);
         VT_RT_CHECK((module != nullptr) == enabled);
         if (module) {
            VT_RT_CHECK(!llvm::verifyModule(*module));
            unsigned shuffles = 0;
            for (llvm::Function &function : *module)
               for (llvm::BasicBlock &block : function)
                  for (llvm::Instruction &instruction : block) {
                     auto *call = llvm::dyn_cast<llvm::CallInst>(&instruction);
                     if (!call || !call->getCalledFunction() ||
                         call->getCalledFunction()->getName() !=
                            "llvm.riscv.ventus.shuffle.bfly.i32")
                        continue;
                     auto *mask = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
                     VT_RT_CHECK(mask && mask->getZExtValue() >= 1 && mask->getZExtValue() <= 3);
                     VT_RT_CHECK(call->isConvergent());
                     ++shuffles;
                  }
            VT_RT_CHECK(shuffles == (op == nir_intrinsic_quad_broadcast ? 4u : 2u));
            delete module;
         }
         ralloc_free(shader);
      }
   }
   quad_options.texture_explicit_quad = true;
   for (unsigned target : {1u,2u,3u,4u}) {
      shader = build_quad_shader(&nir_options, nir_intrinsic_quad_broadcast, target);
      shaders[0] = shader;
      module = vt_nir_to_llvm_translate(shaders, 1, &quad_options);
      VT_RT_CHECK((module != nullptr) == (target < 4));
      delete module;ralloc_free(shader);
   }
   for (unsigned bad : {0u,1u,2u}) {
      shader = build_quad_shader(&nir_options, nir_intrinsic_quad_broadcast, 0,
          bad == 0, bad == 1 ? MESA_SHADER_COMPUTE : MESA_SHADER_RAYGEN,
          bad == 2 ? 16 : 32);
      shaders[0] = shader;
      module = vt_nir_to_llvm_translate(shaders, 1, &quad_options);
      VT_RT_CHECK(module == nullptr);
      ralloc_free(shader);
   }
   for (nir_intrinsic_op op : {nir_intrinsic_ddx,nir_intrinsic_ddy,
          nir_intrinsic_ddx_fine,nir_intrinsic_ddy_fine,
          nir_intrinsic_ddx_coarse,nir_intrinsic_ddy_coarse}) {
      shader = build_quad_shader(&nir_options, op);
      shaders[0] = shader;
      module = vt_nir_to_llvm_translate(shaders, 1, &quad_options);
      VT_RT_CHECK(module == nullptr);
      ralloc_free(shader);
   }

   glsl_type_singleton_decref();
   return 0;
}
