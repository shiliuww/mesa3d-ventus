/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "ventus_nir_to_llvm.h"

#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "nir/nir.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"
#include "vulkan/vulkan_core.h"

#include "ventus/compiler/vt_nir.h"

using namespace llvm;

namespace {

enum {
   VTVK_KNL_ARG_BASE = 4,
   VENTUS_KERNEL_CALLING_CONV = 104,
};

static std::string
stage_name(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_RAYGEN:
      return "raygen";
   case MESA_SHADER_MISS:
      return "miss";
   case MESA_SHADER_CLOSEST_HIT:
      return "closest_hit";
   case MESA_SHADER_ANY_HIT:
      return "any_hit";
   case MESA_SHADER_INTERSECTION:
      return "intersection";
   default:
      return "shader_" + std::to_string(stage);
   }
}

struct unsupported_source_state {
   const std::unordered_set<const nir_def *> *defs;
   bool found;
};

static bool
source_is_unsupported(nir_src *src, void *data)
{
   auto *state = static_cast<unsupported_source_state *>(data);
   if (src->ssa && state->defs->count(src->ssa)) {
      state->found = true;
      return false;
   }
   return true;
}

static const nir_variable *
residual_function_temp(const nir_shader *shader)
{
   nir_foreach_function_impl(impl, const_cast<nir_shader *>(shader)) {
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

   return nullptr;
}

class VentusNirToLlvm {
public:
   VentusNirToLlvm(LLVMContext &context, Module &module)
      : context(context), module(module), builder(context),
        i1(Type::getInt1Ty(context)), i32(Type::getInt32Ty(context)),
        i64(Type::getInt64Ty(context)), f32(Type::getFloatTy(context)),
        noopRtIntrinsics(envEnabled("VENTUS_VK_NOOP_RTU")),
        earlyExit(envEnabled("VENTUS_VK_EARLY_EXIT")),
        debugLowering(envEnabled("VENTUS_VK_DEBUG_NIR_LLVM")) {}

   bool translate(nir_shader *shader, bool requireSsaFunctionTemps)
   {
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      if (!impl)
         return true;

      if (requireSsaFunctionTemps) {
         const nir_variable *residual = residual_function_temp(shader);
         if (residual) {
            fprintf(stderr,
                    "ventus-nir-to-llvm: RT shader has non-SSA function "
                    "temporary %s\n",
                    residual->name ? residual->name : "<unnamed>");
            return false;
         }
      }

      resetFunctionState();

      Function *fn = createEntry(shader);
      BasicBlock *entry = BasicBlock::Create(context, "entry", fn);
      builder.SetInsertPoint(entry);

      if (earlyExit && fn->getName() == "main") {
         emitKernelReturn();
         verifyFunction(*fn, &errs());
         return true;
      }

      initializeRaygenLaunchId(shader);

      mapBlocks(fn, impl);
      allocateVariables(shader, impl);

      BasicBlock *first = firstBlock(&impl->body);
      if (first) {
         if (needsRaygenLaunchGuard(shader)) {
            returnTarget = BasicBlock::Create(context, "raygen.exit", fn);
            builder.CreateCondBr(raygenLaunchIdInBounds(), first, returnTarget);
         } else {
            builder.CreateBr(first);
         }
      }

      if (!visitCFList(&impl->body))
         return false;
      finishPhis();

      if (returnTarget) {
         if (builder.GetInsertBlock() &&
             !builder.GetInsertBlock()->getTerminator())
            builder.CreateBr(returnTarget);
         builder.SetInsertPoint(returnTarget);
         if (!returnTarget->getTerminator())
            emitFunctionExit();
      } else if (builder.GetInsertBlock() &&
                 !builder.GetInsertBlock()->getTerminator()) {
         emitReturn();
      }

      verifyFunction(*fn, &errs());
      return true;
   }

private:
   LLVMContext &context;
   Module &module;
   IRBuilder<> builder;
   Type *i1;
   Type *i32;
   Type *i64;
   Type *f32;
   bool noopRtIntrinsics;
   bool earlyExit;
   bool debugLowering;
   bool translationFailed = false;

   std::unordered_map<const nir_def *, Value *> defs;
   std::unordered_set<const nir_def *> unsupportedDefs;
   std::unordered_map<const nir_variable *, Value *> variables;
   std::unordered_map<const nir_def *, const nir_variable *> derefVars;
   std::unordered_map<const nir_block *, BasicBlock *> blocks;
   std::unordered_map<const nir_block *, BasicBlock *> blockEnds;
   std::vector<std::pair<nir_phi_instr *, PHINode *>> phis;
   std::vector<BasicBlock *> breakTargets;
   std::vector<BasicBlock *> continueTargets;
   BasicBlock *returnTarget = nullptr;
   Value *raygenLaunchId[3] = {nullptr, nullptr, nullptr};

   static bool envEnabled(const char *name)
   {
      const char *value = getenv(name);
      return value && value[0] && strcmp(value, "0") != 0;
   }

   void resetFunctionState()
   {
      defs.clear();
      unsupportedDefs.clear();
      variables.clear();
      derefVars.clear();
      blocks.clear();
      blockEnds.clear();
      phis.clear();
      breakTargets.clear();
      continueTargets.clear();
      returnTarget = nullptr;
      raygenLaunchId[0] = nullptr;
      raygenLaunchId[1] = nullptr;
      raygenLaunchId[2] = nullptr;
      translationFailed = false;
   }

   bool instructionUsesUnsupportedDef(nir_instr *instr) const
   {
      unsupported_source_state state = {
         .defs = &unsupportedDefs,
         .found = false,
      };
      nir_foreach_src(instr, source_is_unsupported, &state);
      return state.found;
   }

   void failUnsupportedControlFlow(const nir_def *def)
   {
      if (translationFailed)
         return;

      const char *operation = "unknown";
      const nir_instr *instr = nir_def_instr(def);
      if (instr && instr->type == nir_instr_type_alu)
         operation = nir_op_infos[nir_instr_as_alu(instr)->op].name;
      else if (instr && instr->type == nir_instr_type_intrinsic)
         operation = nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].name;

      fprintf(stderr,
              "ventus-nir-to-llvm: unsupported value from %s reached control flow\n",
              operation);
      translationFailed = true;
   }

   Type *scalarType(unsigned bitSize, bool preferFloat = false)
   {
      if (preferFloat && bitSize == 32)
         return f32;
      if (bitSize == 1)
         return i1;
      if (bitSize == 64)
         return i64;
      return i32;
   }

   Type *defType(const nir_def *def, bool preferFloat = false)
   {
      Type *scalar = scalarType(def->bit_size, preferFloat);
      if (def->num_components <= 1)
         return scalar;
      std::vector<Type *> fields(def->num_components, scalar);
      return StructType::get(context, fields);
   }

   Value *undefFor(const nir_def *def, bool preferFloat = false)
   {
      return UndefValue::get(defType(def, preferFloat));
   }

   Value *zeroFor(const nir_def *def, bool preferFloat = false)
   {
      return Constant::getNullValue(defType(def, preferFloat));
   }

   /*
    * The Ventus backend represents divergent values in 32-bit VGPRs.  A
    * plain LLVM i1 PHI, however, has no divergent VGPR register class: LLVM
    * assigns the PHI a scalar register even when all of its incoming values
    * are lane-varying comparisons.  Keep the NIR boolean representation at
    * CFG merge points as canonical 0/1 i32 values.  asBool() turns it back
    * into an LLVM branch condition at the use site.
    */
   Type *phiType(const nir_def *def)
   {
      if (def->bit_size != 1)
         return defType(def);

      if (def->num_components <= 1)
         return i32;

      std::vector<Type *> fields(def->num_components, i32);
      return StructType::get(context, fields);
   }

   Value *coercePhiIncoming(const nir_def *def, Value *incoming)
   {
      if (def->bit_size != 1)
         return incoming;

      if (def->num_components <= 1)
         return builder.CreateZExt(asBool(incoming), i32);

      return buildVector(def->num_components, i32, [&](unsigned i) {
         return builder.CreateZExt(asBool(component(incoming, i)), i32);
      });
   }

   Function *declare(StringRef name, Type *ret, ArrayRef<Type *> args)
   {
      if (Function *fn = module.getFunction(name))
         return fn;
      return Function::Create(FunctionType::get(ret, args, false),
                              GlobalValue::ExternalLinkage, name, module);
   }

   Value *sharedPtr(Value *addr)
   {
      return builder.CreateIntToPtr(asInt32(addr), PointerType::get(context, 3));
   }

   Value *globalPtr(Value *addr)
   {
      return builder.CreateIntToPtr(asInt32(addr), PointerType::get(context, 1));
   }

   Value *privatePtr(Value *addr)
   {
      return builder.CreateIntToPtr(asInt32(addr), PointerType::get(context, 5));
   }

   Value *rtScratchBase()
   {
      /* Current backend lowers addrspace(5) constants relative to the private
       * frame base. RT scratch offsets are PDS-logical absolute offsets, so use
       * a runtime zero until RT slot load/store gets a first-class lowering.
       */
      Value *a = dispatchArgI32Volatile(VT_NIR_RT_DISPATCH_ARG_WIDTH);
      Value *b = dispatchArgI32Volatile(VT_NIR_RT_DISPATCH_ARG_WIDTH);
      return builder.CreateSub(a, b);
   }

   Value *rtScratchPtr(Value *offset)
   {
      return privatePtr(builder.CreateAdd(rtScratchBase(), asInt32(offset)));
   }

   Value *rtSlotOffset(enum vt_rt_slot_word word)
   {
      return ConstantInt::get(i32,
                              VT_RT_ABI_PDS_BASE_BYTES +
                                 word * VT_RT_ABI_WORD_BYTES);
   }

   Value *loadI32PrivateSlot(enum vt_rt_slot_word word)
   {
      LoadInst *load = builder.CreateLoad(i32, rtScratchPtr(rtSlotOffset(word)));
      load->setAlignment(Align(4));
      return load;
   }

   void storeI32PrivateSlot(enum vt_rt_slot_word word, Value *value)
   {
      StoreInst *store =
         builder.CreateStore(asInt32(value), rtScratchPtr(rtSlotOffset(word)));
      store->setAlignment(Align(4));
   }

   Value *loadI32Global(Value *base, uint32_t offset)
   {
      Value *addr = asInt32(base);
      if (offset)
         addr = builder.CreateAdd(addr, ConstantInt::get(i32, offset));
      LoadInst *load = builder.CreateLoad(i32, globalPtr(addr));
      load->setAlignment(Align(4));
      return load;
   }

   Value *loadI32GlobalVolatile(Value *base, uint32_t offset)
   {
      Value *addr = asInt32(base);
      if (offset)
         addr = builder.CreateAdd(addr, ConstantInt::get(i32, offset));
      LoadInst *load = builder.CreateLoad(i32, globalPtr(addr));
      load->setAlignment(Align(4));
      load->setVolatile(true);
      return load;
   }

   Value *dispatchArgsBase()
   {
      Value *metadataBase =
         callI32Intrinsic("llvm.riscv.ventus.kernel.metadata");
      return loadI32Global(metadataBase, VTVK_KNL_ARG_BASE);
   }

   Value *dispatchArgI32(uint32_t offset)
   {
      return loadI32Global(dispatchArgsBase(), offset);
   }

   Value *dispatchArgI32Volatile(uint32_t offset)
   {
      return loadI32GlobalVolatile(dispatchArgsBase(), offset);
   }

   Value *primaryDispatchPixelComponent(unsigned i)
   {
      switch (i) {
      case 0:
         return callI32Intrinsic("llvm.riscv.ventus.rt.dispatch.id.x");
      case 1:
         return callI32Intrinsic("llvm.riscv.ventus.rt.dispatch.id.y");
      default:
         return callI32Intrinsic("llvm.riscv.ventus.rt.dispatch.id.z");
      }
   }

   Value *raygenLaunchIdComponent(unsigned i)
   {
      assert(i < 3);
      if (!raygenLaunchId[i])
         raygenLaunchId[i] = primaryDispatchPixelComponent(i);
      return raygenLaunchId[i];
   }

   bool needsRaygenLaunchGuard(nir_shader *shader)
   {
      return !noopRtIntrinsics && shader->info.stage == MESA_SHADER_RAYGEN;
   }

   Value *raygenLaunchIdInBounds()
   {
      Value *x = raygenLaunchIdComponent(0);
      Value *y = raygenLaunchIdComponent(1);
      Value *z = raygenLaunchIdComponent(2);
      Value *width = dispatchArgI32(VT_NIR_RT_DISPATCH_ARG_WIDTH);
      Value *height = dispatchArgI32(VT_NIR_RT_DISPATCH_ARG_HEIGHT);
      Value *depth = dispatchArgI32(VT_NIR_RT_DISPATCH_ARG_DEPTH);

      Value *validXY = builder.CreateAnd(builder.CreateICmpULT(x, width),
                                         builder.CreateICmpULT(y, height));
      return builder.CreateAnd(validXY, builder.CreateICmpULT(z, depth));
   }

   void initializeRaygenLaunchId(nir_shader *shader)
   {
      if (noopRtIntrinsics || shader->info.stage != MESA_SHADER_RAYGEN)
         return;

      for (unsigned i = 0; i < 3; ++i) {
         raygenLaunchId[i] = primaryDispatchPixelComponent(i);
      }

      storeI32PrivateSlot(VT_RT_SLOT_LAUNCH_ID_X, raygenLaunchId[0]);
      storeI32PrivateSlot(VT_RT_SLOT_LAUNCH_ID_Y, raygenLaunchId[1]);
      storeI32PrivateSlot(VT_RT_SLOT_LAUNCH_ID_Z, raygenLaunchId[2]);
   }

   Value *rayLaunchId()
   {
      if (noopRtIntrinsics)
         return ConstantAggregateZero::get(StructType::get(context, {i32, i32, i32}));
      return buildVector(3, i32, [&](unsigned i) {
         switch (i) {
         case 0:
            return loadI32PrivateSlot(VT_RT_SLOT_LAUNCH_ID_X);
         case 1:
            return loadI32PrivateSlot(VT_RT_SLOT_LAUNCH_ID_Y);
         default:
            return loadI32PrivateSlot(VT_RT_SLOT_LAUNCH_ID_Z);
         }
      });
   }

   Value *rayLaunchSize()
   {
      return buildVector(3, i32, [&](unsigned i) {
         switch (i) {
         case 0:
            return dispatchArgI32(VT_NIR_RT_DISPATCH_ARG_WIDTH);
         case 1:
            return dispatchArgI32(VT_NIR_RT_DISPATCH_ARG_HEIGHT);
         default:
            return dispatchArgI32(VT_NIR_RT_DISPATCH_ARG_DEPTH);
         }
      });
   }

   Value *dispatchArg(const nir_def *def, uint32_t offset)
   {
      return loadGlobalForDef(dispatchArgsBase(), ConstantInt::get(i32, offset),
                             def);
   }

   Value *descriptorRecordBase(Value *binding)
   {
      Value *table = dispatchArgI32(VT_NIR_RT_DISPATCH_ARG_DESCRIPTOR_TABLE_ADDR);
      Value *stride =
         dispatchArgI32(VT_NIR_RT_DISPATCH_ARG_DESCRIPTOR_RECORD_STRIDE);
      return builder.CreateAdd(
         table, builder.CreateMul(asInt32(binding), stride));
   }

   Value *descriptorRecordI32(Value *binding, uint32_t offset)
   {
      return loadI32Global(descriptorRecordBase(binding), offset);
   }

   Value *descriptorAddressForBinding(Value *binding, const nir_def *def)
   {
      (void)def;
      return descriptorRecordI32(binding, VT_NIR_RT_DESCRIPTOR_RECORD_DEVICE_ADDR);
   }

   Value *descriptorAddressForBinding(uint32_t binding, const nir_def *def)
   {
      return descriptorAddressForBinding(ConstantInt::get(i32, binding), def);
   }

   Value *descriptorValueForDef(const nir_def *def, Value *addr)
   {
      if (def->bit_size == 64 && addr->getType()->isIntegerTy(64))
         return addr;
      Value *addr32 = asInt32(addr);
      if (def->bit_size == 64)
         return builder.CreateZExt(addr32, i64);
      if (def->num_components > 1)
         return buildVector(def->num_components, i32, [&](unsigned i) {
            return i == 0 ? addr32 : ConstantInt::get(i32, 0);
         });
      return addr32;
   }

   Value *loadI32At(Value *addr)
   {
      LoadInst *load = builder.CreateLoad(i32, globalPtr(addr));
      load->setAlignment(Align(4));
      return load;
   }

   Value *loadGlobalForDef(Value *base, Value *offset, const nir_def *def)
   {
      Value *addr = builder.CreateAdd(asInt32(base), asInt32(offset));
      auto loadScalar = [&](unsigned component) -> Value * {
         Value *componentAddr = addr;
         if (component)
            componentAddr =
               builder.CreateAdd(componentAddr, ConstantInt::get(i32, component * 4));

         if (def->bit_size == 64) {
            Value *lo = loadI32At(componentAddr);
            Value *hi = loadI32At(builder.CreateAdd(componentAddr,
                                                    ConstantInt::get(i32, 4)));
            return builder.CreateOr(builder.CreateZExt(lo, i64),
                                    builder.CreateShl(builder.CreateZExt(hi, i64),
                                                      ConstantInt::get(i64, 32)));
         }

         Value *word = loadI32At(componentAddr);
         if (def->bit_size == 1)
            return builder.CreateICmpNE(word, ConstantInt::get(i32, 0));
         return word;
      };

      if (def->num_components <= 1)
         return loadScalar(0);

      return buildVector(def->num_components, scalarType(def->bit_size),
                         loadScalar);
   }

   bool isKernelMain(void) const
   {
      BasicBlock *bb = builder.GetInsertBlock();
      if (!bb)
         return false;
      Function *fn = bb->getParent();
      return fn && fn->getName() == "main";
   }

   void emitKernelReturn()
   {
      /*
       * crt0 calls main and executes endprg after it returns.  Writing the
       * shared HTIF tohost word here lets the first completed resident warp
       * stop Spike while divergent warps are still replaying their pending
       * SIMT paths.
       */
      builder.CreateRetVoid();
   }

   void emitReturn()
   {
      if (returnTarget) {
         branchTo(returnTarget);
         return;
      }
      emitFunctionExit();
   }

   void emitFunctionExit()
   {
      if (isKernelMain())
         emitKernelReturn();
      else
         builder.CreateRetVoid();
   }

   Function *createEntry(nir_shader *shader)
   {
      std::string name = "vt_" + stage_name(shader->info.stage);
      if (shader->info.stage == MESA_SHADER_RAYGEN)
         name = "main";
      Function *fn =
         Function::Create(FunctionType::get(Type::getVoidTy(context), false),
                          GlobalValue::ExternalLinkage, name, module);
      if (name == "main")
         fn->setCallingConv(static_cast<CallingConv::ID>(VENTUS_KERNEL_CALLING_CONV));
      return fn;
   }

   Value *callI32Intrinsic(StringRef name)
   {
      Function *fn = declare(name, i32, {});
      return builder.CreateCall(fn);
   }

   void allocateVariables(nir_shader *shader, nir_function_impl *impl)
   {
      auto storageWords = [](const glsl_type *type) -> unsigned {
         unsigned bytes = glsl_get_explicit_size(type, false);
         if (!bytes)
            bytes = glsl_get_cl_size(type);
         return std::max(4u, (bytes + 3u) / 4u);
      };

      auto allocVar = [&](nir_variable *var) {
         if (!var || variables.count(var))
            return;
         AllocaInst *alloca =
            builder.CreateAlloca(ArrayType::get(i32, storageWords(var->type)), nullptr,
                                 var->name ? var->name : "nir.var");
         alloca->setAlignment(Align(16));
         variables[var] = alloca;
      };

      nir_foreach_variable_in_shader(var, shader)
         allocVar(var);

      foreach_list_typed(nir_variable, var, node, &impl->locals)
         allocVar(var);
   }

   void mapBlocks(Function *fn, nir_function_impl *impl)
   {
      nir_foreach_block(block, impl) {
         blocks[block] = BasicBlock::Create(context,
                                            "nir.bb." + std::to_string(block->index),
                                            fn);
      }
   }

   BasicBlock *firstBlock(struct exec_list *list)
   {
      if (exec_list_is_empty(list))
         return nullptr;

      nir_cf_node *node = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
      switch (node->type) {
      case nir_cf_node_block:
         return blocks[nir_cf_node_as_block(node)];
      case nir_cf_node_if:
         return firstBlock(&nir_cf_node_as_if(node)->then_list);
      case nir_cf_node_loop:
         return firstBlock(&nir_cf_node_as_loop(node)->body);
      default:
         return nullptr;
      }
   }

   void branchTo(BasicBlock *target)
   {
      if (!target)
         return;
      BasicBlock *current = builder.GetInsertBlock();
      if (!current || current->getTerminator() || current == target)
         return;
      builder.CreateBr(target);
   }

   Value *value(const nir_src &src)
   {
      if (!src.ssa)
         return UndefValue::get(i32);
      auto it = defs.find(src.ssa);
      if (it != defs.end())
         return it->second;
      return undefFor(src.ssa);
   }

   Value *component(Value *v, unsigned idx)
   {
      if (auto *vecType = dyn_cast<FixedVectorType>(v->getType())) {
         unsigned count = vecType->getNumElements();
         if (idx >= count)
            idx = count - 1;
         return builder.CreateExtractElement(v, uint64_t(idx));
      }
      if (auto *structType = dyn_cast<StructType>(v->getType())) {
         unsigned count = structType->getNumElements();
         if (!count)
            return UndefValue::get(i32);
         if (idx >= count)
            idx = count - 1;
         return builder.CreateExtractValue(v, {idx});
      }
      return v;
   }

   AllocaInst *entryAlloca(Type *type, StringRef name, Align align)
   {
      Function *fn = builder.GetInsertBlock()->getParent();
      IRBuilder<> entryBuilder(&fn->getEntryBlock(),
                               fn->getEntryBlock().getFirstInsertionPt());
      AllocaInst *alloca = entryBuilder.CreateAlloca(type, nullptr, name);
      alloca->setAlignment(align);
      return alloca;
   }

   Value *reinterpretI32AsF32(Value *v)
   {
      if (auto *constant = dyn_cast<Constant>(v))
         return ConstantExpr::getBitCast(constant, f32);
      return builder.CreateBitCast(v, f32);
   }

   Value *reinterpretF32AsI32(Value *v)
   {
      if (auto *constant = dyn_cast<Constant>(v))
         return ConstantExpr::getBitCast(constant, i32);
      return builder.CreateBitCast(v, i32);
   }

   Value *aluSrcScalar(nir_alu_instr *alu, unsigned srcIdx, unsigned componentIdx = 0)
   {
      unsigned swizzle = alu->src[srcIdx].swizzle[componentIdx];
      return component(value(alu->src[srcIdx].src), swizzle);
   }

   Value *aluSrcVector(nir_alu_instr *alu, unsigned srcIdx, unsigned components)
   {
      Value *src = value(alu->src[srcIdx].src);
      if (components <= 1)
         return component(src, alu->src[srcIdx].swizzle[0]);

      Type *scalar = component(src, alu->src[srcIdx].swizzle[0])->getType();
      return buildVector(components, scalar, [&](unsigned i) {
         return component(src, alu->src[srcIdx].swizzle[i]);
      });
   }

   Value *asBool(Value *v)
   {
      if (v->getType()->isIntegerTy(1))
         return v;
      if (v->getType()->isIntegerTy())
         return builder.CreateICmpNE(v, ConstantInt::get(v->getType(), 0));
      if (v->getType()->isFloatingPointTy())
         return builder.CreateFCmpONE(v, ConstantFP::get(v->getType(), 0.0));
      if (isa<FixedVectorType>(v->getType()))
         return asBool(builder.CreateExtractElement(v, uint64_t(0)));
      if (isa<StructType>(v->getType()))
         return asBool(component(v, 0));
      if (v->getType()->isPointerTy())
         return builder.CreateIsNotNull(v);
      return ConstantInt::getFalse(context);
   }

   Value *asFloat(Value *v)
   {
      if (v->getType()->isFloatingPointTy())
         return v;
      if (v->getType()->isIntegerTy(32))
         return reinterpretI32AsF32(v);
      return UndefValue::get(f32);
   }

   Value *asInt32(Value *v)
   {
      if (v->getType()->isIntegerTy(32))
         return v;
      if (isa<FixedVectorType>(v->getType()))
         return asInt32(component(v, 0));
      if (isa<StructType>(v->getType()))
         return asInt32(component(v, 0));
      if (v->getType()->isFloatingPointTy())
         return reinterpretF32AsI32(v);
      if (v->getType()->isIntegerTy())
         return builder.CreateZExtOrTrunc(v, i32);
      if (v->getType()->isPointerTy())
         return builder.CreatePtrToInt(v, i32);
      return UndefValue::get(i32);
   }

   Value *asInt64(Value *v)
   {
      if (v->getType()->isIntegerTy(64))
         return v;
      if (v->getType()->isIntegerTy())
         return builder.CreateZExtOrTrunc(v, i64);
      return builder.CreateZExt(asInt32(v), i64);
   }

   Value *asSignedInt64(Value *v)
   {
      if (v->getType()->isIntegerTy(64))
         return v;
      if (v->getType()->isIntegerTy())
         return builder.CreateSExtOrTrunc(v, i64);
      return builder.CreateSExt(asInt32(v), i64);
   }

   Value *asInteger(Value *v, unsigned bits, bool isSigned = false)
   {
      if (bits == 64)
         return isSigned ? asSignedInt64(v) : asInt64(v);
      return asInt32(v);
   }

   Value *asStorageBits(Value *v)
   {
      if (auto *vecType = dyn_cast<FixedVectorType>(v->getType())) {
         unsigned count = vecType->getNumElements();
         Value *bits = ConstantAggregateZero::get(FixedVectorType::get(i32, count));
         for (unsigned i = 0; i < count; ++i)
            bits = builder.CreateInsertElement(bits, asInt32(component(v, i)),
                                               uint64_t(i));
         return bits;
      }
      if (auto *structType = dyn_cast<StructType>(v->getType())) {
         unsigned count = structType->getNumElements();
         return buildVector(count, i32, [&](unsigned i) {
            return asInt32(component(v, i));
         });
      }
      return asInt32(v);
   }

   Value *buildVector(unsigned components, Type *scalarType,
                      const std::function<Value *(unsigned)> &element)
   {
      if (components <= 1)
         return element(0);

      std::vector<Type *> fields(components, scalarType);
      Value *vec = UndefValue::get(StructType::get(context, fields));
      for (unsigned i = 0; i < components; ++i)
         vec = builder.CreateInsertValue(vec, element(i), {i});
      return vec;
   }

   Value *localPtrOffset(Value *ptr, Value *byteOffset)
   {
      Value *wordOffset =
         builder.CreateUDiv(asInt32(byteOffset), ConstantInt::get(i32, 4));
      return builder.CreateInBoundsGEP(i32, ptr, wordOffset);
   }

   Value *localPtrOffset(Value *ptr, unsigned byteOffset)
   {
      if (!byteOffset)
         return ptr;
      return localPtrOffset(ptr, ConstantInt::get(i32, byteOffset));
   }

   Value *localSlotPtr(Value *ptr, unsigned component)
   {
      return builder.CreateInBoundsGEP(i32, ptr,
                                       ConstantInt::get(i32, component));
   }

   Value *loadLocalValue(Value *ptr, const nir_def *def)
   {
      auto loadScalar = [&](unsigned component) -> Value * {
         LoadInst *load = builder.CreateLoad(i32, localSlotPtr(ptr, component));
         load->setAlignment(Align(4));
         if (def->bit_size == 1)
            return builder.CreateICmpNE(load, ConstantInt::get(i32, 0));
         if (def->bit_size == 64)
            return builder.CreateZExt(load, i64);
         return load;
      };

      if (def->num_components <= 1)
         return loadScalar(0);
      return buildVector(def->num_components, scalarType(def->bit_size),
                         loadScalar);
   }

   void storeLocalValue(Value *ptr, Value *value)
   {
      unsigned components = 1;
      if (auto *structType = dyn_cast<StructType>(value->getType()))
         components = std::min<unsigned>(structType->getNumElements(), 4);
      else if (auto *vecType = dyn_cast<FixedVectorType>(value->getType()))
         components = std::min<unsigned>(vecType->getNumElements(), 4);

      for (unsigned i = 0; i < components; ++i) {
         StoreInst *store =
            builder.CreateStore(asInt32(component(value, i)), localSlotPtr(ptr, i));
         store->setAlignment(Align(4));
      }
   }

   Value *floatResult(const nir_def *def,
                      const std::function<Value *(unsigned)> &element)
   {
      if (def->num_components <= 1)
         return element(0);

      return buildVector(def->num_components, i32, [&](unsigned i) {
         return asInt32(element(i));
      });
   }

   Value *intResult(const nir_def *def,
                    const std::function<Value *(unsigned)> &element)
   {
      Type *scalar = scalarType(def->bit_size);
      return buildVector(def->num_components, scalar, [&](unsigned i) {
         return asInteger(element(i), def->bit_size);
      });
   }

   Value *boolResult(const nir_def *def,
                     const std::function<Value *(unsigned)> &element)
   {
      Type *scalar = def->bit_size == 1 ? i1 : i32;
      return buildVector(def->num_components, scalar, [&](unsigned i) {
         Value *v = element(i);
         if (def->bit_size == 1)
            return asBool(v);
         return builder.CreateZExtOrTrunc(asBool(v), i32);
      });
   }

   Value *reduceBooleanComparison(nir_alu_instr *alu, bool all,
                                  bool floatSources, bool equal)
   {
      const unsigned components = alu->src[0].src.ssa
                                    ? alu->src[0].src.ssa->num_components
                                    : 1;
      Value *reduced = ConstantInt::get(i1, all);
      for (unsigned i = 0; i < components; i++) {
         Value *lhs = floatSources ? asFloat(aluSrcScalar(alu, 0, i))
                                   : asInt32(aluSrcScalar(alu, 0, i));
         Value *rhs = floatSources ? asFloat(aluSrcScalar(alu, 1, i))
                                   : asInt32(aluSrcScalar(alu, 1, i));
         Value *comparison = floatSources
                               ? (equal ? builder.CreateFCmpOEQ(lhs, rhs)
                                        : builder.CreateFCmpONE(lhs, rhs))
                               : (equal ? builder.CreateICmpEQ(lhs, rhs)
                                        : builder.CreateICmpNE(lhs, rhs));
         reduced = all ? builder.CreateAnd(reduced, comparison)
                       : builder.CreateOr(reduced, comparison);
      }
      return boolResult(&alu->def, [&](unsigned) { return reduced; });
   }

   Function *unaryFloatIntrinsic(StringRef name)
   {
      return declare(("llvm." + name + ".f32").str(), f32, {f32});
   }

   Value *sqrtF32(Value *v)
   {
      return builder.CreateCall(unaryFloatIntrinsic("sqrt"), {asFloat(v)});
   }

   Value *byteComponent(Value *v)
   {
      return builder.CreateAnd(asInt32(v), ConstantInt::get(i32, 0xff));
   }

   Value *unorm8Component(Value *v)
   {
      Value *bits = asInt32(v);
      Value *sign = builder.CreateICmpNE(
         builder.CreateAnd(bits, ConstantInt::get(i32, 0x80000000u)),
         ConstantInt::get(i32, 0));
      Value *geOne =
         builder.CreateICmpUGE(bits, ConstantInt::get(i32, 0x3f800000u));
      Value *tooSmall =
         builder.CreateICmpULT(bits, ConstantInt::get(i32, 0x3b800000u));

      Value *exp = builder.CreateAnd(builder.CreateLShr(bits, 23),
                                     ConstantInt::get(i32, 0xff));
      Value *mant = builder.CreateOr(
         builder.CreateAnd(bits, ConstantInt::get(i32, 0x007fffffu)),
         ConstantInt::get(i32, 0x00800000u));
      Value *scaled = builder.CreateMul(mant, ConstantInt::get(i32, 255));

      Value *shift = builder.CreateSub(ConstantInt::get(i32, 150), exp);
      shift = builder.CreateSelect(builder.CreateICmpULT(exp,
                                                         ConstantInt::get(i32, 119)),
                                   ConstantInt::get(i32, 31), shift);
      shift = builder.CreateSelect(builder.CreateICmpUGT(exp,
                                                         ConstantInt::get(i32, 150)),
                                   ConstantInt::get(i32, 0), shift);

      Value *converted = builder.CreateLShr(scaled, shift);
      converted = builder.CreateSelect(geOne, ConstantInt::get(i32, 255),
                                       converted);
      return builder.CreateSelect(builder.CreateOr(sign, tooSmall),
                                  ConstantInt::get(i32, 0), converted);
   }

   Value *packRGBA8(Value *value, unsigned r_channel, unsigned g_channel,
                    unsigned b_channel, unsigned a_channel)
   {
      Value *r = byteComponent(unorm8Component(component(value, r_channel)));
      Value *g = byteComponent(unorm8Component(component(value, g_channel)));
      Value *b = byteComponent(unorm8Component(component(value, b_channel)));
      Value *a =
         byteComponent((a_channel == 3 &&
                       !isa<FixedVectorType>(value->getType())) ?
                            ConstantInt::get(i32, 255)
                            : unorm8Component(component(value, a_channel)));

      return builder.CreateOr(
         builder.CreateOr(r, builder.CreateShl(g, ConstantInt::get(i32, 8))),
         builder.CreateOr(builder.CreateShl(b, ConstantInt::get(i32, 16)),
                          builder.CreateShl(a, ConstantInt::get(i32, 24))));
   }

   void storeStorageImage(Value *coord, Value *rgba)
   {
      Value *imageBinding = ConstantInt::get(i32, 1);
      Value *imageBase =
         descriptorRecordI32(imageBinding, VT_NIR_RT_DESCRIPTOR_RECORD_DEVICE_ADDR);
      Value *stride =
         descriptorRecordI32(imageBinding, VT_NIR_RT_DESCRIPTOR_RECORD_STRIDE);
      Value *format =
         descriptorRecordI32(imageBinding, VT_NIR_RT_DESCRIPTOR_RECORD_FORMAT);
      Value *x = asInt32(component(coord, 0));
      Value *y = (isa<FixedVectorType>(coord->getType()) ||
                  isa<StructType>(coord->getType())) ?
                    asInt32(component(coord, 1)) : ConstantInt::get(i32, 0);
      Function *fn = declare("llvm.riscv.ventus.image.store.rgba8",
                             builder.getVoidTy(),
                             {i32, i32, i32, i32, i32});

      Value *packedRgba = packRGBA8(rgba, 0, 1, 2, 3);
      Value *packedBgra = packRGBA8(rgba, 2, 1, 0, 3);
      Value *isBgra = builder.CreateICmpEQ(
         format, ConstantInt::get(i32, VK_FORMAT_B8G8R8A8_UNORM));

      Value *packed = builder.CreateSelect(isBgra, packedBgra, packedRgba);
      builder.CreateCall(fn, {imageBase, x, y, stride, packed});
   }

   Value *lowerLoadConst(nir_load_const_instr *instr)
   {
      if (instr->def.num_components == 1)
         return ConstantInt::get(scalarType(instr->def.bit_size), instr->value[0].u64);

      std::vector<Constant *> values;
      Type *scalar = scalarType(instr->def.bit_size);
      for (unsigned i = 0; i < instr->def.num_components; ++i)
         values.push_back(ConstantInt::get(scalar, instr->value[i].u64));
      return ConstantStruct::getAnon(values);
   }

   Value *lowerDeref(nir_deref_instr *deref)
   {
      if (deref->deref_type == nir_deref_type_var && deref->var) {
         derefVars[&deref->def] = deref->var;
         auto it = variables.find(deref->var);
         if (it != variables.end())
            return it->second;
      }

      switch (deref->deref_type) {
      case nir_deref_type_cast:
         return asInt32(value(deref->parent));
      case nir_deref_type_struct: {
         Value *baseValue = value(deref->parent);
         nir_deref_instr *parent = nir_deref_instr_parent(deref);
         int offset = parent ? glsl_get_struct_field_offset(parent->type,
                                                            deref->strct.index)
                             : -1;
         if (baseValue->getType()->isPointerTy())
            return localPtrOffset(baseValue, offset > 0 ? (unsigned)offset : 0);

         Value *base = asInt32(baseValue);
         if (offset > 0)
            base = builder.CreateAdd(base, ConstantInt::get(i32, offset));
         return base;
      }
      case nir_deref_type_array:
      case nir_deref_type_ptr_as_array: {
         Value *baseValue = value(deref->parent);
         unsigned stride = nir_deref_instr_array_stride(deref);
         Value *index = asInt32(value(deref->arr.index));
         if (baseValue->getType()->isPointerTy()) {
            if (!stride)
               return baseValue;
            return localPtrOffset(
               baseValue, builder.CreateMul(index, ConstantInt::get(i32, stride)));
         }

         Value *base = asInt32(baseValue);
         if (stride)
            base = builder.CreateAdd(
               base, builder.CreateMul(index, ConstantInt::get(i32, stride)));
         return base;
      }
      case nir_deref_type_array_wildcard:
         return asInt32(value(deref->parent));
      default:
         break;
      }

      return UndefValue::get(PointerType::get(context, 0));
   }

   Value *lowerAlu(nir_alu_instr *alu)
   {
      auto src = [&](unsigned idx, unsigned component = 0) {
         return aluSrcScalar(alu, idx, component);
      };
      auto intBin = [&](const std::function<Value *(Value *, Value *)> &op) {
         unsigned bits = alu->def.bit_size == 64 ? 64 : 32;
         return intResult(&alu->def, [&](unsigned i) {
            return op(asInteger(src(0, i), bits), asInteger(src(1, i), bits));
         });
      };
      auto floatBin = [&](const std::function<Value *(Value *, Value *)> &op) {
         return floatResult(&alu->def, [&](unsigned i) {
            return op(asFloat(src(0, i)), asFloat(src(1, i)));
         });
      };
      auto cmpBin = [&](const std::function<Value *(Value *, Value *)> &op,
                        bool floatSources) {
         return boolResult(&alu->def, [&](unsigned i) {
            Value *lhs = floatSources ? asFloat(src(0, i)) : asInt32(src(0, i));
            Value *rhs = floatSources ? asFloat(src(1, i)) : asInt32(src(1, i));
            return op(lhs, rhs);
         });
      };

      switch (alu->op) {
      case nir_op_mov:
         return aluSrcVector(alu, 0, alu->def.num_components);
      case nir_op_iadd:
         return intBin([&](Value *a, Value *b) { return builder.CreateAdd(a, b); });
      case nir_op_isub:
         return intBin([&](Value *a, Value *b) { return builder.CreateSub(a, b); });
      case nir_op_imul:
         return intBin([&](Value *a, Value *b) { return builder.CreateMul(a, b); });
      case nir_op_idiv:
         return intBin([&](Value *a, Value *b) { return builder.CreateSDiv(a, b); });
      case nir_op_udiv:
         return intBin([&](Value *a, Value *b) { return builder.CreateUDiv(a, b); });
      case nir_op_imod:
         return intBin([&](Value *a, Value *b) { return builder.CreateSRem(a, b); });
      case nir_op_umod:
         return intBin([&](Value *a, Value *b) { return builder.CreateURem(a, b); });
      case nir_op_udiv_aligned_4:
         return intResult(&alu->def, [&](unsigned i) {
            unsigned bits = alu->def.bit_size == 64 ? 64 : 32;
            return builder.CreateLShr(asInteger(src(0, i), bits),
                                      ConstantInt::get(bits == 64 ? i64 : i32, 2));
         });
      case nir_op_iand:
         return intBin([&](Value *a, Value *b) { return builder.CreateAnd(a, b); });
      case nir_op_ior:
         return intBin([&](Value *a, Value *b) { return builder.CreateOr(a, b); });
      case nir_op_ixor:
         return intBin([&](Value *a, Value *b) { return builder.CreateXor(a, b); });
      case nir_op_inot:
         if (alu->def.bit_size == 1) {
            return boolResult(&alu->def, [&](unsigned i) {
               return builder.CreateNot(asBool(src(0, i)));
            });
         }
         return intResult(&alu->def, [&](unsigned i) {
            unsigned bits = alu->def.bit_size == 64 ? 64 : 32;
            return builder.CreateNot(asInteger(src(0, i), bits));
         });
      case nir_op_ishl:
         return intBin([&](Value *a, Value *b) { return builder.CreateShl(a, b); });
      case nir_op_ushr:
         return intBin([&](Value *a, Value *b) { return builder.CreateLShr(a, b); });
      case nir_op_ishr:
         return intBin([&](Value *a, Value *b) { return builder.CreateAShr(a, b); });
      case nir_op_fadd:
         return floatBin([&](Value *a, Value *b) { return builder.CreateFAdd(a, b); });
      case nir_op_fsub:
         return floatBin([&](Value *a, Value *b) { return builder.CreateFSub(a, b); });
      case nir_op_fmul:
         return floatBin([&](Value *a, Value *b) { return builder.CreateFMul(a, b); });
      case nir_op_fdiv:
         return floatBin([&](Value *a, Value *b) { return builder.CreateFDiv(a, b); });
      case nir_op_fneg:
         return floatResult(&alu->def, [&](unsigned i) {
            return builder.CreateFNeg(asFloat(src(0, i)));
         });
      case nir_op_fabs:
         return floatResult(&alu->def, [&](unsigned i) {
            return builder.CreateCall(unaryFloatIntrinsic("fabs"),
                                      {asFloat(src(0, i))});
         });
      case nir_op_fsqrt:
         return floatResult(&alu->def, [&](unsigned i) {
            return sqrtF32(src(0, i));
         });
      case nir_op_frsq:
         return floatResult(&alu->def, [&](unsigned i) {
            return builder.CreateFDiv(ConstantFP::get(f32, 1.0),
                                      sqrtF32(src(0, i)));
         });
      case nir_op_frcp:
         return floatResult(&alu->def, [&](unsigned i) {
            return builder.CreateFDiv(ConstantFP::get(f32, 1.0),
                                      asFloat(src(0, i)));
         });
      case nir_op_ffma:
      case nir_op_ffma_weak:
      case nir_op_fmad:
      case nir_op_fmadz:
      case nir_op_ffmaz:
         return floatResult(&alu->def, [&](unsigned i) {
            return builder.CreateFAdd(
               builder.CreateFMul(asFloat(src(0, i)), asFloat(src(1, i))),
               asFloat(src(2, i)));
         });
      case nir_op_flrp:
         return floatResult(&alu->def, [&](unsigned i) {
            Value *t = asFloat(src(2, i));
            return builder.CreateFAdd(
               builder.CreateFMul(asFloat(src(0, i)),
                                  builder.CreateFSub(ConstantFP::get(f32, 1.0), t)),
               builder.CreateFMul(asFloat(src(1, i)), t));
         });
      case nir_op_fmin:
         return floatBin([&](Value *a, Value *b) {
            return builder.CreateSelect(builder.CreateFCmpOLT(a, b), a, b);
         });
      case nir_op_fmax:
         return floatBin([&](Value *a, Value *b) {
            return builder.CreateSelect(builder.CreateFCmpOGT(a, b), a, b);
         });
      case nir_op_imin:
         return intBin([&](Value *a, Value *b) {
            return builder.CreateSelect(builder.CreateICmpSLT(a, b), a, b);
         });
      case nir_op_imax:
         return intBin([&](Value *a, Value *b) {
            return builder.CreateSelect(builder.CreateICmpSGT(a, b), a, b);
         });
      case nir_op_umin:
         return intBin([&](Value *a, Value *b) {
            return builder.CreateSelect(builder.CreateICmpULT(a, b), a, b);
         });
      case nir_op_umax:
         return intBin([&](Value *a, Value *b) {
            return builder.CreateSelect(builder.CreateICmpUGT(a, b), a, b);
         });
      case nir_op_feq:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateFCmpOEQ(a, b);
         }, true);
      case nir_op_fneu:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateFCmpONE(a, b);
         }, true);
      case nir_op_flt:
      case nir_op_slt:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateFCmpOLT(a, b);
         }, true);
      case nir_op_fge:
      case nir_op_sge:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateFCmpOGE(a, b);
         }, true);
      case nir_op_seq:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateFCmpOEQ(a, b);
         }, true);
      case nir_op_sne:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateFCmpONE(a, b);
         }, true);
      case nir_op_ieq:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateICmpEQ(a, b);
         }, false);
      case nir_op_ine:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateICmpNE(a, b);
         }, false);
      case nir_op_ilt:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateICmpSLT(a, b);
         }, false);
      case nir_op_ige:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateICmpSGE(a, b);
         }, false);
      case nir_op_ult:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateICmpULT(a, b);
         }, false);
      case nir_op_uge:
         return cmpBin([&](Value *a, Value *b) {
            return builder.CreateICmpUGE(a, b);
         }, false);
      case nir_op_ball_fequal2:
      case nir_op_ball_fequal3:
      case nir_op_ball_fequal4:
      case nir_op_b32all_fequal2:
      case nir_op_b32all_fequal3:
      case nir_op_b32all_fequal4:
         return reduceBooleanComparison(alu, true, true, true);
      case nir_op_ball_iequal2:
      case nir_op_ball_iequal3:
      case nir_op_ball_iequal4:
      case nir_op_b32all_iequal2:
      case nir_op_b32all_iequal3:
      case nir_op_b32all_iequal4:
         return reduceBooleanComparison(alu, true, false, true);
      case nir_op_bany_fnequal2:
      case nir_op_bany_fnequal3:
      case nir_op_bany_fnequal4:
      case nir_op_b32any_fnequal2:
      case nir_op_b32any_fnequal3:
      case nir_op_b32any_fnequal4:
         return reduceBooleanComparison(alu, false, true, false);
      case nir_op_bany_inequal2:
      case nir_op_bany_inequal3:
      case nir_op_bany_inequal4:
      case nir_op_b32any_inequal2:
      case nir_op_b32any_inequal3:
      case nir_op_b32any_inequal4:
         return reduceBooleanComparison(alu, false, false, false);
      case nir_op_b2i32:
         return intResult(&alu->def, [&](unsigned i) {
            return builder.CreateZExt(asBool(src(0, i)), i32);
         });
      case nir_op_b2f32:
         return floatResult(&alu->def, [&](unsigned i) {
            return builder.CreateSelect(asBool(src(0, i)),
                                        ConstantFP::get(f32, 1.0),
                                        ConstantFP::get(f32, 0.0));
         });
      case nir_op_u2f32:
         return floatResult(&alu->def, [&](unsigned i) {
            return builder.CreateUIToFP(asInt32(src(0, i)), f32);
         });
      case nir_op_f2u32:
         return intResult(&alu->def, [&](unsigned i) {
            return builder.CreateFPToUI(asFloat(src(0, i)), i32);
         });
      case nir_op_i2f32:
         return floatResult(&alu->def, [&](unsigned i) {
            return builder.CreateSIToFP(asInt32(src(0, i)), f32);
         });
      case nir_op_f2i32:
         return intResult(&alu->def, [&](unsigned i) {
            return builder.CreateFPToSI(asFloat(src(0, i)), i32);
         });
      case nir_op_f2f32:
         return floatResult(&alu->def, [&](unsigned i) {
            return asFloat(src(0, i));
         });
      case nir_op_i2i32:
         return intResult(&alu->def, [&](unsigned i) {
            return asInt32(src(0, i));
         });
      case nir_op_u2u32:
         return intResult(&alu->def, [&](unsigned i) {
            return asInt32(src(0, i));
         });
      case nir_op_i2i64:
         return intResult(&alu->def, [&](unsigned i) {
            return asSignedInt64(src(0, i));
         });
      case nir_op_u2u64:
         return intResult(&alu->def, [&](unsigned i) {
            return asInt64(src(0, i));
         });
      case nir_op_unpack_64_2x32_split_x:
         return builder.CreateTrunc(asInt64(src(0)), i32);
      case nir_op_unpack_64_2x32_split_y:
         return builder.CreateTrunc(
            builder.CreateLShr(asInt64(src(0)), ConstantInt::get(i64, 32)),
            i32);
      case nir_op_pack_64_2x32_split:
         return builder.CreateOr(builder.CreateZExt(asInt32(src(0)), i64),
                                 builder.CreateShl(builder.CreateZExt(asInt32(src(1)), i64),
                                                   ConstantInt::get(i64, 32)));
      case nir_op_bcsel:
         if (alu->def.num_components <= 1)
            return builder.CreateSelect(asBool(src(0)), src(1), src(2));
         return buildVector(alu->def.num_components, i32, [&](unsigned i) {
            return asInt32(builder.CreateSelect(asBool(src(0, i)), src(1, i),
                                                src(2, i)));
         });
      case nir_op_fdot2:
      case nir_op_fdot2_replicated:
      case nir_op_fdot3:
      case nir_op_fdot3_replicated:
      case nir_op_fdot4:
      case nir_op_fdot4_replicated: {
         unsigned count = alu->op == nir_op_fdot2 ||
                                 alu->op == nir_op_fdot2_replicated
                             ? 2
                             : alu->op == nir_op_fdot3 ||
                                      alu->op == nir_op_fdot3_replicated
                                  ? 3
                                  : 4;
         Value *sum = ConstantFP::get(f32, 0.0);
         for (unsigned i = 0; i < count; ++i)
            sum = builder.CreateFAdd(
               sum,
               builder.CreateFMul(asFloat(src(0, i)), asFloat(src(1, i))));
         if (alu->op == nir_op_fdot2_replicated ||
             alu->op == nir_op_fdot3_replicated ||
             alu->op == nir_op_fdot4_replicated)
            return floatResult(&alu->def, [&](unsigned) { return sum; });
         return sum;
      }
      case nir_op_fdph:
      case nir_op_fdph_replicated: {
         Value *sum = ConstantFP::get(f32, 0.0);
         for (unsigned i = 0; i < 3; ++i)
            sum = builder.CreateFAdd(
               sum,
               builder.CreateFMul(asFloat(src(0, i)), asFloat(src(1, i))));
         sum = builder.CreateFAdd(sum, asFloat(src(1, 3)));
         if (alu->op == nir_op_fdph_replicated)
            return floatResult(&alu->def, [&](unsigned) { return sum; });
         return sum;
      }
      case nir_op_vec2:
      case nir_op_vec3:
      case nir_op_vec4: {
         unsigned count = alu->op == nir_op_vec2 ? 2 : alu->op == nir_op_vec3 ? 3 : 4;
         return buildVector(count, i32, [&](unsigned i) {
            return asInt32(src(i));
         });
      }
      default:
         if (debugLowering)
            fprintf(stderr, "ventus-nir-to-llvm: unsupported ALU op %s\n",
                    nir_op_infos[alu->op].name);
         unsupportedDefs.insert(&alu->def);
         return undefFor(&alu->def);
      }
   }

   Value *lowerIntrinsic(nir_intrinsic_instr *intrin)
   {
      switch (intrin->intrinsic) {
      case nir_intrinsic_load_ray_launch_id: {
         return rayLaunchId();
      }
      case nir_intrinsic_ventus_rt_dispatch_arg: {
         return loadGlobalForDef(dispatchArgsBase(), asInt32(value(intrin->src[0])),
                                &intrin->def);
      }
      case nir_intrinsic_load_ray_launch_size: {
         return rayLaunchSize();
      }
      case nir_intrinsic_vulkan_resource_index: {
         return ConstantInt::get(i32, nir_intrinsic_binding(intrin));
      }
      case nir_intrinsic_vulkan_resource_reindex: {
         return builder.CreateAdd(asInt32(value(intrin->src[0])),
                                  asInt32(value(intrin->src[1])));
      }
      case nir_intrinsic_load_vulkan_descriptor: {
         Value *bindingValue = asInt32(value(intrin->src[0]));
         Value *addr = descriptorAddressForBinding(bindingValue, &intrin->def);
         return descriptorValueForDef(&intrin->def, addr);
      }
      case nir_intrinsic_load_ubo: {
         Value *base = asInt32(value(intrin->src[0]));
         Value *offset = asInt32(value(intrin->src[1]));
         return loadGlobalForDef(base, offset, &intrin->def);
      }
      case nir_intrinsic_load_global: {
         return loadGlobalForDef(value(intrin->src[0]), ConstantInt::get(i32, 0),
                                &intrin->def);
      }
      case nir_intrinsic_store_global: {
         Value *addr = asInt32(value(intrin->src[1]));
         Value *stored = value(intrin->src[0]);
         const unsigned components = nir_src_num_components(intrin->src[0]);

         for (unsigned i = 0; i < components; ++i) {
            Value *componentAddr = addr;
            if (i)
               componentAddr = builder.CreateAdd(componentAddr,
                                                  ConstantInt::get(i32, i * 4));
            StoreInst *store = builder.CreateStore(asInt32(component(stored, i)),
                                                    globalPtr(componentAddr));
            store->setAlignment(Align(4));
         }
         return nullptr;
      }
      case nir_intrinsic_global_atomic: {
         const nir_atomic_op op = nir_intrinsic_atomic_op(intrin);
         AtomicRMWInst::BinOp binop;

         switch (op) {
         case nir_atomic_op_iadd:
            binop = AtomicRMWInst::Add;
            break;
         case nir_atomic_op_iand:
            binop = AtomicRMWInst::And;
            break;
         case nir_atomic_op_ior:
            binop = AtomicRMWInst::Or;
            break;
         case nir_atomic_op_ixor:
            binop = AtomicRMWInst::Xor;
            break;
         default:
            unsupportedDefs.insert(&intrin->def);
            return undefFor(&intrin->def);
         }

         AtomicRMWInst *atomic = builder.CreateAtomicRMW(
            binop, globalPtr(asInt32(value(intrin->src[0]))),
            asInt32(value(intrin->src[1])), MaybeAlign(4),
            AtomicOrdering::SequentiallyConsistent);
         return atomic;
      }
      case nir_intrinsic_image_deref_store:
      case nir_intrinsic_image_store: {
         storeStorageImage(value(intrin->src[1]), value(intrin->src[3]));
         return nullptr;
      }
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_image_size: {
         return buildVector(3, i32, [&](unsigned i) {
            switch (i) {
            case 0:
               return dispatchArgI32(VT_NIR_RT_DISPATCH_ARG_WIDTH);
            case 1:
               return dispatchArgI32(VT_NIR_RT_DISPATCH_ARG_HEIGHT);
            default:
               return dispatchArgI32(VT_NIR_RT_DISPATCH_ARG_DEPTH);
            }
         });
      }
      case nir_intrinsic_load_deref: {
         if (intrin->src[0].ssa) {
            auto varIt = derefVars.find(intrin->src[0].ssa);
            if (varIt != derefVars.end()) {
               const nir_variable *var = varIt->second;
               if (var->data.mode == nir_var_system_value) {
                  switch (var->data.location) {
                  case SYSTEM_VALUE_RAY_LAUNCH_ID:
                     return rayLaunchId();
                  case SYSTEM_VALUE_RAY_LAUNCH_SIZE:
                     return rayLaunchSize();
                  default:
                     break;
                  }
               }
            }
         }
         Value *ptr = value(intrin->src[0]);
         if (!ptr->getType()->isPointerTy())
            return loadGlobalForDef(ptr, ConstantInt::get(i32, 0), &intrin->def);
         return loadLocalValue(ptr, &intrin->def);
      }
      case nir_intrinsic_store_deref: {
         Value *ptr = value(intrin->src[0]);
         if (ptr->getType()->isPointerTy())
            storeLocalValue(ptr, value(intrin->src[1]));
         return nullptr;
      }
      case nir_intrinsic_load_shared: {
         return builder.CreateLoad(defType(&intrin->def),
                                   sharedPtr(value(intrin->src[0])));
      }
      case nir_intrinsic_store_shared: {
         builder.CreateStore(value(intrin->src[0]),
                             sharedPtr(value(intrin->src[1])));
         return nullptr;
      }
      case nir_intrinsic_ventus_rt_load_slot: {
         return builder.CreateLoad(defType(&intrin->def),
                                   rtScratchPtr(value(intrin->src[0])));
      }
      case nir_intrinsic_ventus_rt_store_slot: {
         builder.CreateStore(asInt32(value(intrin->src[0])),
                             rtScratchPtr(value(intrin->src[1])));
         return nullptr;
      }
      case nir_intrinsic_ventus_rt_traverse: {
         if (noopRtIntrinsics)
            return ConstantInt::get(i32, 0);
         Function *fn = declare("llvm.riscv.ventus.rt.traverse.i32", i32, {i32});
         return builder.CreateCall(fn, {asInt32(value(intrin->src[0]))});
      }
      case nir_intrinsic_ventus_rt_release: {
         if (noopRtIntrinsics)
            return nullptr;
         Function *fn = declare("llvm.riscv.ventus.rt.release.i32",
                                Type::getVoidTy(context), {i32});
         builder.CreateCall(fn, {asInt32(value(intrin->src[0]))});
         return nullptr;
      }
      default:
         if (nir_intrinsic_infos[intrin->intrinsic].has_dest)
            unsupportedDefs.insert(&intrin->def);
         return nir_intrinsic_infos[intrin->intrinsic].has_dest ? undefFor(&intrin->def) : nullptr;
      }
   }

   bool visitInstr(nir_instr *instr)
   {
      Value *result = nullptr;
      nir_def *def = nullptr;

      switch (instr->type) {
      case nir_instr_type_load_const:
         result = lowerLoadConst(nir_instr_as_load_const(instr));
         def = &nir_instr_as_load_const(instr)->def;
         break;
      case nir_instr_type_deref:
         result = lowerDeref(nir_instr_as_deref(instr));
         def = &nir_instr_as_deref(instr)->def;
         break;
      case nir_instr_type_alu:
         result = lowerAlu(nir_instr_as_alu(instr));
         def = &nir_instr_as_alu(instr)->def;
         break;
      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         result = lowerIntrinsic(intrin);
         if (nir_intrinsic_infos[intrin->intrinsic].has_dest)
            def = &intrin->def;
         break;
      }
      case nir_instr_type_undef:
         result = zeroFor(&nir_instr_as_undef(instr)->def);
         def = &nir_instr_as_undef(instr)->def;
         break;
      case nir_instr_type_phi:
         break;
      case nir_instr_type_jump:
         return visitJump(nir_instr_as_jump(instr));
      default:
         if (nir_def *instrDef = nir_instr_def(instr)) {
            result = undefFor(instrDef);
            def = instrDef;
         }
         break;
      }

      if (result && def)
         defs[def] = result;
      if (def && instructionUsesUnsupportedDef(instr))
         unsupportedDefs.insert(def);
      return true;
   }

   bool visitJump(nir_jump_instr *jump)
   {
      switch (jump->type) {
      case nir_jump_return:
      case nir_jump_halt:
         emitReturn();
         return true;
      case nir_jump_break:
         if (!breakTargets.empty())
            builder.CreateBr(breakTargets.back());
         return true;
      case nir_jump_continue:
         if (!continueTargets.empty())
            builder.CreateBr(continueTargets.back());
         return true;
      default:
         return false;
      }
   }

   bool visitBlock(nir_block *block)
   {
      BasicBlock *bb = blocks[block];
      branchTo(bb);
      builder.SetInsertPoint(bb);

      nir_foreach_phi(phi, block)
         visitPhi(phi);

      nir_foreach_instr(instr, block) {
         visitInstr(instr);
         if (builder.GetInsertBlock()->getTerminator())
            break;
      }

      blockEnds[block] = builder.GetInsertBlock();
      return true;
   }

   void visitPhi(nir_phi_instr *phi)
   {
      PHINode *llvmPhi = builder.CreatePHI(phiType(&phi->def), 0, "nir.phi");
      defs[&phi->def] = llvmPhi;
      if (instructionUsesUnsupportedDef(&phi->instr))
         unsupportedDefs.insert(&phi->def);
      phis.push_back({phi, llvmPhi});
   }

   bool visitIf(nir_if *ifStmt)
   {
      if (ifStmt->condition.ssa && unsupportedDefs.count(ifStmt->condition.ssa)) {
         failUnsupportedControlFlow(ifStmt->condition.ssa);
         return false;
      }

      Function *fn = builder.GetInsertBlock()->getParent();
      BasicBlock *thenBB = firstBlock(&ifStmt->then_list);
      BasicBlock *elseBB = firstBlock(&ifStmt->else_list);
      BasicBlock *mergeBB = nullptr;
      nir_cf_node *next = nir_cf_node_next(&ifStmt->cf_node);
      if (next && next->type == nir_cf_node_block)
         mergeBB = blocks[nir_cf_node_as_block(next)];
      if (!mergeBB)
         mergeBB = BasicBlock::Create(context, "nir.if.end", fn);

      if (!thenBB)
         thenBB = mergeBB;
      if (!elseBB)
         elseBB = mergeBB;

      builder.CreateCondBr(asBool(value(ifStmt->condition)), thenBB, elseBB);

      if (!exec_list_is_empty(&ifStmt->then_list)) {
         builder.SetInsertPoint(thenBB);
         if (!visitCFList(&ifStmt->then_list))
            return false;
         branchTo(mergeBB);
      }

      if (!exec_list_is_empty(&ifStmt->else_list)) {
         builder.SetInsertPoint(elseBB);
         if (!visitCFList(&ifStmt->else_list))
            return false;
         branchTo(mergeBB);
      }

      builder.SetInsertPoint(mergeBB);
      return true;
   }

   bool visitLoop(nir_loop *loop)
   {
      BasicBlock *header = firstBlock(&loop->body);
      if (!header)
         return true;

      Function *fn = builder.GetInsertBlock()->getParent();
      BasicBlock *after = BasicBlock::Create(context, "nir.loop.end", fn);
      branchTo(header);

      breakTargets.push_back(after);
      continueTargets.push_back(header);
      builder.SetInsertPoint(header);
      if (!visitCFList(&loop->body)) {
         continueTargets.pop_back();
         breakTargets.pop_back();
         return false;
      }
      branchTo(header);
      continueTargets.pop_back();
      breakTargets.pop_back();

      builder.SetInsertPoint(after);
      return true;
   }

   bool visitCFNode(nir_cf_node *node)
   {
      switch (node->type) {
      case nir_cf_node_block:
         return visitBlock(nir_cf_node_as_block(node));
      case nir_cf_node_if:
         return visitIf(nir_cf_node_as_if(node));
      case nir_cf_node_loop:
         return visitLoop(nir_cf_node_as_loop(node));
      default:
         return false;
      }
   }

   bool visitCFList(struct exec_list *list)
   {
      foreach_list_typed(nir_cf_node, node, node, list) {
         if (!visitCFNode(node))
            return false;
      }
      return true;
   }

   BasicBlock *endBlockFor(nir_block *block)
   {
      auto endIt = blockEnds.find(block);
      if (endIt != blockEnds.end())
         return endIt->second;
      auto it = blocks.find(block);
      return it == blocks.end() ? nullptr : it->second;
   }

   void finishPhis()
   {
      for (auto &entry : phis) {
         nir_phi_instr *nirPhi = entry.first;
         PHINode *llvmPhi = entry.second;

         nir_foreach_phi_src(src, nirPhi) {
            BasicBlock *pred = endBlockFor(src->pred);
            if (!pred)
               continue;
            Value *incoming = value(src->src);

            /* The conversion belongs to the incoming CFG edge, not to the
             * block where finishPhis() happens to leave the IRBuilder. */
            IRBuilderBase::InsertPoint savedInsertPoint = builder.saveIP();
            if (Instruction *terminator = pred->getTerminator())
               builder.SetInsertPoint(terminator);
            else
               builder.SetInsertPoint(pred);
            incoming = coercePhiIncoming(&nirPhi->def, incoming);
            builder.restoreIP(savedInsertPoint);
            if (incoming->getType() != llvmPhi->getType())
               incoming = UndefValue::get(llvmPhi->getType());
            llvmPhi->addIncoming(incoming, pred);
         }
      }
   }
};

} // namespace

extern "C" Module *
vt_nir_to_llvm_translate(nir_shader **shaders, unsigned shader_count,
                         const struct vt_nir_to_llvm_options *options)
{
   static LLVMContext context;

   const char *moduleName = options && options->module_name ? options->module_name : "ventus_nir";
   auto *module = new Module(moduleName, context);
   module->setTargetTriple(options && options->target_triple ? options->target_triple : "riscv32-unknown-elf");
   module->setDataLayout(options && options->data_layout ? options->data_layout : "e-m:e-p:32:32-i64:64-n32-S128");

   VentusNirToLlvm translator(context, *module);
   for (unsigned i = 0; i < shader_count; ++i) {
      if (shaders[i] && !translator.translate(
                            shaders[i],
                            options && options->require_ssa_function_temps)) {
         delete module;
         return nullptr;
      }
   }

   return module;
}
