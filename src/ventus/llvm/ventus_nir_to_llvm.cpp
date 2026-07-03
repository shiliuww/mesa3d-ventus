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
#include <utility>
#include <vector>

#include "nir/nir.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"

using namespace llvm;

namespace {

enum {
   VTVK_KNL_ARG_BASE = 4,
   VTVK_RT_ARGS_TLAS_ADDR = 0,
   VTVK_RT_ARGS_IMAGE_ADDR = 8,
   VTVK_RT_ARGS_CAMERA_ADDR = 16,
   VTVK_RT_ARGS_WIDTH = 48,
   VTVK_RT_ARGS_HEIGHT = 52,
   VTVK_RT_ARGS_DEPTH = 56,
   VTVK_RT_ARGS_IMAGE_STRIDE = 64,
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

class VentusNirToLlvm {
public:
   VentusNirToLlvm(LLVMContext &context, Module &module)
      : context(context), module(module), builder(context),
        i1(Type::getInt1Ty(context)), i32(Type::getInt32Ty(context)),
        i64(Type::getInt64Ty(context)), f32(Type::getFloatTy(context)),
        noopRtIntrinsics(envEnabled("VENTUS_VK_NOOP_RTU")),
        earlyExit(envEnabled("VENTUS_VK_EARLY_EXIT")),
        debugLowering(envEnabled("VENTUS_VK_DEBUG_NIR_LLVM")) {}

   void translate(nir_shader *shader)
   {
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      if (!impl)
         return;

      resetFunctionState();

      Function *fn = createEntry(shader);
      BasicBlock *entry = BasicBlock::Create(context, "entry", fn);
      builder.SetInsertPoint(entry);

      if (earlyExit && fn->getName() == "main") {
         emitEndprg();
         verifyFunction(*fn, &errs());
         return;
      }

      mapBlocks(fn, impl);
      allocateVariables(shader, impl);

      BasicBlock *first = firstBlock(&impl->body);
      if (first)
         builder.CreateBr(first);

      visitCFList(&impl->body);
      finishPhis();

      if (builder.GetInsertBlock() && !builder.GetInsertBlock()->getTerminator())
         emitReturn();

      verifyFunction(*fn, &errs());
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

   std::unordered_map<const nir_def *, Value *> defs;
   std::unordered_map<const nir_variable *, Value *> variables;
   std::unordered_map<const nir_def *, const nir_variable *> derefVars;
   std::unordered_map<const nir_block *, BasicBlock *> blocks;
   std::unordered_map<const nir_block *, BasicBlock *> blockEnds;
   std::vector<std::pair<nir_phi_instr *, PHINode *>> phis;
   std::vector<BasicBlock *> breakTargets;
   std::vector<BasicBlock *> continueTargets;

   static bool envEnabled(const char *name)
   {
      const char *value = getenv(name);
      return value && value[0] && strcmp(value, "0") != 0;
   }

   void resetFunctionState()
   {
      defs.clear();
      variables.clear();
      derefVars.clear();
      blocks.clear();
      blockEnds.clear();
      phis.clear();
      breakTargets.clear();
      continueTargets.clear();
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
      return FixedVectorType::get(scalar, def->num_components);
   }

   Value *undefFor(const nir_def *def, bool preferFloat = false)
   {
      return UndefValue::get(defType(def, preferFloat));
   }

   Value *zeroFor(const nir_def *def, bool preferFloat = false)
   {
      return Constant::getNullValue(defType(def, preferFloat));
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

   Value *loadI32Global(Value *base, uint32_t offset)
   {
      Value *addr = asInt32(base);
      if (offset)
         addr = builder.CreateAdd(addr, ConstantInt::get(i32, offset));
      LoadInst *load = builder.CreateLoad(i32, globalPtr(addr));
      load->setAlignment(Align(4));
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

   Value *rayLaunchId()
   {
      Type *vecType = FixedVectorType::get(i32, 3);
      if (noopRtIntrinsics)
         return ConstantAggregateZero::get(vecType);
      Value *vec = ConstantAggregateZero::get(vecType);
      vec = builder.CreateInsertElement(
         vec, callI32Intrinsic("llvm.riscv.workitem.id.x"), uint64_t(0));
      vec = builder.CreateInsertElement(
         vec, callI32Intrinsic("llvm.riscv.workitem.id.y"), uint64_t(1));
      vec = builder.CreateInsertElement(
         vec, callI32Intrinsic("llvm.riscv.workitem.id.z"), uint64_t(2));
      return vec;
   }

   Value *rayLaunchSize()
   {
      Type *vecType = FixedVectorType::get(i32, 3);
      Value *vec = ConstantAggregateZero::get(vecType);
      vec = builder.CreateInsertElement(
         vec, dispatchArgI32(VTVK_RT_ARGS_WIDTH), uint64_t(0));
      vec = builder.CreateInsertElement(
         vec, dispatchArgI32(VTVK_RT_ARGS_HEIGHT), uint64_t(1));
      vec = builder.CreateInsertElement(
         vec, dispatchArgI32(VTVK_RT_ARGS_DEPTH), uint64_t(2));
      return vec;
   }

   Value *descriptorAddressForBinding(uint32_t binding)
   {
      switch (binding) {
      case 0:
         return dispatchArgI32(VTVK_RT_ARGS_TLAS_ADDR);
      case 1:
         return dispatchArgI32(VTVK_RT_ARGS_IMAGE_ADDR);
      case 2:
         return dispatchArgI32(VTVK_RT_ARGS_CAMERA_ADDR);
      default:
         return ConstantInt::get(i32, 0);
      }
   }

   Value *descriptorValueForDef(const nir_def *def, Value *addr)
   {
      Value *addr32 = asInt32(addr);
      if (def->bit_size == 64)
         return builder.CreateZExt(addr32, i64);
      if (def->num_components > 1) {
         Value *vec =
            Constant::getNullValue(FixedVectorType::get(i32, def->num_components));
         vec = builder.Insert(InsertElementInst::Create(
            vec, addr32, ConstantInt::get(i32, 0)));
         return vec;
      }
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

      Type *vecType = defType(def);
      Value *vec = ConstantAggregateZero::get(vecType);
      for (unsigned i = 0; i < def->num_components; ++i)
         vec = builder.CreateInsertElement(vec, loadScalar(i), uint64_t(i));
      return vec;
   }

   bool isKernelMain(void) const
   {
      BasicBlock *bb = builder.GetInsertBlock();
      if (!bb)
         return false;
      Function *fn = bb->getParent();
      return fn && fn->getName() == "main";
   }

   void emitEndprg()
   {
      GlobalVariable *tohost = module.getGlobalVariable("tohost");
      if (!tohost) {
         tohost = new GlobalVariable(module, i64, false,
                                     GlobalValue::ExternalLinkage, nullptr,
                                     "tohost");
      }

      builder.CreateStore(ConstantInt::get(i64, 1), tohost);

      Function *fn = builder.GetInsertBlock()->getParent();
      BasicBlock *loop = BasicBlock::Create(context, "kernel.exit.spin", fn);
      builder.CreateBr(loop);
      builder.SetInsertPoint(loop);
      builder.CreateBr(loop);
   }

   void emitReturn()
   {
      if (isKernelMain())
         emitEndprg();
      else
         builder.CreateRetVoid();
   }

   Function *createEntry(nir_shader *shader)
   {
      std::string name = "vt_" + stage_name(shader->info.stage);
      if (shader->info.stage == MESA_SHADER_RAYGEN)
         name = "main";
      return Function::Create(FunctionType::get(Type::getVoidTy(context), false),
                              GlobalValue::ExternalLinkage, name, module);
   }

   Value *callI32Intrinsic(StringRef name)
   {
      Function *fn = declare(name, i32, {});
      return builder.CreateCall(fn);
   }

   void allocateVariables(nir_shader *shader, nir_function_impl *impl)
   {
      auto allocVar = [&](nir_variable *var) {
         if (!var || variables.count(var))
            return;
         AllocaInst *alloca =
            builder.CreateAlloca(FixedVectorType::get(i32, 4), nullptr,
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
      Value *vec = ConstantAggregateZero::get(FixedVectorType::get(scalar, components));
      for (unsigned i = 0; i < components; ++i)
         vec = builder.CreateInsertElement(
            vec, component(src, alu->src[srcIdx].swizzle[i]), uint64_t(i));
      return vec;
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
      return asInt32(v);
   }

   Value *buildVector(unsigned components, Type *scalarType,
                      const std::function<Value *(unsigned)> &element)
   {
      if (components <= 1)
         return element(0);

      Value *vec =
         Constant::getNullValue(FixedVectorType::get(scalarType, components));
      for (unsigned i = 0; i < components; ++i)
         vec = builder.Insert(InsertElementInst::Create(
            vec, element(i), ConstantInt::get(i32, i)));
      return vec;
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

   Value *packRGBA8(Value *value)
   {
      Value *r = byteComponent(component(value, 0));
      Value *g = byteComponent(component(value, 1));
      Value *b = byteComponent(component(value, 2));
      Value *a = isa<FixedVectorType>(value->getType()) ?
         byteComponent(component(value, 3)) : ConstantInt::get(i32, 255);

      return builder.CreateOr(
         builder.CreateOr(r, builder.CreateShl(g, ConstantInt::get(i32, 8))),
         builder.CreateOr(builder.CreateShl(b, ConstantInt::get(i32, 16)),
                          builder.CreateShl(a, ConstantInt::get(i32, 24))));
   }

   void storeStorageImage(Value *coord, Value *rgba)
   {
      Value *imageBase = dispatchArgI32(VTVK_RT_ARGS_IMAGE_ADDR);
      Value *stride = dispatchArgI32(VTVK_RT_ARGS_IMAGE_STRIDE);
      Value *x = asInt32(component(coord, 0));
      Value *y = isa<FixedVectorType>(coord->getType()) ?
         asInt32(component(coord, 1)) : ConstantInt::get(i32, 0);
      Function *fn = declare("llvm.riscv.ventus.image.store.rgba8",
                             builder.getVoidTy(),
                             {i32, i32, i32, i32, i32});
      builder.CreateCall(fn, {imageBase, x, y, stride, packRGBA8(rgba)});
   }

   Value *lowerLoadConst(nir_load_const_instr *instr)
   {
      if (instr->def.num_components == 1)
         return ConstantInt::get(scalarType(instr->def.bit_size), instr->value[0].u64);

      std::vector<Constant *> values;
      Type *scalar = scalarType(instr->def.bit_size);
      for (unsigned i = 0; i < instr->def.num_components; ++i)
         values.push_back(ConstantInt::get(scalar, instr->value[i].u64));
      return ConstantVector::get(values);
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
         Value *base = asInt32(value(deref->parent));
         nir_deref_instr *parent = nir_deref_instr_parent(deref);
         int offset = parent ? glsl_get_struct_field_offset(parent->type,
                                                            deref->strct.index)
                             : -1;
         if (offset > 0)
            base = builder.CreateAdd(base, ConstantInt::get(i32, offset));
         return base;
      }
      case nir_deref_type_array:
      case nir_deref_type_ptr_as_array: {
         Value *base = asInt32(value(deref->parent));
         unsigned stride = nir_deref_instr_array_stride(deref);
         Value *index = asInt32(value(deref->arr.index));
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
      case nir_op_iand:
         return intBin([&](Value *a, Value *b) { return builder.CreateAnd(a, b); });
      case nir_op_ior:
         return intBin([&](Value *a, Value *b) { return builder.CreateOr(a, b); });
      case nir_op_ixor:
         return intBin([&](Value *a, Value *b) { return builder.CreateXor(a, b); });
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
         Value *vec = ConstantAggregateZero::get(FixedVectorType::get(i32, count));
         for (unsigned i = 0; i < count; ++i)
            vec = builder.CreateInsertElement(vec, asInt32(src(i)), i);
         return vec;
      }
      default:
         if (debugLowering)
            fprintf(stderr, "ventus-nir-to-llvm: unsupported ALU op %s\n",
                    nir_op_infos[alu->op].name);
         return undefFor(&alu->def);
      }
   }

   Value *lowerIntrinsic(nir_intrinsic_instr *intrin)
   {
      switch (intrin->intrinsic) {
      case nir_intrinsic_load_ray_launch_id: {
         return rayLaunchId();
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
         if (auto *bindingConst = dyn_cast<ConstantInt>(bindingValue)) {
            Value *addr =
               descriptorAddressForBinding(bindingConst->getZExtValue());
            return descriptorValueForDef(&intrin->def, addr);
         }
         return descriptorValueForDef(&intrin->def, ConstantInt::get(i32, 0));
      }
      case nir_intrinsic_load_ubo: {
         Value *base = asInt32(value(intrin->src[0]));
         Value *offset = asInt32(value(intrin->src[1]));
         return loadGlobalForDef(base, offset, &intrin->def);
      }
      case nir_intrinsic_image_deref_store:
      case nir_intrinsic_image_store: {
         storeStorageImage(value(intrin->src[1]), value(intrin->src[3]));
         return nullptr;
      }
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_image_size: {
         Type *vecType = FixedVectorType::get(i32, 3);
         Value *vec = ConstantAggregateZero::get(vecType);
         vec = builder.CreateInsertElement(
            vec, dispatchArgI32(VTVK_RT_ARGS_WIDTH), uint64_t(0));
         vec = builder.CreateInsertElement(
            vec, dispatchArgI32(VTVK_RT_ARGS_HEIGHT), uint64_t(1));
         vec = builder.CreateInsertElement(
            vec, dispatchArgI32(VTVK_RT_ARGS_DEPTH), uint64_t(2));
         return vec;
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
         return builder.CreateLoad(defType(&intrin->def), ptr);
      }
      case nir_intrinsic_store_deref: {
         Value *ptr = value(intrin->src[0]);
         if (ptr->getType()->isPointerTy())
            builder.CreateStore(asStorageBits(value(intrin->src[1])), ptr);
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
                                   privatePtr(value(intrin->src[0])));
      }
      case nir_intrinsic_ventus_rt_store_slot: {
         builder.CreateStore(asInt32(value(intrin->src[0])),
                             privatePtr(value(intrin->src[1])));
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
      PHINode *llvmPhi = builder.CreatePHI(defType(&phi->def), 0, "nir.phi");
      defs[&phi->def] = llvmPhi;
      phis.push_back({phi, llvmPhi});
   }

   bool visitIf(nir_if *ifStmt)
   {
      Function *fn = builder.GetInsertBlock()->getParent();
      BasicBlock *thenBB = firstBlock(&ifStmt->then_list);
      BasicBlock *elseBB = firstBlock(&ifStmt->else_list);
      BasicBlock *mergeBB = BasicBlock::Create(context, "nir.if.end", fn);

      if (!thenBB)
         thenBB = mergeBB;
      if (!elseBB)
         elseBB = mergeBB;

      builder.CreateCondBr(asBool(value(ifStmt->condition)), thenBB, elseBB);

      if (!exec_list_is_empty(&ifStmt->then_list)) {
         builder.SetInsertPoint(thenBB);
         visitCFList(&ifStmt->then_list);
         branchTo(mergeBB);
      }

      if (!exec_list_is_empty(&ifStmt->else_list)) {
         builder.SetInsertPoint(elseBB);
         visitCFList(&ifStmt->else_list);
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
      visitCFList(&loop->body);
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
      foreach_list_typed(nir_cf_node, node, node, list)
         visitCFNode(node);
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
      if (shaders[i])
         translator.translate(shaders[i]);
   }

   return module;
}
