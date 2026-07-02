/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "ventus_nir_to_llvm.h"

#include <memory>
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

using namespace llvm;

namespace {

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
        i64(Type::getInt64Ty(context)), f32(Type::getFloatTy(context)) {}

   void translate(nir_shader *shader)
   {
      nir_function_impl *impl = nir_shader_get_entrypoint(shader);
      if (!impl)
         return;

      resetFunctionState();

      Function *fn = createEntry(shader);
      mapBlocks(fn, impl);
      builder.SetInsertPoint(BasicBlock::Create(context, "entry", fn));
      allocateVariables(shader, impl);

      BasicBlock *first = firstBlock(&impl->body);
      if (first)
         builder.CreateBr(first);

      visitCFList(&impl->body);
      finishPhis();

      if (builder.GetInsertBlock() && !builder.GetInsertBlock()->getTerminator())
         builder.CreateRetVoid();

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

   std::unordered_map<const nir_def *, Value *> defs;
   std::unordered_map<const nir_variable *, Value *> variables;
   std::unordered_map<const nir_block *, BasicBlock *> blocks;
   std::unordered_map<const nir_block *, BasicBlock *> blockEnds;
   std::vector<std::pair<nir_phi_instr *, PHINode *>> phis;
   std::vector<BasicBlock *> breakTargets;
   std::vector<BasicBlock *> continueTargets;

   void resetFunctionState()
   {
      defs.clear();
      variables.clear();
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

   Value *privatePtr(Value *addr)
   {
      return builder.CreateIntToPtr(asInt32(addr), PointerType::get(context, 5));
   }

   Function *createEntry(nir_shader *shader)
   {
      std::string name = "vt_" + stage_name(shader->info.stage);
      if (shader->info.stage == MESA_SHADER_RAYGEN)
         name = "main";
      return Function::Create(FunctionType::get(Type::getVoidTy(context), false),
                              GlobalValue::ExternalLinkage, name, module);
   }

   void allocateVariables(nir_shader *shader, nir_function_impl *impl)
   {
      auto allocVar = [&](nir_variable *var) {
         if (!var || variables.count(var))
            return;
         variables[var] = builder.CreateAlloca(i32, nullptr,
                                               var->name ? var->name : "nir.var");
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

   Value *asBool(Value *v)
   {
      if (v->getType()->isIntegerTy(1))
         return v;
      if (v->getType()->isIntegerTy())
         return builder.CreateICmpNE(v, ConstantInt::get(v->getType(), 0));
      if (v->getType()->isFloatingPointTy())
         return builder.CreateFCmpONE(v, ConstantFP::get(v->getType(), 0.0));
      if (auto *vec = dyn_cast<FixedVectorType>(v->getType()))
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
         return builder.CreateBitCast(v, f32);
      return UndefValue::get(f32);
   }

   Value *asInt32(Value *v)
   {
      if (v->getType()->isIntegerTy(32))
         return v;
      if (v->getType()->isFloatingPointTy())
         return builder.CreateBitCast(v, i32);
      if (v->getType()->isIntegerTy())
         return builder.CreateZExtOrTrunc(v, i32);
      return UndefValue::get(i32);
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
         auto it = variables.find(deref->var);
         if (it != variables.end())
            return it->second;
      }

      if (deref->deref_type != nir_deref_type_var)
         return value(deref->parent);

      return UndefValue::get(PointerType::get(context, 0));
   }

   Value *lowerAlu(nir_alu_instr *alu)
   {
      auto src = [&](unsigned idx) { return value(alu->src[idx].src); };

      switch (alu->op) {
      case nir_op_mov:
         return src(0);
      case nir_op_iadd:
         return builder.CreateAdd(asInt32(src(0)), asInt32(src(1)));
      case nir_op_imul:
         return builder.CreateMul(asInt32(src(0)), asInt32(src(1)));
      case nir_op_fadd:
         return builder.CreateFAdd(asFloat(src(0)), asFloat(src(1)));
      case nir_op_fsub:
         return builder.CreateFSub(asFloat(src(0)), asFloat(src(1)));
      case nir_op_fmul:
         return builder.CreateFMul(asFloat(src(0)), asFloat(src(1)));
      case nir_op_fge:
         return builder.CreateFCmpOGE(asFloat(src(0)), asFloat(src(1)));
      case nir_op_u2f32:
         return builder.CreateUIToFP(asInt32(src(0)), f32);
      case nir_op_f2u32:
         return builder.CreateFPToUI(asFloat(src(0)), i32);
      case nir_op_bcsel:
         return builder.CreateSelect(asBool(src(0)), src(1), src(2));
      case nir_op_vec2:
      case nir_op_vec3:
      case nir_op_vec4: {
         unsigned count = alu->op == nir_op_vec2 ? 2 : alu->op == nir_op_vec3 ? 3 : 4;
         Value *vec = UndefValue::get(FixedVectorType::get(i32, count));
         for (unsigned i = 0; i < count; ++i)
            vec = builder.CreateInsertElement(vec, asInt32(src(i)), i);
         return vec;
      }
      default:
         return undefFor(&alu->def);
      }
   }

   Value *loadLaunchComponent(const char *name)
   {
      Function *fn = declare(name, i32, {});
      return builder.CreateCall(fn);
   }

   Value *lowerIntrinsic(nir_intrinsic_instr *intrin)
   {
      switch (intrin->intrinsic) {
      case nir_intrinsic_load_ray_launch_id: {
         Value *vec = UndefValue::get(FixedVectorType::get(i32, 3));
         vec = builder.CreateInsertElement(vec, loadLaunchComponent("llvm.riscv.ventus.rt.launch.id.x"), uint64_t(0));
         vec = builder.CreateInsertElement(vec, loadLaunchComponent("llvm.riscv.ventus.rt.launch.id.y"), uint64_t(1));
         vec = builder.CreateInsertElement(vec, loadLaunchComponent("llvm.riscv.ventus.rt.launch.id.z"), uint64_t(2));
         return vec;
      }
      case nir_intrinsic_load_ray_launch_size: {
         Value *vec = UndefValue::get(FixedVectorType::get(i32, 3));
         vec = builder.CreateInsertElement(vec, loadLaunchComponent("llvm.riscv.ventus.rt.launch.size.x"), uint64_t(0));
         vec = builder.CreateInsertElement(vec, loadLaunchComponent("llvm.riscv.ventus.rt.launch.size.y"), uint64_t(1));
         vec = builder.CreateInsertElement(vec, loadLaunchComponent("llvm.riscv.ventus.rt.launch.size.z"), uint64_t(2));
         return vec;
      }
      case nir_intrinsic_load_deref: {
         Value *ptr = value(intrin->src[0]);
         if (!ptr->getType()->isPointerTy())
            return undefFor(&intrin->def);
         return builder.CreateLoad(defType(&intrin->def), ptr);
      }
      case nir_intrinsic_store_deref: {
         Value *ptr = value(intrin->src[0]);
         if (ptr->getType()->isPointerTy())
            builder.CreateStore(value(intrin->src[1]), ptr);
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
         builder.CreateStore(value(intrin->src[0]),
                             privatePtr(value(intrin->src[1])));
         return nullptr;
      }
      case nir_intrinsic_ventus_rt_traverse: {
         Function *fn = declare("llvm.riscv.ventus.rt.traverse.i32", i32, {i32});
         return builder.CreateCall(fn, {asInt32(value(intrin->src[0]))});
      }
      case nir_intrinsic_ventus_rt_release: {
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
         result = undefFor(&nir_instr_as_undef(instr)->def);
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
         builder.CreateRetVoid();
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
