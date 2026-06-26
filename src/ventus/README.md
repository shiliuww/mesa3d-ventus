# Ventus Compiler Backend

NIR → Ventus LLVM 编译链路，位于 Mesa 编译器体系内，参照 `src/amd/` 的架构分层。

## 目录规划

```
src/ventus/
├── README.md          # 本文件
├── compiler/          # Ventus-specific NIR lowering passes
├── llvm/              # mesa前端llvm接口
└── vulkan/            # Ventus Vulkan driver（远期建设）

~/ventus/llvm          # NIR → LLVM IR 翻译桥（核心）
```

## LLVM依赖关系

```
RTcore/mesa/src/ventus/llvm/ventus_nir_to_llvm.cpp
    │
    ├─ include: ~/ventus-env/llvm/install/include/
    ├─ link:    ~/ventus-env/llvm/install/lib/libLLVM*.a
    │
    ├─ include: RTcore/mesa/src/compiler/nir/
    ├─ link:    RTcore/mesa/build/src/compiler/nir/libnir.a
    └─ ...
```

实际构建时 `LLVM_DIR=~/ventus-env/llvm/install/lib/cmake/llvm`

## 参考架构：AMD 的布局

| AMD 目录 | 职责 | Ventus 对应 |
|----------|------|------------|
| `src/amd/vulkan/` | RADV Vulkan driver：SPIR-V→NIR→LLVM/ACO→AMD ISA | `ventus/vulkan/`（远期） |
| `src/amd/compiler/` | ACO 编译器：NIR→ACO IR→AMD GCN/RDNA ISA | `ventus/compiler/`（NIR lowering） |
| `src/amd/llvm/` | NIR→LLVM 桥 (`ac_nir_to_llvm.c`) | `ventus/llvm/` |
| `src/amd/common/` | 共享 `ac_nir_lower_*` passes | 复用 compiler 公共层 |

## 编译流水线

```
SPIR-V (.spv)
  │
  ▼
spirv_to_nir()                          ← src/compiler/spirv/ (libvtn)
  │
  ▼
NIR shader (nir_shader*)
  │
  ├─ Ventus-specific NIR passes        ← src/ventus/compiler/
  │   · lower RT I/O (payload → shared memory slot)
  │   · build traversal shader (BVH walk loop)
  │   · lower RT ABI (CSR + slot layout)
  |   可能还有其他的？
  │
  ▼
ventus_nir_to_llvm()                    ← ~/ventus/llvm/
  │   遍历 NIR → LLVM IR (riscv32)
  │
  ▼
LLVM Module (riscv32-unknown-elf)
  │
  ▼
Ventus LLVM backend                     ← llvm/lib/Target/RISCV/ (已有)
  │   DAG → Ventus MachineInstr → MC
  │
  ▼
Ventus ELF (.elf)                       ← spike / 硬件 运行
```

## ventus/compiler/ — NIR RT Lowering 参考 AMD 管线

AMD 的 RT shader 编译入口在 `src/amd/vulkan/radv_pipeline_rt.c:375`，
函数 `radv_rt_nir_to_asm()` 串联全部 NIR pass：

```
radv_rt_nir_to_asm()
  ├─ radv_nir_lower_rt_io_*()           ← RT I/O 模式选择 (monolithic/CPS/func)
  ├─ nir_lower_shader_calls()            ← 函数调用→continuation (CPS only)
  ├─ radv_nir_lower_rt_abi_*()           ← RT ABI 降低
  ├─ radv_optimize_nir()                 ← 15+ NIR pass 优化循环
  ├─ radv_postprocess_nir()              ← 后端预处理
  └─ radv_shader_nir_to_asm()            ← NIR→binary (ACO/LLVM)
```

### AMD RT 专属 NIR passes (`src/amd/vulkan/nir/`)

| 文件 | 行数 | 职责 | 对 Ventus |
|------|------|------|----------|
| `radv_nir_rt_common.c` | 1403 | BVH 遍历 builder：硬件 BVH 指令序列 | 需重写（换 Ventus BVH 指令），NIR builder 写法可借鉴 |
| `radv_nir_rt_traversal_shader.c` | 1401 | 遍历 shader：完整 BVH walk loop（stack、求交、any-hit） | **最重要参考**——需要类似 traversal 生成器 |
| `radv_nir_rt_stage_functions.c` | 919 | 函数调用模式：payload 通过函数参数传递 | 参考 IO/ABI lowering 接口设计 |
| `radv_nir_rt_stage_cps.c` | 677 | CPS 模式：continuation 回调 | Ventus 用 shared memory slot |
| `radv_nir_rt_stage_monolithic.c` | 516 | 单块模式：全部 inline 进 ray_gen | 参考 inline 逻辑 |
| `radv_nir_rt_stage_common.c` | 423 | SBT 索引、shader record 加载 helper | payload/SBT 概念通用 |

### AMD 通用 NIR passes（非 RT 专属，RT 路径也会调到）

| 文件 | 行数 | 职责 |
|------|------|------|
| `radv_nir_lower_descriptors.c` | 717 | Vulkan descriptor set → NIR address lowering |
| `radv_nir_lower_abi.c` | ~350 | 通用 ABI lowering（SGPR/VGPR 约定） |
| `radv_nir_to_llvm.c` | ~280 | LLVM 路径入口：`ac_nir_translate()` |

### Ventus 需要的对应文件

| AMD 文件 | Ventus 对应 (`src/ventus/compiler/`) | 关键差异 |
|----------|--------------------------------------|---------|
| `radv_nir_rt_stage_functions.c` | `vt_nir_lower_rt_io.c` | payload：Ventus 用 shared memory slot，AMD 有三种模式 |
| `radv_nir_lower_rt_abi_*` | `vt_nir_lower_rt_abi.c` | ABI：AMD = SGPR/VGPR，Ventus = CSR + slot layout |
| `radv_nir_lower_descriptors.c` | `vt_nir_lower_descriptors.c` | 地址空间：Ventus 全局地址 + 固定段映射 |
| `radv_nir_rt_traversal_shader.c` | `vt_nir_build_traversal.c` | 遍历逻辑通用，BVH 指令换 Ventus |
| `radv_nir_rt_common.c` | 合并进 `vt_nir_build_traversal.c` | BVH 节点访问 helper |

### VT LLVM 端已具备的机制（不需要在 NIR 层重复）

`~/ventus-env/llvm/` 中 `VentusRTIntrinsicLowering.cpp` 已实现：

- `trace_ray` → 写 ray 参数到 shared memory slot → 调 `vt.rt.intersect`
- 自动插入 shader dispatch poll loop
- 系统值 intrinsic（`launch_size_x` 等）→ CSR 读取（`csrr a0, 0xCC0`）
- UBO / push constant / descriptor 加载 → 地址映射后的 plain load

**因此 NIR 层（`ventus/compiler/`）只需负责**：
1. 多 stage payload 传递约定 → 统一的 shared memory slot 模型
2. traversal shader NIR 生成（BVH walk loop）
3. descriptor/UBO 地址映射

## 需要链接的 `src/compiler/` 库

保持 `src/compiler/` 原封不动，只 include + link：

| 库 | 路径 | 用途 |
|---|------|------|
| `libnir` | `src/compiler/nir/` | NIR IR、builder、优化/降低 pass、序列化 |
| `libvtn` | `src/compiler/spirv/` | `spirv_to_nir()` — SPIR-V → NIR |
| `libcompiler` | `src/compiler/` | `shader_enums`、`glsl_types`、`shader_info` |
| `libmesa_util` | `src/util/` | `ralloc`、`hash_table`、`blob`、`u_math` |

include path：
- `-I src/compiler/` → `nir/`, `spirv/` 头文件
- `-I src/compiler/nir/` → `nir_*.h`
- `-I src/` → `util/`
- `-I build/src/compiler/nir/` → 构建生成的 `nir_opcodes.h` 等

## ventus/llvm/ — NIR→LLVM 翻译器

AMD 的 `src/amd/llvm/ac_nir_to_llvm.c`（~4000 行 + `ac_llvm_build.c` ~3600 行）：

```
ac_nir_context      — 翻译状态（LLVMContext, builder, defs map, blocks map）
visit_alu()         — nir_alu_instr → LLVM binop/select/...
visit_intrinsic()   — 93 个 case，load/store/atomic/ubo/ssbo/image/ballot...
visit_load_const()  — nir_load_const_instr → LLVM constant
visit_tex()         — nir_tex_instr → image_sample
visit_cf_list()     — if/loop/block → LLVM BasicBlock + PHI
```

### 能直接给 Ventus 用吗？不能。

1. **Target**：硬编码 `amdgcn--`，triple、data layout、calling convention 全是 AMD
2. **Intrinsics**：到处都是 `llvm.amdgcn.*`（`struct.buffer.load`、`image.sample`、`ballot.i64`、`wqm.demote`……）
3. **地址空间**：`FLAT/GLOBAL/LDS/CONST` 是 AMD 内存模型
4. **硬件模型**：`wave_size`、`gfx_level`、`vscnt`/`vmcnt` 等待计数器
5. **ABI 回调**：`load_ubo`/`load_ssbo`/`load_sampler_desc` 绑定 Vulkan descriptor set

### 那它的价值在哪？极好的模板。

骨架是干净且通用的：
- NIR SSA def 跟踪（`ssa_defs` 数组映射 `nir_def → LLVMValueRef`）
- 基本块→LLVM BasicBlock 映射
- CFG 遍历（`visit_cf_list → visit_if / visit_loop`）
- PHI 延迟解析（`phi_post_pass`）
- 指令分发 switch 结构

翻译部分必须重写，但有一个完整的参考样本。从骨架开始，先搞定最窄的 RT shader 子集。

## 构建集成

最终通过 meson 集成到 Mesa 构建系统：

```python
# src/ventus/meson.build 草案
if with_ventus_vk or with_ventus_compiler
  subdir('compiler')
  subdir('llvm')
endif
if with_ventus_vk
  subdir('vulkan')
endif
```

短期内可用独立 CMake 构建，链接 prebuilt 的 `libnir.a`、`libvtn.a`、`libmesa_util.a`。

## 现阶段优先级

| 序号 | 任务 | 说明 |
|------|------|------|
| 1 | `ventus/compiler/vt_nir_lower_rt_io.c` | RT I/O lowering：raygen/miss/hit payload 统一到 shared memory slot |
| 2 | `ventus/compiler/vt_nir_build_traversal.c` | traversal shader NIR 生成（BVH walk loop） |
| 3 | `ventus/compiler/vt_nir_lower_rt_abi.c` | RT ABI lowering：payload → CSR + shared memory layout |
| 4 | `ventus/llvm/ventus_nir_to_llvm.cpp` | NIR→LLVM 翻译器，从 `ac_nir_to_llvm.c` 裁剪适配 |
| 5 | `ventus/vulkan/` driver 骨架 | 参照 `radv_shader.c` 的 shader 编译入口 |
| 6 | meson 集成 | 将 ventus 编译目标接入 Mesa 构建 |
