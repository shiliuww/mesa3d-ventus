> This is a chronological development log. Earlier “current gap” statements
> describe their version only. The latest verified state is v14.

# v1

实现内容：

  - 新增 Ventus RT NIR 公共接口：mesa/src/ventus/compiler/vt_nir.h:1
  - 新增 v1 shared-slot traversal builder：mesa/src/ventus/compiler/vt_nir_build_traversal.c:1
  - 新增 monolithic trace_ray lowering：mesa/src/ventus/compiler/vt_nir_lower_rt_io.c:1
      - 在 nir_intrinsic_trace_ray 位置写 Ventus RT slot
      - 生成 if (hit) closest_hit else miss 的 NIR CFG
      - closest/miss 通过 callback 接入，避免伪造还不存在的 pipeline/SBT 实现

  - 新增 RT ABI placeholder：mesa/src/ventus/compiler/vt_nir_lower_rt_abi.c:1
  - 新增 CFG-aware NIR-to-LLVM 桥：mesa/src/ventus/llvm/ventus_nir_to_llvm.cpp:1
      - nir_block -> llvm::BasicBlock 映射
      - if/loop/jump CFG traversal
      - PHI 延迟补 incoming
      - 基础 ALU/deref/load-store/RT launch/trace 边界 lowering

  - 新增局部 Meson 清单：mesa/src/ventus/meson.build:1

  验证：

  - vt_nir_*.c 已用 RTcore mesa/build 的新 Mesa/NIR include 和宏做 cc -fsyntax-only，通过。
  - ventus_nir_to_llvm.cpp 已用 LLVM 18 c++ -fsyntax-only + RTcore Mesa include 做语法检查，通过。
  - rg 确认 mesa/src/ventus:1 没有引用 compiler-old、gpgpusim、ptx、lavapipe。

  注意：上一轮中断前我已经动过 compiler-old/mesa_ventus_backend/nir_to_ventus_llvm.cpp，但 compiler-old 是未跟踪目录，没有 git 基线可安全恢
  复。我没有继续使用它；后续应只看 mesa/src/ventus。当前还没把 mesa/src/ventus/meson.build 接入 Mesa 顶层构建，这是下一步。

# v2
- mesa/src/ventus/compiler/vt_nir.h:1
      - 增加 VT_RT_EXECUTION_ITERATIVE_RAYGEN_LOOP
      - 增加 per-lane L1 scratchpad layout
      - 增加固定 payload offsets：depth/weight/color/rayOrigin/rayDirection

  - mesa/src/ventus/compiler/vt_nir_build_traversal.c:1
      - 默认 raygen/closest-hit/miss 各自 layout，stride 默认 256B
      - TraceRay 写 VT_RT_STATUS_TRACE_REQUEST
      - hit 判断改为 status == VT_RT_STATUS_HIT
      - 增加 vt_nir_load_payload_word/store_payload_word

  - mesa/src/ventus/compiler/vt_nir_lower_rt_io.c:1
      - 只接受 iterative raygen-loop 执行模型
      - 默认不内联 closest-hit/miss callback，符合“closest-hit 只填 payload，不递归 TraceRay”

  - mesa/src/ventus/compiler/vt_nir_lower_rt_abi.c:1
      - 根据 shader stage 选择 scratchpad layout，并设置 shader->scratch_size

# v3
新增 mesa/src/ventus/compiler/vt_nir_scratch_layout.c:1
      - vt_nir_lower_shader_scratch()：调用 Mesa nir_lower_vars_to_scratch()，让 NIR 自动给 shader 内 scratch 变量分配常量 offset，并更新
        shader->scratch_size

      - vt_nir_pipeline_compute_scratch()：pipeline creation 传入 shader 列表，返回 max(shader->scratch_size)
      - vt_nir_pipeline_scratch_info：保留 raygen / closest-hit / miss 各自 scratch size，方便调试和上报

  - 修改 mesa/src/ventus/compiler/vt_nir_lower_rt_abi.c:1
      - 撤掉上一轮错误的 “scratch_size 至少等于 stride”
      - 改为使用 per-shader 自动 scratch lowering 结果

  - 更新 mesa/src/ventus/compiler/vt_nir.h:1
      - 暴露 pipeline-level scratch sizing API

# v4
 本轮完成内容：

  - 在 Mesa build 中接入 src/ventus。
  - 新增 src/ventus/tests/vt_rt_minimal_loop_test.c。
  - 测试构造最小 raygen NIR，包含 trace_ray。
  - 跑 vt_nir_lower_rt_io_monolithic() 和 vt_nir_lower_rt_abi()。
  - 验证：
      - trace_ray 被消除。
      - 生成了 store_shared / load_shared，说明进入 per-lane scratchpad ABI 路径。
      - pipeline scratch sizing 取 max：raygen 48、miss 16、closest-hit 80，最终 per_lane_scratch = 80。

  验证命令已通过：

  meson test -C /home/guanys/RTcore/mesa/build vt_rt_minimal_loop --print-errorlogs

  结果：

  1/1 ventus - mesa:vt_rt_minimal_loop OK

  注意边界：这已经是 NIR lowering + iterative RT scratch layout 的最小闭环，但还不是完整 GLSL/HLSL -> SPIR-V -> NIR -> LLVM -> RISC-V 机器码
  闭环。下一步要把这个测试从手造 NIR 扩展到真实 SPIR-V 输入，随后接 NIR-to-LLVM intrinsic emission。手工 gcc 链接 NIR 会踩 Mesa ABI/config坑，后续这类验证应走 Meson target。

# v5
测试链路是：

  GLSL raygen (.rgen) → glslangValidator → .spv → spirv_to_nir() → trace_ray NIR → Ventus RT lowering → shared scratch load/store → pipeline scratch max

  验证已通过：

  meson test -C /home/guanys/RTcore/mesa/build vt_rt_minimal_loop --print-errorlogs

  结果：

  1/1 ventus - mesa:vt_rt_minimal_loop OK

  手动运行也确认：

  PASS vt_rt_minimal_loop spirv_input=1 trace_ray_lowered=1 shared_store=18 shared_load=3 per_lane_scratch=80

# v6
 现在链路是：

  GLSL raygen .rgen → glslangValidator → SPIR-V → spirv_to_nir() → Ventus RT lowering → image smoke renderer → PPM

  验证通过：

  meson test -C /home/guanys/RTcore/mesa/build --suite ventus --print-errorlogs

  结果：

  1/2 ventus - mesa:vt_rt_minimal_loop OK
  2/2 ventus - mesa:vt_rt_image_smoke  OK

  生成的图像在：

  mesa/build/src/ventus/tests/vt_rt_smoke.ppm
  mesa/build/src/ventus/tests/vt_rt_smoke.png

这一步还不是 LLVM/RISC-V 机器码执行出的图，而是“真实 SPIR-V 必须先通过 Mesa NIR 和 Ventus RT lowering，然后 host-side smoke renderer 生成图像”

# v7
- 增加 Ventus RT ABI layout allocator：ray_state / payload / hit_record / continuation 分区，continuation 预留且 v1 为 0，避免和 hit record冲突。

- 增加 hit record scratch layout，包括 status / hit_t / sbt_index / shader_record_ptr / primitive_id / instance_id / geometry_id /hit_kind / barycentrics 等字段。

- 增加 NIR lowering：
    - ray payload / incoming payload deref -> shared scratch 固定 offset
    - RT system values -> hit record loads
    - load_shader_record_ptr -> hit record 中的 shader record pointer

- 更新 vt_nir_lower_rt_abi() 调用顺序，并保证 shader->scratch_size >= ABI total size。
- 扩展 vt_rt_minimal_loop 测试，验证 payload deref、hit record、shader record ptr lowering 都生效。

  验证已通过：

  meson test -C mesa/build --suite ventus --print-errorlogs

  注意边界：shader_record_ptr 当前已经有 ABI slot 和 lowering，但 traversal builder 里真实 SBT loader 还没实现，目前是占位来源。下一步应接真实 SBT entry 地址计算，把 selected hit/miss shader record pointer 写入 hit record。

# v8
 已按 SBT 公式替换临时逻辑，并提交：

  de42e5828 Compute Ventus RT shader record pointers from SBT

  改动核心：

  shader_record_ptr =
      hit_sbt_base_addr
    + hit_sbt_record_stride * sbt_index
    + shader_group_handle_size;

  具体在 vt_nir_build_traversal.c 里新增 build_hit_shader_record_ptr()，不再用 sbt_index 伪装 64-bit pointer。SBT region 参数先放在
  vt_nir_rt_config：

  uint64_t hit_sbt_base_addr;
  uint32_t hit_sbt_record_stride;
  uint32_t shader_group_handle_size;

  验证方式也加了：vt_rt_minimal_loop_test 设置：

  hit_sbt_base = 0x100000000
  hit_sbt_stride = 96
  shader_group_handle_size = 32

  然后扫描 lowering 后的 NIR，确认有 imul 96，也就是已经生成 sbt_index * hit_sbt_record_stride 这步真实 SBT entry 计算。

  直接运行输出为：

  PASS vt_rt_minimal_loop spirv_input=1 trace_ray_lowered=1 abi_hit_record=1 sbt_entry_calc=1 hit_sbt_base=0x100000000 hit_sbt_stride=96
  shader_group_handle_size=32 shared_store=27 shared_load=14 per_lane_scratch=80

  测试也通过：

  meson test -C mesa/build --suite ventus --print-errorlogs

# v9
去除 vt_rt_minimal_loop_test.c手动写的max size，保证测试验证与实际pass一致性

# v10
- 扩展 RT ABI layout：mesa/src/ventus/compiler/vt_nir.h
      - 新增 ANY_HIT / INTERSECTION shader layout。
      - 新增 control 区：done/incomplete/accept/ignore/terminate/skip_closest_hit。
      - 新增 candidate_hit_record，保留原 hit_record 作为 committed hit。
      - 新增 front_face/opaque/need_software_opacity_test 字段。
      - 新增独立 hit_attrib 区，避免和 payload / continuation scratch 冲突。

  - 补齐 ABI allocator 和 helper：mesa/src/ventus/compiler/vt_nir_rt_abi_layout.c、mesa/src/ventus/compiler/vt_nir_build_traversal.c
      - 当前 byte layout：
          - ray_state: 0..79
          - payload: 80..123
          - control: 128..151
          - candidate_hit: 160..231
          - committed_hit: 240..311
          - hit_attrib: 320..351
          - total per-lane ABI scratch: 352

  - 新增 NIR lowering：mesa/src/ventus/compiler/vt_nir_lower_rt_abi_data.c
      - IgnoreIntersectionKHR -> 写 accept=false / ignore=true / terminate=false
      - TerminateRayKHR -> 写 accept=true / ignore=false / terminate=true
      - ReportIntersectionKHR -> 检查 tmin <= hitT <= committedTmax，写 candidate hit_t/hit_kind/front_face/opaque/need_software_opacity_test
      - frontFace、ray_t_min、ray_flags、cull_mask、hit builtins 都能从 ABI scratch lowering。
      - bool flag 统一以 32-bit scratch word 表示，避免后续 NIR-to-LLVM 遇到 1-bit shared memory。

  验证：

  - ninja -C mesa/build src/ventus/tests/vt_rt_minimal_loop_test src/ventus/tests/vt_rt_image_smoke_test
  - meson test -C mesa/build --suite ventus --print-errorlogs
  - 结果：2/2 OK
  - minimal 输出确认：
      - abi_candidate_committed=1
      - abi_control=1
      - per_lane_scratch=352
      - control_base=128 candidate_base=160 committed_base=240 hit_attrib_base=320

  注意：这轮已经补齐 ABI layout 和 NIR intrinsic lowering；还没有实现真实 BVH traversal 中“发现 non-opaque candidate 后实际 dispatch any-hit/
  intersection shader”的完整运行时调度。下一步 NIR-to-LLVM 可以先消费这些固定 offset 的 scratch load/store。

新增内容：

  - mesa/src/ventus/tests/vt_rt_flagged_raygen.rgen
      - GLSL raygen 输入。
      - traceRayEXT 使用 ray_flags=0x16，即 NoOpaque | TerminateOnFirstHit | CullBackFacingTriangles。
      - 同时使用 cull_mask=0x7f、sbt_offset=3、sbt_stride=5、miss_index=2。

  - mesa/src/ventus/tests/vt_rt_flagged_image_smoke_test.c
      - 跑完整链路：.rgen -> glslangValidator -> SPIR-V -> spirv_to_nir() -> Ventus RT lowering -> image smoke renderer -> PPM。
      - 在 spirv_to_nir() 后检查 trace_ray 的 flags/cull/SBT/miss 参数确实是 shader 里的值。
      - 在 Ventus lowering 后检查 trace_ray 被消除，并生成 shared scratch load/store。
      - 额外构造 any-hit / intersection probe，验证刚新增的：
          - load_front_face
          - load_ray_flags
          - load_cull_mask
          - ignore_ray_intersection
          - load_ray_t_min
          - report_ray_intersection

  - mesa/src/ventus/tests/meson.build
      - 新增 vt_rt_flagged_raygen.spv custom target。
      - 新增 vt_rt_flagged_image_smoke 测试。

# v11
- mesa/src/ventus/compiler/vt_nir_rt_abi_layout.c 的默认 ABI 顺序改为 RTcore-friendly：

    ray_state
    control
    candidate_hit
    committed_hit
    hit_attrib
    payload
    continuation

  - mesa/src/ventus/tests/vt_rt_minimal_loop_test.c 更新了 layout 顺序断言和输出。

  当前实际 byte layout：

  ray_state:      0..79
  control:        80..103
  candidate_hit:  112..183
  committed_hit:  192..263
  hit_attrib:     272..303
  payload:        304..347
  total:          352
- mesa/src/ventus/compiler/vt_nir_rt_abi_layout.c 的默认 ABI 顺序改为 RTcore-friendly：

    ray_state
    control
    candidate_hit
    committed_hit
    hit_attrib
    payload
    continuation

  - mesa/src/ventus/tests/vt_rt_minimal_loop_test.c 更新了 layout 顺序断言和输出。

  当前实际 byte layout：

  ray_state:      0..79
  control:        80..103
  candidate_hit:  112..183
  committed_hit:  192..263
  hit_attrib:     272..303
  payload:        304..347
  total:          352

# v12
 1. 新增 Ventus RT 专用 NIR intrinsic：
      - ventus_rt_load_slot
      - ventus_rt_store_slot

  2. 完成 LDS/PDS 分层：
      - 普通 nir_load_shared/store_shared 继续走 LLVM addrspace(3)，对应 LDS
      - RT ABI per-lane slot 读写改走 ventus_rt_load_slot/store_slot
      - NIR-to-LLVM 将 RT slot lowering 到 LLVM addrspace(5)，对应 PDS

  3. release 规则也在这个提交里：
      - rt.release(slot) 只在 terminal traversal status 下插入：
          - COMPLETE_HIT
          - COMPLETE_MISS
          - TERMINATED

  4. 测试更新：
      - RT smoke/minimal 测试现在检查 ventus_rt_*_slot
      - 最小链路输出确认：
          - rt_slot_store=58
          - rt_slot_load=17
          - shared_store=0
          - shared_load=0

  验证已跑过：

  meson test -C /home/guanys/RTcore/mesa/build vt_rt_minimal_loop vt_rt_image_smoke vt_rt_flagged_image_smoke --print-errorlogs

# v13
已推进到一个明确里程碑：full app 已经能通过 Vulkan Loader 找到 libvulkan_ventus.so，枚举到 Ventus RT Spike，SaschaWillems raytracingbasic 能跑完 1 frame benchmark，并且 RT pipeline 确认走的是：

  SPIR-V -> NIR -> Ventus lowering -> LLVM

  不是 lavapipe。ldd libvulkan_ventus.so 里也没有 lavapipe/llvmpipe/gallium，只看到 libLLVM.so.20.1。

  本次主要落点：

  - 新增 Mesa vulkan-drivers=ventus 路径，生成 libvulkan_ventus.so 和 ventus_devenv_icd.x86_64.json
  - vulkaninfo --summary 能看到 deviceName = Ventus RT Spike
  - raytracingbasic --benchmark --benchmarkframes 1 --width 160 --height 96 正常退出
  - RT pipeline 在 mesa/src/ventus/vulkan/vtvk_pipeline.cpp 中实际调用 vk_pipeline_shader_stage_to_nir、Ventus RT lowering、vt_nir_to_llvm_translate
  - 补了 AS 对象和非零 pseudo buffer/AS device address，不再让 full app 靠 vkCreateAccelerationStructureKHR 失败后继续跑

  验证结果：

  device : Ventus RT Spike
  MESA: info: ventus-vk: RT pipeline compiled via SPIR-V -> NIR -> Ventus lowering -> LLVM
  Benchmark finished
  frames : 1

  但我必须把边界说清楚：这还不是“真实 Spike 渲染完成”。现在 CmdBuildAccelerationStructuresKHR 和 CmdTraceRaysKHR 已经进入 Ventus ICD，但日志明确显示：

  Vulkan AS -> Spike BVH is not wired yet
  Spike backend is not wired yet

  所以当前是 full app Vulkan 控制流、ICD、device、pipeline、AS/SBT 地址链条打通；还差最后真正 backend：AS 转 Spike BVH、LLVM code object/ELF 发射、Spike memory 初始化、执行 raygen kernel、storage image 写回。现有 Ventus Mesa RT tests 也已通过：3/3 OK。

# v14

v13 中列出的 backend 缺口已经闭合为一条有界的真实 Spike 出图链路：

  - `CmdBuildAccelerationStructuresKHR` 生成 Ventus BVH ABI v1 的 triangle BLAS
    和 instance TLAS；
  - RT pipeline 从 SPIR-V/NIR 经 Ventus lowering 和 LLVM 生成 RISC-V ELF；
  - `CmdTraceRaysKHR` 收集 TLAS、SBT、descriptor、camera 和 storage image；
  - `vtvk_driver_bridge` 上传 ELF 与资源并启动 Ventus Spike；
  - Spike 执行 raygen kernel，storage image 回读为 P6 PPM。

2026-07-19 使用干净提交 Mesa `d413bd2`、LLVM `29da3a1`、driver
`c5ce8a7`、Spike `b1ef3ef` 运行 SaschaWillems `raytracingshadows`
160x96 单帧成功。输出包含 136 种颜色，SHA-256 为：

  f43328945bdeb0dda69b3cc5612212170e5596456450184acfa627db8d5d1212

该证据只闭合当前受限 Vulkan/triangle/Spike functional vertical slice；它不代表
Vulkan RT conformance、AABB/procedural/callable 完整支持、submit-time command
execution、RTL RTcore 或硬件 timing/backpressure 已完成。
