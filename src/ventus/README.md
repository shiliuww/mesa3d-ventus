# Mesa Ventus Vulkan RT Backend

`src/ventus/` contains the Mesa-side Ventus Vulkan UMD and RT compiler path.
It is no longer a future directory skeleton: the matched Mesa, LLVM, driver,
and Spike revisions can run the SaschaWillems `raytracingshadows` application,
emit a Ventus RISC-V ELF, execute it on Ventus Spike, and read back a PPM image.

## Current Pipeline

```text
Vulkan RT application
  -> Vulkan loader
  -> libvulkan_ventus.so
  -> SPIR-V -> Mesa NIR
  -> Ventus RT NIR lowering
  -> Ventus NIR-to-LLVM translation
  -> LLVM Ventus/RISC-V code generation and ELF link
  -> Ventus driver bridge
  -> Ventus Spike
  -> storage image readback
```

The current full-app baseline is native to the Ventus ICD. It does not render
through lavapipe or llvmpipe.

## Directory Map

| Path | Responsibility |
| --- | --- |
| `compiler/` | RT NIR ABI layout, trace lowering, traversal loop, shader scratch, descriptors, and tests |
| `llvm/` | NIR control flow and intrinsic translation to Ventus LLVM IR |
| `vulkan/` | Vulkan objects, RT pipeline/groups, AS/SBT/descriptors, trace dispatch, ELF emission, driver upload, and image readback |
| `tests/` | Focused NIR/compiler tests; these do not replace the full-app Spike test |
| `tools/rt_probe/` | Diagnostic precompiled kernel for isolating bridge failures |
| `RT_SBT_MEMORY_ABI.md` | Current bring-up dispatch-args, descriptor, SBT, and per-lane scratch contract |
| `log.md` | Chronological development notes; old entries are not current status |

## Compiler Flow

RT pipeline creation uses Mesa's SPIR-V frontend and NIR infrastructure, then
applies the Ventus-specific path:

```text
vkCreateRayTracingPipelinesKHR
  -> vk_pipeline_shader_stage_to_nir()
  -> vt_nir_lower_rt_io()
  -> vt_nir_build_traversal()
  -> vt_nir_lower_rt_abi() and descriptor/scratch lowering
  -> vt_nir_to_llvm_translate()
  -> llc + ld.lld
  -> one pipeline-owned RISC-V ELF
```

The backend emits the current hard-operation roles:

```text
RT_TRAVERSE(result_out, handoff_addr)
RT_RELEASE(handoff_addr)
```

Ray, AS, SBT, payload, hit and launch data remain in the shader-visible data
plane. They are not extra hard-operation operands.

## Vulkan/Driver Execution

The verified `raytracingshadows` path currently performs all of the following:

- creates the Ventus Vulkan device and RT pipeline;
- builds triangle BLAS and instance TLAS into Ventus BVH ABI v1;
- materializes SBT regions and a restricted descriptor table;
- collects dispatch state in `vkCmdTraceRaysKHR`;
- emits and links the generated Ventus ELF;
- uploads ELF, BVH, descriptors, SBT, camera data and storage image through
  `vtvk_driver_bridge`;
- launches the raygen kernel on the Ventus Spike backend;
- copies the storage image back and optionally writes P6 PPM.

Relevant runtime controls include:

| Variable | Purpose |
| --- | --- |
| `VK_DRIVER_FILES` | Select the generated Ventus ICD JSON |
| `VENTUS_VK_DRIVER_BACKEND=spike` | Select the Spike driver backend |
| `VENTUS_VK_DRIVER_BRIDGE=1` | Enable resource upload and backend launch |
| `VENTUS_VK_EMIT_ELF=1` | Compile/link a generated RT ELF |
| `VENTUS_VK_SHADER_ARTIFACT_DIR` | Directory for generated LLVM/ELF artifacts |
| `VENTUS_VK_DUMP_IMAGE` | PPM path for storage-image readback |
| `VENTUS_VK_DRIVER_LIB` | Optional explicit driver shared library |

## Build and Focused Tests

The repository-level build script carries the exact options needed by this
path. From the `ventus-env` root, build all runtime pieces and the pinned
workload with:

```sh
git submodule update --init
bash build-ventus.sh --build "rt-toolchain;spike;driver-spike;mesa;rt-workload"
```

`rt-toolchain` installs the Ventus `llc`, `ld.lld`, `llvm-nm`, and `crt0.o`
needed for runtime ELF generation. It deliberately does not build the
aggregate OpenCL libclc object, which is outside the Vulkan RT path. Its
standalone riscv32 libclc build uses `llvm/build-rt-libclc/`, separate from the
general OpenCL build directory. The superproject's
`docs/rtcore-spike-p1.md` records the complete review and reproduction flow.

Focused compiler tests are registered in the `ventus` Meson suite, including
`vt_rt_minimal_loop`, `vt_rt_image_smoke`, and
`vt_rt_flagged_image_smoke`. The image smokes validate compiler-side lowering;
they must not be cited as evidence that Spike rendered a full Vulkan sample.

Run the full application and compare its PPM byte-for-byte with the frozen P1
baseline using:

```sh
tools/rtcore/verify_compat_p1_image.sh
```

The runner resolves Mesa, LLVM, driver, Spike, workload and scene paths below
the current superproject root and fails if the driver links to a different
Spike library. Workload source pins and overlay provenance live under
`workloads/rt/`.

## Current Limits

The existing image is a functional vertical slice, not a Vulkan RT conformance
claim:

- the maintained full-app evidence currently covers one triangle/instance
  scene and raygen/miss/closest-hit shader set;
- AABB/procedural geometry is not wired through the verified image path;
- callable, indirect trace/build, AS copy/serialization, and deferred
  operations are unsupported or placeholder entrypoints;
- descriptor handling is a restricted set-0 profile;
- trace execution currently enters the backend from `vkCmdTraceRaysKHR` rather
  than a complete submit-time command execution model;
- feature/property advertisement, recursion validation, synchronization and
  standard image readback still need conformance-oriented closure;
- Spike proves functional behavior only; it does not prove RTL scheduling,
  timing or backpressure.

Semantic operation and capability authority lives outside Mesa under
`RTcore/docs/design/`. This README describes only the implementation present in
this source tree.
