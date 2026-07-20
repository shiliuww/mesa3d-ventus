# Ventus RT SBT Memory ABI

This document records the SBT memory contract shared by the Vulkan bridge,
NIR RT lowering, generated ELF probes, and the Ventus RTU/Spike model.

## Dispatch Args

`CSR_KNL + KNL_ARG_BASE` points at a driver-uploaded dispatch args block in
backend device memory. All SBT addresses in this block are backend device
addresses, not host `VkDeviceAddress` handles.

| Offset | Field |
| ---: | --- |
| 0 | `uint64_t tlas_addr` |
| 8 | `uint64_t image_addr` |
| 16 | `uint64_t camera_addr` |
| 24 | `uint64_t raygen_sbt_addr` |
| 32 | `uint64_t raygen_sbt_size` |
| 40 | `uint64_t raygen_sbt_stride` |
| 48 | `uint64_t miss_sbt_addr` |
| 56 | `uint64_t miss_sbt_size` |
| 64 | `uint64_t miss_sbt_stride` |
| 72 | `uint64_t hit_sbt_addr` |
| 80 | `uint64_t hit_sbt_size` |
| 88 | `uint64_t hit_sbt_stride` |
| 96 | `uint64_t callable_sbt_addr` |
| 104 | `uint64_t callable_sbt_size` |
| 112 | `uint64_t callable_sbt_stride` |
| 120 | `uint32_t width` |
| 124 | `uint32_t height` |
| 128 | `uint32_t depth` |
| 132 | `uint32_t image_format` |
| 136 | `uint32_t image_stride` |
| 140 | `uint32_t camera_size` |
| 144 | `uint32_t scratch_stride` |
| 148 | `uint32_t reserved` |
| 152 | `uint64_t descriptor_table_addr` |
| 160 | `uint32_t descriptor_count` |
| 164 | `uint32_t descriptor_record_stride` |
| 168 | `uint64_t continuation_arena_addr` |
| 176 | `uint32_t continuation_frame_stride` |
| 180 | `uint32_t continuation_context_count` |

The legacy `tlas_addr`, `image_addr`, and `camera_addr` fields are retained for
older probes. RT shader descriptor accesses use `descriptor_table_addr`.

## Descriptor Records

`descriptor_table_addr` points at an array indexed by set-0 binding number for
the current restricted ABI. Each record is 40 bytes:

| Offset | Field |
| ---: | --- |
| 0 | `uint32_t type` (`VkDescriptorType`) |
| 4 | `uint32_t set` |
| 8 | `uint32_t binding` |
| 12 | `uint32_t format` |
| 16 | `uint64_t device_addr` |
| 24 | `uint64_t byte_size` |
| 32 | `uint32_t stride` |
| 36 | `uint32_t aux` |

Supported bindings for the `raytracingshadows` RT path are:

| Binding | Resource | Record semantics |
| ---: | --- | --- |
| 0 | `accelerationStructureEXT` | `device_addr` is the uploaded TLAS address |
| 1 | `storage image2D` | `device_addr` is pixel memory, `format` is `VkFormat`, `stride` is row bytes |
| 2 | UBO | `device_addr` is uploaded buffer data at descriptor offset |
| 3 | vertices SSBO | `device_addr` is uploaded buffer data at descriptor offset |
| 4 | indices SSBO | `device_addr` is uploaded buffer data at descriptor offset |

Vulkan object addresses may remain host fake addresses until trace time. The
bridge owns conversion to backend physical addresses and patches descriptor
records only after uploading the referenced resource to the selected backend.
Descriptor arrays, dynamic offsets, sampled images, and texel buffers are not
part of this ABI revision.

The continuation fields reserve a compiler/runtime CPS arena carrier. They do
not carry RTCore traversal stacks. A zero `continuation_frame_stride` disables
the arena and requires the address to remain zero.

## Per-Lane Scratch ABI V2

The current per-lane scratch stride is 4480 bytes. The first 4096 bytes are
reserved for compiler-lowered private/local storage; the shader/RTCore-visible
RT ABI region occupies the following 384 bytes. Offsets in the second column
are relative to the RT ABI base (`VT_RT_ABI_PDS_BASE_BYTES`):

| Per-Lane Offset | RT Region Offset | Region |
| ---: | ---: | --- |
| 0..4095 | n/a | compiler private/local reservation |
| 4096..4187 | 0..91 | `ray_state`, 23 words |
| 4188..4191 | 92..95 | alignment gap |
| 4192..4207 | 96..111 | compiler CPS header, 4 words |
| 4208..4231 | 112..135 | `control`, 6 words |
| 4232..4239 | 136..143 | alignment gap |
| 4240..4311 | 144..215 | candidate hit record, 18 words |
| 4312..4319 | 216..223 | candidate hit record extension gap |
| 4320..4391 | 224..295 | committed hit record, 18 words |
| 4392..4399 | 296..303 | committed hit record extension gap |
| 4400..4431 | 304..335 | hit attributes |
| 4432..4479 | 336..383 | default payload region |

Hit record word 18 lives in each hit-record extension gap and stores
`instance_sbt_record_offset`. Existing 18-word fields are not moved.

The four CPS header words are `frame_base`, `active_level`, `fragment_id`, and
`flags`. They reserve the per-lane link from generated shader code to a future
compiler-managed continuation frame. The current compiler does not produce a
nonzero `continuation_frame_size`, so the driver leaves this header and the
dispatch CPS arena address zero in the verified compatibility path.

RT traversal cursor, frontier, and stack state are deliberately absent from
V2 scratch. The Spike functional RTCore model owns that state privately while a
ray is paused for an any-hit or intersection callback. Shader-visible
candidate/committed records and control words remain in this ABI.

`VT_RT_ABI_VERSION` is currently a build-time constant, not a runtime version
field. Mesa, generated kernels, and Spike therefore have to be updated as one
compatibility profile; mixed V1/V2 binaries are not negotiated or rejected by
a runtime version check yet.

## SBT Record Pointer Rules

NIR RT lowering owns Vulkan SBT indexing. LLVM only lowers already-computed
pointers.

Miss record pointer:

```text
miss_sbt_addr + miss_sbt_stride * missIndex + shaderGroupHandleSize
```

Hit record pointer:

```text
hit_sbt_addr
  + hit_sbt_stride * (
      instance_sbt_record_offset
    + geometry_index * traceRay.sbtRecordStride
    + traceRay.sbtRecordOffset)
  + shaderGroupHandleSize
```

RTU/Spike must provide `instance_sbt_record_offset` and `geometry_index` in
the hit record. It must not read Vulkan SBT memory or apply Vulkan SBT byte
stride rules.

## Driver Bridge

The Vulkan object layer can keep host-style fake `VkDeviceAddress` values.
At trace time, `vtvk_driver_bridge` resolves each SBT region and descriptor
resource back to a host object/range, uploads only that data to the selected
backend, and writes backend device addresses into dispatch args and descriptor
records.

Backend selection is controlled by `VENTUS_VK_DRIVER_BACKEND=spike|rtlsim|cyclesim`
or an explicit `VENTUS_VK_DRIVER_LIB` override.
