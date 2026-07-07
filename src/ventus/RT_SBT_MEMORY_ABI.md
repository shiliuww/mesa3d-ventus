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

## Per-Lane Scratch

The stable per-lane scratch layout remains 352 bytes:

| Byte Range | Region |
| ---: | --- |
| 0..79 | `ray_state`, 20 words |
| 80..103 | `control`, 6 words |
| 104..111 | alignment gap |
| 112..183 | candidate hit record, 18 words |
| 184..191 | candidate hit record extension gap |
| 192..263 | committed hit record, 18 words |
| 264..271 | committed hit record extension gap |
| 272..303 | hit attributes |
| 304..347 | payload |
| 348..351 | alignment |

Hit record word 18 lives in the extension gap and stores
`instance_sbt_record_offset`. Existing fields are not moved.

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
