/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VTVK_DRIVER_BRIDGE_H
#define VTVK_DRIVER_BRIDGE_H

#include "vtvk_private.h"
#include <ventus/vulkan/ventus_rt_profile.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vtvk_rt_sbt_region {
   uint64_t addr;
   uint64_t size;
   uint64_t stride;
};

struct vtvk_rt_dispatch_args {
   uint64_t tlas_addr;
   uint64_t image_addr;
   uint64_t camera_addr;
   struct vtvk_rt_sbt_region raygen_sbt;
   struct vtvk_rt_sbt_region miss_sbt;
   struct vtvk_rt_sbt_region hit_sbt;
   struct vtvk_rt_sbt_region callable_sbt;
   uint32_t width;
   uint32_t height;
   uint32_t depth;
   uint32_t image_format;
   uint32_t image_stride;
   uint32_t camera_size;
   uint32_t scratch_stride;
   uint32_t reserved;
   uint64_t descriptor_table_addr;
   uint32_t descriptor_count;
   uint32_t descriptor_record_stride;
   uint64_t continuation_arena_addr;
   uint32_t continuation_frame_stride;
   uint32_t continuation_context_count;
   uint64_t submission_mailbox_base;
   uint32_t submission_mailbox_stride;
   uint32_t submission_mailbox_generation;
   uint64_t global_queue_base;
   uint32_t global_queue_capacity;
   uint32_t global_queue_generation;
   uint64_t global_cps_arena_addr;
   uint32_t global_cps_frame_stride;
   uint32_t max_ray_recursion_depth;
   uint64_t global_completion_base;
   uint32_t global_completion_capacity;
   uint32_t global_completion_generation;
   uint64_t global_ray_ref_list_base;
   uint32_t global_ray_ref_count;
   uint32_t global_ray_ref_generation;
   uint32_t global_completion_field_stride;
   uint32_t reserved_global;
   uint64_t push_constants_addr;
   uint32_t push_constants_size;
   uint32_t reserved_push_constants;
};

struct vtvk_rt_descriptor_record {
   uint32_t type;
   uint32_t set;
   uint32_t binding;
   uint32_t format;
   uint64_t device_addr;
   uint64_t byte_size;
   uint32_t stride;
   uint32_t aux;
   uint32_t sampler_mag_filter;
   uint32_t sampler_min_filter;
   uint32_t sampler_address_mode_u;
   uint32_t sampler_address_mode_v;
};

struct vtvk_rt_descriptor_source {
   VkDescriptorType type;
   uint32_t set;
   uint32_t binding;
   VkAccelerationStructureKHR acceleration_structure;
   VkImage image;
   VkSampler sampler;
   VkBuffer buffer;
   VkDeviceSize offset;
   VkDeviceSize range;
};

struct vtvk_driver_trace_rays_info {
   struct vtvk_acceleration_structure *tlas;
   void *storage_image_data;
   VkExtent3D image_extent;
   VkFormat image_format;
   void *camera_data;
   VkDeviceSize camera_range;
   const void *push_constants;
   uint32_t push_constants_size;
   const VkStridedDeviceAddressRegionKHR *raygen_sbt;
   const VkStridedDeviceAddressRegionKHR *miss_sbt;
   const VkStridedDeviceAddressRegionKHR *hit_sbt;
   const VkStridedDeviceAddressRegionKHR *callable_sbt;
   uint32_t width;
   uint32_t height;
   uint32_t depth;
   uint32_t per_lane_scratch_size;
   uint32_t hit_attribute_size_bytes;
   uint32_t continuation_frame_size;
   uint32_t max_ray_recursion_depth;
   VkVentusRtExecutionProfile execution_profile;
   VkVentusRtStageRole stage_role;
   uint32_t state_descriptor_set;
   uint32_t state_descriptor_binding;
   const char *kernel_path;
   uint64_t kernel_entry;
   const char *const *resume_kernel_paths;
   const uint64_t *resume_kernel_entries;
   uint32_t resume_kernel_count;
   const char *const *callback_kernel_paths;
   const uint64_t *callback_kernel_entries;
   const uint32_t *callback_kernel_groups;
   const uint32_t *callback_kernel_stages;
   const uint32_t *callback_kernel_lds_sizes;
   uint32_t callback_kernel_count;
   const struct vtvk_rt_descriptor_source *descriptors;
   uint32_t descriptor_count;
};

bool vtvk_driver_trace_rays(const struct vtvk_driver_trace_rays_info *info);

#ifdef __cplusplus
}
#endif

#endif
