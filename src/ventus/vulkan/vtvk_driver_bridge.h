/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VTVK_DRIVER_BRIDGE_H
#define VTVK_DRIVER_BRIDGE_H

#include "vtvk_private.h"

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
};

struct vtvk_driver_trace_rays_info {
   struct vtvk_acceleration_structure *tlas;
   void *storage_image_data;
   VkExtent3D image_extent;
   VkFormat image_format;
   void *camera_data;
   VkDeviceSize camera_range;
   const VkStridedDeviceAddressRegionKHR *raygen_sbt;
   const VkStridedDeviceAddressRegionKHR *miss_sbt;
   const VkStridedDeviceAddressRegionKHR *hit_sbt;
   const VkStridedDeviceAddressRegionKHR *callable_sbt;
   uint32_t width;
   uint32_t height;
   uint32_t depth;
   uint32_t per_lane_scratch_size;
   const char *kernel_path;
   uint64_t kernel_entry;
};

bool vtvk_driver_trace_rays(const struct vtvk_driver_trace_rays_info *info);

#ifdef __cplusplus
}
#endif

#endif
