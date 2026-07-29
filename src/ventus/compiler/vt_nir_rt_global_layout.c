/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

bool
vt_nir_rt_global_queue_capacity(uint32_t width, uint32_t height,
                                uint32_t depth,
                                uint32_t max_ray_recursion_depth,
                                uint32_t children_per_ray,
                                uint32_t *capacity_rays)
{
   if (!width || !height || !depth || !max_ray_recursion_depth ||
       !children_per_ray || !capacity_rays)
      return false;

   uint64_t capacity = width;
   if (__builtin_mul_overflow(capacity, (uint64_t)height, &capacity) ||
       __builtin_mul_overflow(capacity, (uint64_t)depth, &capacity) ||
       __builtin_mul_overflow(capacity, (uint64_t)max_ray_recursion_depth,
                              &capacity) ||
       __builtin_mul_overflow(capacity, (uint64_t)children_per_ray,
                              &capacity) ||
       capacity > UINT32_MAX)
      return false;

   *capacity_rays = (uint32_t)capacity;
   return true;
}

bool
vt_nir_rt_submission_mailbox_capacity(uint32_t resident_workers,
                                       uint32_t lanes_per_worker,
                                       uint64_t *mailbox_records)
{
   if (!resident_workers || !lanes_per_worker || !mailbox_records)
      return false;

   const uint64_t worker_lanes =
      (uint64_t)resident_workers * lanes_per_worker;
   if (worker_lanes > UINT64_MAX / VT_RT_SUBMISSION_MAILBOX_RECORD_KINDS)
      return false;

   *mailbox_records =
      worker_lanes * VT_RT_SUBMISSION_MAILBOX_RECORD_KINDS;
   return true;
}

struct vt_rt_global_queue_layout
vt_nir_rt_global_queue_layout(uint32_t capacity_rays)
{
   const uint64_t unaligned_stride = (uint64_t)capacity_rays * 4u;
   const uint64_t field_stride =
      (unaligned_stride + VT_RT_GLOBAL_ALIGNMENT_BYTES - 1u) &
      ~(uint64_t)(VT_RT_GLOBAL_ALIGNMENT_BYTES - 1u);
   if (field_stride > UINT64_MAX / VT_RT_GLOBAL_TRACE_FIELD_COUNT)
      return (struct vt_rt_global_queue_layout){};
   const uint64_t fields_size =
      (uint64_t)VT_RT_GLOBAL_TRACE_FIELD_COUNT * field_stride;
   if (VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES > UINT64_MAX - fields_size)
      return (struct vt_rt_global_queue_layout){};
   const uint64_t total_size = VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES + fields_size;

   struct vt_rt_global_queue_layout layout = {
      .capacity_rays = capacity_rays,
      .header_size_bytes = VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES,
      .field_stride_bytes = field_stride,
   };

   layout.fields_base_bytes = layout.header_size_bytes;
   layout.total_size_bytes = total_size;
   return layout;
}

struct vt_rt_global_completion_layout
vt_nir_rt_global_completion_layout(uint32_t capacity_rays)
{
   const uint64_t unaligned_stride = (uint64_t)capacity_rays * 4u;
   const uint64_t field_stride =
      (unaligned_stride + VT_RT_GLOBAL_ALIGNMENT_BYTES - 1u) &
      ~(uint64_t)(VT_RT_GLOBAL_ALIGNMENT_BYTES - 1u);
   if (field_stride > UINT64_MAX / VT_RT_GLOBAL_COMPLETION_FIELD_COUNT)
      return (struct vt_rt_global_completion_layout){};
   const uint64_t fields_size =
      (uint64_t)VT_RT_GLOBAL_COMPLETION_FIELD_COUNT * field_stride;
   if (VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES > UINT64_MAX - fields_size)
      return (struct vt_rt_global_completion_layout){};

   return (struct vt_rt_global_completion_layout){
      .capacity_rays = capacity_rays,
      .header_size_bytes = VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES,
      .field_stride_bytes = field_stride,
      .fields_base_bytes = VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES,
      .total_size_bytes = VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES + fields_size,
   };
}
