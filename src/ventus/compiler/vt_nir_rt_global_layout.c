/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

struct vt_rt_global_queue_layout
vt_nir_rt_global_queue_layout(uint32_t capacity_rays)
{
   struct vt_rt_global_queue_layout layout = {
      .capacity_rays = capacity_rays,
      .header_size_bytes = VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES,
      .field_stride_bytes = vt_rt_global_align_u32(capacity_rays * 4u,
                                                    VT_RT_GLOBAL_ALIGNMENT_BYTES),
   };

   layout.fields_base_bytes = layout.header_size_bytes;
   layout.total_size_bytes = layout.fields_base_bytes +
      VT_RT_GLOBAL_TRACE_FIELD_COUNT * layout.field_stride_bytes;
   return layout;
}
