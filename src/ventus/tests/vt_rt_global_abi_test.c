/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>

#include "ventus/compiler/vt_nir.h"

#define CHECK(condition) do { \
   if (!(condition)) { \
      fprintf(stderr, "CHECK failed: %s\n", #condition); \
      return 1; \
   } \
} while (0)

int
main(void)
{
   const struct vt_rt_global_queue_layout layout =
      vt_nir_rt_global_queue_layout(65);

   CHECK(VT_RT_GLOBAL_ABI_VERSION == 3);
   CHECK(layout.header_size_bytes == VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES);
   CHECK(layout.field_stride_bytes == 384);
   CHECK(layout.fields_base_bytes == VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES);
   CHECK(layout.total_size_bytes == VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES +
         VT_RT_GLOBAL_TRACE_FIELD_COUNT * 384);
   CHECK(vt_rt_global_field_offset_bytes(&layout,
                                         VT_RT_GLOBAL_TRACE_ORIGIN_X, 0) ==
         VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES + VT_RT_GLOBAL_TRACE_ORIGIN_X * 384);
   CHECK(vt_rt_global_field_offset_bytes(&layout,
                                         VT_RT_GLOBAL_TRACE_ORIGIN_X, 31) ==
         VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES + VT_RT_GLOBAL_TRACE_ORIGIN_X * 384 + 124);
   CHECK(vt_rt_global_field_offset_bytes(&layout,
                                         VT_RT_GLOBAL_TRACE_ORIGIN_X, 32) ==
         VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES + VT_RT_GLOBAL_TRACE_ORIGIN_X * 384 + 128);
   CHECK(vt_rt_global_field_offset_bytes(&layout,
                                         VT_RT_GLOBAL_TRACE_TMAX, 64) ==
         VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES + VT_RT_GLOBAL_TRACE_TMAX * 384 + 256);

   printf("PASS vt_rt_global_abi capacity=%u stride=%u total=%u\n",
          layout.capacity_rays, layout.field_stride_bytes,
          layout.total_size_bytes);
   return 0;
}
