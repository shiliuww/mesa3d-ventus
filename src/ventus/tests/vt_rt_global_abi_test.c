/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <inttypes.h>

#include "ventus/common/ventus_rt_address.h"
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
   uint32_t capacity = 0;

   CHECK(VT_RT_GLOBAL_ABI_VERSION == 6);
   CHECK(vt_nir_rt_global_queue_capacity(160, 96, 2, 3, &capacity));
   CHECK(capacity == 160 * 96 * 2 * 3);
   uint64_t mailbox_records = 0;
   CHECK(vt_nir_rt_submission_mailbox_capacity(2, 32, &mailbox_records));
   CHECK(mailbox_records == 2ull * 32 * 4);
   CHECK(vt_rt_submission_mailbox_record_index(
            0, 0, VT_RT_SUBMISSION_MAILBOX_PRIMARY) == 0);
   CHECK(vt_rt_submission_mailbox_record_index(
            0, 31, VT_RT_SUBMISSION_MAILBOX_CHILD_REFLECTION) ==
         31ull * 4 + 2);
   CHECK(vt_rt_submission_mailbox_record_index(
            1, 31, VT_RT_SUBMISSION_MAILBOX_CHILD_REFRACTION) ==
         mailbox_records - 1);
   CHECK(!vt_nir_rt_submission_mailbox_capacity(0, 32, &mailbox_records));
   CHECK(!vt_nir_rt_global_queue_capacity(UINT32_MAX, 2, 2, 3, &capacity));
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
   CHECK(vt_nir_rt_global_queue_layout(UINT32_MAX).total_size_bytes > UINT32_MAX);
   CHECK(ventus_rt_device_range_fits_u32(0, 1));
   CHECK(ventus_rt_device_range_fits_u32(0xfffffff0ull, 16));
   CHECK(!ventus_rt_device_range_fits_u32(0xfffffff0ull, 17));
   CHECK(!ventus_rt_device_range_fits_u32(0x100000000ull, 1));
   CHECK(!ventus_rt_device_range_fits_u32(UINT64_MAX, 2));

   printf("PASS vt_rt_global_abi capacity=%u stride=%" PRIu64 " total=%" PRIu64 "\n",
          layout.capacity_rays, layout.field_stride_bytes,
          layout.total_size_bytes);
   return 0;
}
