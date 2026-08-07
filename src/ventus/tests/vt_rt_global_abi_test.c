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
   struct vt_nir_rt_config legacy_config = vt_nir_rt_default_config();
   struct vt_nir_rt_config global_config = vt_nir_rt_default_config();
   const struct vt_nir_rt_abi_layout abi = vt_nir_rt_default_abi_layout();
   uint32_t capacity = 0;

   vt_nir_rt_config_apply_global_cps_layout(&global_config, &abi);
   CHECK(legacy_config.scratchpad[VT_RT_LAYOUT_RAYGEN].base_bytes ==
         VT_RT_ABI_PDS_BASE_BYTES);
   CHECK(global_config.scratchpad[VT_RT_LAYOUT_RAYGEN].base_bytes == 0);
   CHECK(global_config.scratchpad[VT_RT_LAYOUT_RAYGEN].payload_base_bytes ==
         VT_RT_ABI_PAYLOAD_BASE_BYTES);
   CHECK(global_config.scratchpad[VT_RT_LAYOUT_RAYGEN].total_size_bytes ==
         VT_RT_ABI_REGION_SIZE_BYTES);
   CHECK(global_config.scratchpad[VT_RT_LAYOUT_RAYGEN].stride_bytes ==
         VT_RT_ABI_REGION_SIZE_BYTES);

   CHECK(VT_RT_GLOBAL_ABI_VERSION == 10);
   CHECK(VT_RT_SUBMISSION_MAILBOX_SIZE_BYTES ==
         VT_RT_SUBMISSION_MAILBOX_HEADER_SIZE_BYTES +
            VT_RT_GLOBAL_TRACE_FIELD_COUNT * 4u);
   CHECK(VT_RT_SUBMISSION_MAILBOX_SIZE_BYTES == 140);
   CHECK(vt_nir_rt_global_queue_capacity(160, 96, 1, 2, 3, &capacity));
   CHECK(capacity == 160 * 96 * 2 * 3);
   CHECK(vt_nir_rt_global_queue_capacity(4, 3, 2, 2, 3, &capacity));
   CHECK(capacity == 4 * 3 * 2 * 2 * 3);
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
   CHECK(!vt_nir_rt_global_queue_capacity(1, 1, 0, 2, 3, &capacity));
   CHECK(!vt_nir_rt_global_queue_capacity(1, 1, 1, 2, 3, NULL));
   CHECK(!vt_nir_rt_global_queue_capacity(UINT32_MAX, UINT32_MAX,
                                          UINT32_MAX, UINT32_MAX,
                                          UINT32_MAX, &capacity));
   CHECK(layout.header_size_bytes == VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES);
   CHECK(layout.field_stride_bytes == 384);
   CHECK(vt_nir_rt_global_queue_layout(768).field_stride_bytes == 3072);
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
   const struct vt_rt_global_completion_layout completion =
      vt_nir_rt_global_completion_layout(65);
   CHECK(completion.header_size_bytes == VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES);
   CHECK(completion.field_stride_bytes == 384);
   CHECK(completion.total_size_bytes == VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES +
         VT_RT_GLOBAL_COMPLETION_FIELD_COUNT * 384);
   CHECK(vt_rt_global_completion_field_offset_bytes(
            &completion, VT_RT_GLOBAL_COMPLETION_CPS_FRAME, 31) ==
         VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES +
         VT_RT_GLOBAL_COMPLETION_CPS_FRAME * 384 + 124);
   CHECK(VT_RT_GLOBAL_COMPLETION_CALLBACK_GROUP ==
         VT_RT_GLOBAL_COMPLETION_LAUNCH_ID_Z + 1);
   CHECK(VT_RT_GLOBAL_COMPLETION_CANDIDATE_HIT_BASE >
         VT_RT_GLOBAL_COMPLETION_READY);
   CHECK(VT_RT_GLOBAL_COMPLETION_CANDIDATE_HIT_LAST -
         VT_RT_GLOBAL_COMPLETION_CANDIDATE_HIT_BASE + 1 ==
         VT_RT_HIT_RECORD_WORD_COUNT);
   CHECK(VT_RT_GLOBAL_COMPLETION_CANDIDATE_CONTROL_LAST -
         VT_RT_GLOBAL_COMPLETION_CANDIDATE_CONTROL_BASE + 1 ==
         VT_RT_GLOBAL_CANDIDATE_CONTROL_WORD_COUNT);
   CHECK(VT_RT_CONTROL_WORD_COUNT == 1);
   CHECK(vt_rt_global_completion_field_offset_bytes(
            &completion, VT_RT_GLOBAL_COMPLETION_CANDIDATE_HIT_BASE, 31) !=
         vt_rt_global_completion_field_offset_bytes(
            &completion, VT_RT_GLOBAL_COMPLETION_COMMITTED_HIT_BASE, 31));
   CHECK(VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_BASE == 232);
   CHECK(VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_CAPACITY == 240);
   CHECK(VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_GENERATION == 244);
   CHECK(VT_NIR_RT_DISPATCH_ARG_GLOBAL_RAY_REF_LIST_BASE == 248);
   CHECK(VT_NIR_RT_DISPATCH_ARG_GLOBAL_RAY_REF_COUNT == 256);
   CHECK(VT_NIR_RT_DISPATCH_ARG_GLOBAL_RAY_REF_GENERATION == 260);
   CHECK(VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_FIELD_STRIDE == 264);
   CHECK(VT_NIR_RT_DISPATCH_ARG_PUSH_CONSTANTS_ADDR == 272);
   CHECK(VT_NIR_RT_DISPATCH_ARG_PUSH_CONSTANTS_SIZE == 280);
   CHECK(VT_NIR_RT_DISPATCH_ARG_SIZE == 288);
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
