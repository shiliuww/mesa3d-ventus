/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

static nir_def *
as_u64(nir_builder *b, nir_def *value)
{
   return value->bit_size == 64 ? value : nir_u2u64(b, value);
}

static void
store_scalar_mailbox_field(nir_builder *b,
                   const struct vt_nir_rt_submission_mailbox_args *mailbox,
                   enum vt_rt_global_trace_field field,
                   nir_def *value)
{
   if (!value)
      value = nir_imm_int(b, 0);
   nir_def *addr = nir_iadd_imm(b, mailbox->base_addr,
      vt_rt_submission_mailbox_field_offset_bytes(field));
   nir_store_global(b, value, addr, .align_mul = 4);
}

static void
store_vec3_mailbox_fields(nir_builder *b,
                  const struct vt_nir_rt_submission_mailbox_args *mailbox,
                  enum vt_rt_global_trace_field first_field,
                  nir_def *value)
{
   for (unsigned i = 0; i < 3; ++i)
      store_scalar_mailbox_field(b, mailbox, first_field + i,
                                 value ? nir_channel(b, value, i) : NULL);
}

nir_def *
vt_nir_rt_global_queue_field_addr(
   nir_builder *b, const struct vt_nir_rt_global_queue_args *queue,
   enum vt_rt_global_trace_field field, nir_def *ray_ref)
{
   nir_def *field_offset = nir_imul_imm(b, as_u64(b, queue->field_stride_bytes),
                                        field);
   nir_def *ray_offset = nir_imul_imm(b, as_u64(b, ray_ref), 4);
   nir_def *offset = nir_iadd(b, field_offset, ray_offset);
   offset = nir_iadd_imm(b, offset, VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES);
   return nir_iadd(b, as_u64(b, queue->base_addr), offset);
}

nir_def *
vt_nir_rt_submission_mailbox_addr(
   nir_builder *b, nir_def *mailbox_pool_addr, nir_def *mailbox_stride_bytes,
   nir_def *worker_id, nir_def *lane_id,
   enum vt_rt_submission_mailbox_child_kind kind)
{
   nir_def *worker_lane = nir_iadd(
      b, nir_imul_imm(b, nir_u2u64(b, worker_id), VT_RT_GLOBAL_WARP_SIZE),
      nir_u2u64(b, lane_id));
   nir_def *record_index = nir_iadd_imm(
      b, nir_imul_imm(b, worker_lane,
                       VT_RT_SUBMISSION_MAILBOX_RECORD_KINDS), kind);
   return nir_iadd(b, mailbox_pool_addr,
                   nir_imul(b, record_index,
                            nir_u2u64(b, mailbox_stride_bytes)));
}

void
vt_nir_rt_store_submission_mailbox(
   nir_builder *b, const struct vt_nir_rt_submission_mailbox_args *mailbox,
   const struct vt_nir_rt_trace_params *params, nir_def *parent_frame,
   nir_def *cps_frame, nir_def *continuation_id, nir_def *cps_stack_size,
   nir_def *launch_id, nir_def *phase_generation, nir_def *depth)
{
   /* Prepare all record data before publishing the mailbox valid word. */
   nir_def *tlas = params->accel_struct ? nir_u2u64(b, params->accel_struct) :
      nir_imm_int64(b, 0);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_TLAS_ADDR_LO,
                              nir_unpack_64_2x32_split_x(b, tlas));
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_TLAS_ADDR_HI,
                              nir_unpack_64_2x32_split_y(b, tlas));
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_FLAGS,
                      params->flags);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_CULL_MASK,
                      params->cull_mask);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_SBT_OFFSET,
                      params->sbt_offset);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_SBT_STRIDE,
                      params->sbt_stride);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_MISS_INDEX,
                      params->miss_index);
   store_vec3_mailbox_fields(b, mailbox, VT_RT_GLOBAL_TRACE_ORIGIN_X,
                     params->origin);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_TMIN,
                      params->tmin);
   store_vec3_mailbox_fields(b, mailbox, VT_RT_GLOBAL_TRACE_DIRECTION_X,
                     params->direction);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_TMAX,
                      params->tmax);
   nir_def *payload = params->payload;
   if (payload && payload->bit_size == 64) {
      store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_PAYLOAD_LO,
                                 nir_unpack_64_2x32_split_x(b, payload));
      store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_PAYLOAD_HI,
                                 nir_unpack_64_2x32_split_y(b, payload));
   } else {
      store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_PAYLOAD_LO,
                                 payload);
      store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_PAYLOAD_HI,
                                 nir_imm_int(b, 0));
   }
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_CPS_FRAME,
                      cps_frame);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_PARENT_FRAME,
                      parent_frame);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_CONTINUATION_ID,
                      continuation_id);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_CPS_STACK_SIZE,
                      cps_stack_size);
   store_vec3_mailbox_fields(b, mailbox, VT_RT_GLOBAL_TRACE_LAUNCH_ID_X,
                             launch_id);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_PHASE_GENERATION,
                      phase_generation);
   store_scalar_mailbox_field(b, mailbox, VT_RT_GLOBAL_TRACE_DEPTH, depth);

   nir_store_global(b, nir_imm_int(b, VT_RT_SUBMISSION_MAILBOX_MAGIC_VALUE),
      mailbox->base_addr, .align_mul = 4);
   nir_store_global(b, nir_imm_int(b, VT_RT_GLOBAL_ABI_VERSION),
      nir_iadd_imm(b, mailbox->base_addr, 4), .align_mul = 4);
   nir_store_global(b, mailbox->generation,
      nir_iadd_imm(b, mailbox->base_addr, 8), .align_mul = 4);
   nir_def *queue_base = mailbox->queue_base_addr ?
      as_u64(b, mailbox->queue_base_addr) : nir_imm_int64(b, 0);
   nir_store_global(b, nir_unpack_64_2x32_split_x(b, queue_base),
      nir_iadd_imm(b, mailbox->base_addr, 16), .align_mul = 4);
   nir_store_global(b, nir_unpack_64_2x32_split_y(b, queue_base),
      nir_iadd_imm(b, mailbox->base_addr, 20), .align_mul = 4);
   nir_store_global(b, nir_imm_int(b, 1),
      nir_iadd_imm(b, mailbox->base_addr, 12), .align_mul = 4);
}
