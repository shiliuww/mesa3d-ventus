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

static nir_def *
add_rv32_offset(nir_builder *b, nir_def *base_addr, nir_def *offset)
{
   /* The driver has already proved that the complete allocation fits RV32.
    * Keep dynamic offset arithmetic at 32 bits and widen only for the base add.
    */
   return nir_iadd(b, as_u64(b, base_addr),
                   nir_u2u64(b, nir_u2u32(b, offset)));
}

static nir_def *
rv32_queue_field_stride(nir_builder *b, nir_def *capacity)
{
   /* align(capacity * 4, 128) == ceil(capacity / 32) * 128. */
   return nir_ishl_imm(
      b, nir_ushr_imm(b, nir_iadd_imm(b, capacity, 31), 5), 7);
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
   nir_def *field_offset = nir_imul_imm(
      b, nir_u2u32(b, queue->field_stride_bytes), field);
   nir_def *ray_offset = nir_imul_imm(b, nir_u2u32(b, ray_ref), 4);
   nir_def *offset = nir_iadd(b, field_offset, ray_offset);
   offset = nir_iadd_imm(b, offset, VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES);
   return add_rv32_offset(b, queue->base_addr, offset);
}

nir_def *
vt_nir_rt_submission_mailbox_addr(
   nir_builder *b, nir_def *mailbox_pool_addr, nir_def *mailbox_stride_bytes,
   nir_def *worker_id, nir_def *lane_id,
   enum vt_rt_submission_mailbox_child_kind kind)
{
   nir_def *worker_lane = nir_iadd(
      b, nir_imul_imm(b, nir_u2u32(b, worker_id), VT_RT_GLOBAL_WARP_SIZE),
      nir_u2u32(b, lane_id));
   nir_def *record_index = nir_iadd_imm(
      b, nir_imul_imm(b, worker_lane,
                       VT_RT_SUBMISSION_MAILBOX_RECORD_KINDS), kind);
   return add_rv32_offset(
      b, mailbox_pool_addr,
      nir_imul(b, record_index, nir_u2u32(b, mailbox_stride_bytes)));
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

static nir_def *
mailbox_word_addr(nir_builder *b, nir_def *mailbox_addr, uint32_t word)
{
   return nir_iadd_imm(b, mailbox_addr, word * sizeof(uint32_t));
}

static nir_def *
load_mailbox_word(nir_builder *b, nir_def *mailbox_addr, uint32_t word)
{
   return nir_load_global(b, 1, 32,
                          mailbox_word_addr(b, mailbox_addr, word));
}

static nir_def *
queue_header_word_addr(nir_builder *b, nir_def *queue_addr, uint32_t word)
{
   return nir_iadd_imm(b, queue_addr, word * sizeof(uint32_t));
}

static void
mark_queue_overflow(nir_builder *b, nir_def *queue_addr)
{
   nir_global_atomic(
      b, 32,
      queue_header_word_addr(b, queue_addr,
                             VT_RT_GLOBAL_QUEUE_OVERFLOW_FLAG),
      nir_imm_int(b, 1), .atomic_op = nir_atomic_op_ior);
}

static bool
lower_rt_enqueue(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   (void)data;
   if (intrin->intrinsic != nir_intrinsic_ventus_rt_enqueue)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *mailbox_addr = as_u64(b, intrin->src[0].ssa);
   nir_def *magic = load_mailbox_word(
      b, mailbox_addr, VT_RT_SUBMISSION_MAILBOX_MAGIC);
   nir_def *version = load_mailbox_word(
      b, mailbox_addr, VT_RT_SUBMISSION_MAILBOX_VERSION);
   nir_def *mailbox_generation = load_mailbox_word(
      b, mailbox_addr, VT_RT_SUBMISSION_MAILBOX_GENERATION);
   nir_def *valid = load_mailbox_word(
      b, mailbox_addr, VT_RT_SUBMISSION_MAILBOX_VALID);
   nir_def *queue_lo = load_mailbox_word(
      b, mailbox_addr, VT_RT_SUBMISSION_MAILBOX_QUEUE_BASE_LO);
   nir_def *queue_hi = load_mailbox_word(
      b, mailbox_addr, VT_RT_SUBMISSION_MAILBOX_QUEUE_BASE_HI);
   nir_def *queue_addr = nir_pack_64_2x32_split(b, queue_lo, queue_hi);

   nir_def *header_ok = nir_iand(
      b, nir_ieq_imm(b, magic, VT_RT_SUBMISSION_MAILBOX_MAGIC_VALUE),
      nir_iand(
         b, nir_ieq_imm(b, version, VT_RT_GLOBAL_ABI_VERSION),
         nir_iand(b, nir_ieq_imm(b, valid, 1),
                  nir_ine_imm(b, queue_addr, 0))));

   nir_push_if(b, header_ok);
   nir_def *capacity = nir_load_global(
      b, 1, 32,
      queue_header_word_addr(b, queue_addr,
                             VT_RT_GLOBAL_QUEUE_CAPACITY_RAYS));
   nir_def *queue_generation = nir_load_global(
      b, 1, 32,
      queue_header_word_addr(b, queue_addr,
                             VT_RT_GLOBAL_QUEUE_GENERATION));
   nir_def *generation_ok = nir_iand(
      b, nir_ieq(b, mailbox_generation, queue_generation),
      nir_ine_imm(b, capacity, 0));

   nir_push_if(b, generation_ok);
   nir_def *ray_ref = nir_global_atomic(
      b, 32,
      queue_header_word_addr(b, queue_addr,
                             VT_RT_GLOBAL_QUEUE_RESERVE_TAIL),
      nir_imm_int(b, 1), .atomic_op = nir_atomic_op_iadd);

   nir_push_if(b, nir_ult(b, ray_ref, capacity));
   nir_def *field_stride = rv32_queue_field_stride(b, capacity);
   const struct vt_nir_rt_global_queue_args queue = {
      .base_addr = queue_addr,
      .field_stride_bytes = field_stride,
   };
   for (uint32_t field = 0; field < VT_RT_GLOBAL_TRACE_READY; field++) {
      nir_def *value = nir_load_global(
         b, 1, 32,
         nir_iadd_imm(
            b, mailbox_addr,
            vt_rt_submission_mailbox_field_offset_bytes(
               (enum vt_rt_global_trace_field)field)));
      nir_store_global(
         b, value,
         vt_nir_rt_global_queue_field_addr(
            b, &queue, (enum vt_rt_global_trace_field)field, ray_ref),
         .align_mul = 4);
   }
   nir_global_atomic(
      b, 32,
      vt_nir_rt_global_queue_field_addr(
         b, &queue, VT_RT_GLOBAL_TRACE_READY, ray_ref),
      nir_imm_int(b, 1), .atomic_op = nir_atomic_op_ior);
   nir_push_else(b, NULL);
   mark_queue_overflow(b, queue_addr);
   nir_pop_if(b, NULL);

   nir_push_else(b, NULL);
   mark_queue_overflow(b, queue_addr);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);

   /* The mailbox can be reused only after publication or fail-closed drop. */
   nir_global_atomic(
      b, 32,
      mailbox_word_addr(b, mailbox_addr, VT_RT_SUBMISSION_MAILBOX_VALID),
      nir_imm_int(b, 0), .atomic_op = nir_atomic_op_iand);

   nir_instr_remove(&intrin->instr);
   return true;
}

bool
vt_nir_lower_rt_enqueue(nir_shader *shader)
{
   if (!shader)
      return false;

   const bool progress = nir_shader_intrinsics_pass(
      shader, lower_rt_enqueue, nir_metadata_none, NULL);
   if (progress) {
      nir_opt_dead_cf(shader);
      nir_index_blocks(nir_shader_get_entrypoint(shader));
      nir_index_ssa_defs(nir_shader_get_entrypoint(shader));
   }
   return progress;
}
