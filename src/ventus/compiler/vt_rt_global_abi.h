/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VT_RT_GLOBAL_ABI_H
#define VT_RT_GLOBAL_ABI_H

#include <stdint.h>

/*
 * Shader-visible wavefront RT ABI.  A ray reference is an ordinal in a
 * field-major global queue, never a PDS address or a physical lane number.
 */
#define VT_RT_GLOBAL_ABI_VERSION 6
#define VT_RT_GLOBAL_WARP_SIZE 32u
#define VT_RT_GLOBAL_ALIGNMENT_BYTES 128u
#define VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES VT_RT_GLOBAL_ALIGNMENT_BYTES
#define VT_RT_SUBMISSION_MAILBOX_HEADER_SIZE_BYTES 32u
/*
 * A submission mailbox is transient staging storage, not per-ray state.  It
 * is owned by one physical worker lane until RTcore accepts the record and
 * clears VALID.  The pool therefore scales with resident workers, rather
 * than with launch size or global queue capacity.
 */
#define VT_RT_SUBMISSION_MAILBOX_RECORD_KINDS 4u

enum vt_rt_global_queue_header_word {
   VT_RT_GLOBAL_QUEUE_CAPACITY_RAYS = 0,
   /* RTcore is the only owner of all queue cursors. */
   VT_RT_GLOBAL_QUEUE_RESERVE_TAIL = 1,
   VT_RT_GLOBAL_QUEUE_CONSUME_HEAD = 2,
   VT_RT_GLOBAL_QUEUE_OVERFLOW_FLAG = 3,
   VT_RT_GLOBAL_QUEUE_GENERATION = 4,
   VT_RT_GLOBAL_QUEUE_HEADER_WORD_COUNT = 5,
};

enum vt_rt_global_trace_field {
   /* Resolved RV64 Ventus TLAS device address, never a Vulkan handle. */
   VT_RT_GLOBAL_TRACE_TLAS_ADDR_LO = 0,
   VT_RT_GLOBAL_TRACE_TLAS_ADDR_HI,
   VT_RT_GLOBAL_TRACE_FLAGS,
   VT_RT_GLOBAL_TRACE_CULL_MASK,
   VT_RT_GLOBAL_TRACE_SBT_OFFSET,
   VT_RT_GLOBAL_TRACE_SBT_STRIDE,
   VT_RT_GLOBAL_TRACE_MISS_INDEX,
   VT_RT_GLOBAL_TRACE_ORIGIN_X,
   VT_RT_GLOBAL_TRACE_ORIGIN_Y,
   VT_RT_GLOBAL_TRACE_ORIGIN_Z,
   VT_RT_GLOBAL_TRACE_TMIN,
   VT_RT_GLOBAL_TRACE_DIRECTION_X,
   VT_RT_GLOBAL_TRACE_DIRECTION_Y,
   VT_RT_GLOBAL_TRACE_DIRECTION_Z,
   VT_RT_GLOBAL_TRACE_TMAX,
   VT_RT_GLOBAL_TRACE_PAYLOAD_LO,
   VT_RT_GLOBAL_TRACE_PAYLOAD_HI,
   VT_RT_GLOBAL_TRACE_CHILD_CPS_FRAME,
   VT_RT_GLOBAL_TRACE_PARENT_FRAME,
   VT_RT_GLOBAL_TRACE_PHASE_GENERATION,
   VT_RT_GLOBAL_TRACE_DEPTH,
   /* Written last by RTcore after every record field and CPS blob is valid. */
   VT_RT_GLOBAL_TRACE_READY,
   VT_RT_GLOBAL_TRACE_FIELD_COUNT,
};

enum vt_rt_submission_mailbox_header_word {
   VT_RT_SUBMISSION_MAILBOX_MAGIC = 0,
   VT_RT_SUBMISSION_MAILBOX_VERSION = 1,
   VT_RT_SUBMISSION_MAILBOX_GENERATION = 2,
   /* Written last by the shader; consumed and cleared by RTcore. */
   VT_RT_SUBMISSION_MAILBOX_VALID = 3,
   /* Global queue identity for the RTcore enqueue consumer. */
   VT_RT_SUBMISSION_MAILBOX_QUEUE_BASE_LO = 4,
   VT_RT_SUBMISSION_MAILBOX_QUEUE_BASE_HI = 5,
};

enum vt_rt_submission_mailbox_child_kind {
   VT_RT_SUBMISSION_MAILBOX_PRIMARY = 0,
   VT_RT_SUBMISSION_MAILBOX_CHILD_SHADOW,
   VT_RT_SUBMISSION_MAILBOX_CHILD_REFLECTION,
   VT_RT_SUBMISSION_MAILBOX_CHILD_REFRACTION,
   VT_RT_SUBMISSION_MAILBOX_CHILD_KIND_COUNT =
      VT_RT_SUBMISSION_MAILBOX_RECORD_KINDS,
};

#define VT_RT_SUBMISSION_MAILBOX_MAGIC_VALUE 0x56545251u /* "VTRQ" */

/* worker_id denotes one resident physical warp, globally unique across SMs.
 * lane_id is the lane within that warp.  Neither input is a logical ray_ref. */
static inline uint64_t
vt_rt_submission_mailbox_record_index(uint32_t worker_id, uint32_t lane_id,
                                             enum vt_rt_submission_mailbox_child_kind kind)
{
   return ((uint64_t)worker_id * VT_RT_GLOBAL_WARP_SIZE + lane_id) *
             VT_RT_SUBMISSION_MAILBOX_RECORD_KINDS +
          kind;
}

struct vt_rt_global_queue_layout {
   uint32_t capacity_rays;
   uint64_t header_size_bytes;
   uint64_t field_stride_bytes;
   uint64_t fields_base_bytes;
   uint64_t total_size_bytes;
};

static inline uint64_t
vt_rt_submission_mailbox_field_offset_bytes(enum vt_rt_global_trace_field field)
{
   return VT_RT_SUBMISSION_MAILBOX_HEADER_SIZE_BYTES + (uint64_t)field * 4u;
}

static inline uint32_t
vt_rt_global_align_u32(uint32_t value, uint32_t alignment)
{
   return (value + alignment - 1u) & ~(alignment - 1u);
}

static inline uint64_t
vt_rt_global_field_offset_bytes(const struct vt_rt_global_queue_layout *layout,
                                enum vt_rt_global_trace_field field,
                                uint32_t ray_ref)
{
   return layout->fields_base_bytes +
          (uint64_t)field * layout->field_stride_bytes + (uint64_t)ray_ref * 4u;
}

#endif
