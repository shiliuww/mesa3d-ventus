/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VT_RT_ABI_H
#define VT_RT_ABI_H

/*
 * Hardware-visible Ventus RT per-lane scratch ABI.
 *
 * These offsets are intentionally independent from a particular shader. Shader
 * payload contents may change per pipeline, but RTcore-visible ray/control/hit
 * regions must remain stable for a given ABI version.
 */

#define VT_RT_ABI_VERSION 5
#define VT_RT_ABI_WORD_BYTES 4
#define VT_RT_ABI_ALIGNMENT_BYTES 16

enum vt_rt_slot_word {
   VT_RT_SLOT_STATUS = 0,
   VT_RT_SLOT_ACCEL_LO = 1,
   VT_RT_SLOT_ACCEL_HI = 2,
   VT_RT_SLOT_FLAGS = 3,
   VT_RT_SLOT_CULL_MASK = 4,
   VT_RT_SLOT_SBT_OFFSET = 5,
   VT_RT_SLOT_SBT_STRIDE = 6,
   VT_RT_SLOT_MISS_INDEX = 7,
   VT_RT_SLOT_ORIGIN_X = 8,
   VT_RT_SLOT_ORIGIN_Y = 9,
   VT_RT_SLOT_ORIGIN_Z = 10,
   VT_RT_SLOT_TMIN = 11,
   VT_RT_SLOT_DIRECTION_X = 12,
   VT_RT_SLOT_DIRECTION_Y = 13,
   VT_RT_SLOT_DIRECTION_Z = 14,
   VT_RT_SLOT_TMAX = 15,
   VT_RT_SLOT_PAYLOAD_PTR_LO = 16,
   VT_RT_SLOT_PAYLOAD_PTR_HI = 17,
   VT_RT_SLOT_HIT_T = 18,
   VT_RT_SLOT_SBT_INDEX = 19,
   VT_RT_SLOT_LAUNCH_ID_X = 20,
   VT_RT_SLOT_LAUNCH_ID_Y = 21,
   VT_RT_SLOT_LAUNCH_ID_Z = 22,
   VT_RT_SLOT_WORD_COUNT = 23,
};

/* Slot status is a lifecycle handshake, not a traversal result. */
enum vt_rt_slot_status {
   VT_RT_SLOT_STATUS_IDLE = 0,
   VT_RT_SLOT_STATUS_TRACE_REQUEST = 1,
   VT_RT_SLOT_STATUS_TERMINATED = 2,
};

enum vt_rt_traversal_status {
   VT_RT_TRAVERSAL_COMPLETE_MISS = 0,
   VT_RT_TRAVERSAL_COMPLETE_HIT = 1,
   VT_RT_TRAVERSAL_CANDIDATE_NON_OPAQUE_TRIANGLE = 2,
   VT_RT_TRAVERSAL_CANDIDATE_PROCEDURAL_AABB = 3,
   VT_RT_TRAVERSAL_TERMINATED = 4,
};

enum vt_rt_hit_record_word {
   VT_RT_HIT_RECORD_STATUS = 0,
   VT_RT_HIT_RECORD_HIT_T = 1,
   VT_RT_HIT_RECORD_SBT_INDEX = 2,
   VT_RT_HIT_RECORD_SHADER_RECORD_PTR_LO = 3,
   VT_RT_HIT_RECORD_SHADER_RECORD_PTR_HI = 4,
   VT_RT_HIT_RECORD_PRIMITIVE_ID = 5,
   VT_RT_HIT_RECORD_INSTANCE_ID = 6,
   VT_RT_HIT_RECORD_GEOMETRY_ID = 7,
   VT_RT_HIT_RECORD_HIT_KIND = 8,
   VT_RT_HIT_RECORD_BARYCENTRICS_X = 9,
   VT_RT_HIT_RECORD_BARYCENTRICS_Y = 10,
   VT_RT_HIT_RECORD_PRIMITIVE_ADDR_LO = 11,
   VT_RT_HIT_RECORD_PRIMITIVE_ADDR_HI = 12,
   VT_RT_HIT_RECORD_INSTANCE_ADDR_LO = 13,
   VT_RT_HIT_RECORD_INSTANCE_ADDR_HI = 14,
   VT_RT_HIT_RECORD_FRONT_FACE = 15,
   VT_RT_HIT_RECORD_OPAQUE = 16,
   VT_RT_HIT_RECORD_NEED_SOFTWARE_OPACITY_TEST = 17,
   /*
    * Vulkan hit SBT split field. This lives in the 8-byte gap after the
    * 72-byte hit record storage and does not move the following regions.
    */
   VT_RT_HIT_RECORD_INSTANCE_SBT_RECORD_OFFSET = 18,
   VT_RT_HIT_RECORD_WORD_COUNT = 19,
};

enum vt_rt_hit_record_status {
   VT_RT_HIT_RECORD_STATUS_INVALID = 0,
   VT_RT_HIT_RECORD_STATUS_VALID = 1,
};

enum vt_rt_control_word {
   /* The callback writes one final decision before traverse resumes. */
   VT_RT_CONTROL_CALLBACK_DECISION = 0,
   VT_RT_CONTROL_WORD_COUNT = 1,
};

enum vt_rt_callback_decision {
   VT_RT_CALLBACK_PENDING = 0,
   VT_RT_CALLBACK_ACCEPT = 1,
   VT_RT_CALLBACK_IGNORE = 2,
   VT_RT_CALLBACK_TERMINATE = 3,
};

#define VT_RT_ABI_RAY_STATE_BASE_BYTES 0
#define VT_RT_ABI_RAY_STATE_SIZE_BYTES \
   (VT_RT_SLOT_WORD_COUNT * VT_RT_ABI_WORD_BYTES)

/* RTcore-private traversal state is not part of this ABI. */
#define VT_RT_ABI_CPS_HEADER_BASE_BYTES 96
#define VT_RT_ABI_CPS_HEADER_SIZE_BYTES 16
enum vt_rt_cps_header_word {
   VT_RT_CPS_FRAME_BASE = 0,
   VT_RT_CPS_ACTIVE_LEVEL = 1,
   VT_RT_CPS_FRAGMENT_ID = 2,
   VT_RT_CPS_FLAGS = 3,
   VT_RT_CPS_HEADER_WORD_COUNT = 4,
};

#define VT_RT_ABI_CONTROL_BASE_BYTES 112
#define VT_RT_ABI_CONTROL_SIZE_BYTES \
   (VT_RT_CONTROL_WORD_COUNT * VT_RT_ABI_WORD_BYTES)

#define VT_RT_ABI_CANDIDATE_HIT_RECORD_BASE_BYTES 116
#define VT_RT_ABI_CANDIDATE_HIT_RECORD_SIZE_BYTES 72

#define VT_RT_ABI_COMMITTED_HIT_RECORD_BASE_BYTES 196
#define VT_RT_ABI_COMMITTED_HIT_RECORD_SIZE_BYTES 72

#define VT_RT_ABI_HIT_ATTRIB_BASE_BYTES 276
/* The payload starts after candidate and committed non-barycentric hit
 * attributes.  Intersection/any-hit write the candidate half; an accepted
 * report snapshots it into the committed half before closest-hit runs.
 * The two triangle barycentric words live in hit-record words 9 and 10 and
 * are therefore not part of this variable-sized region. */
#define VT_RT_ABI_HIT_ATTRIB_SIZE_BYTES 0

#define VT_RT_ABI_SHADER_REGION_BASE_BYTES VT_RT_ABI_HIT_ATTRIB_BASE_BYTES
#define VT_RT_ABI_PAYLOAD_BASE_BYTES VT_RT_ABI_SHADER_REGION_BASE_BYTES
/* The linked RT pipeline determines P before lowering:
 * ALIGN16(max(location * 16 + natural_size)). */
#define VT_RT_ABI_DEFAULT_PAYLOAD_SIZE_BYTES 0
#define VT_RT_ABI_REGION_SIZE_BYTES VT_RT_ABI_SHADER_REGION_BASE_BYTES
#define VT_RT_ABI_FIXED_HEADER_SIZE_BYTES VT_RT_ABI_SHADER_REGION_BASE_BYTES

#endif
