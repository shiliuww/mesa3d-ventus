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

#define VT_RT_ABI_VERSION 9
#define VT_RT_ABI_WORD_BYTES 4
#define VT_RT_ABI_ALIGNMENT_BYTES 16

enum vt_rt_slot_word {
   VT_RT_SLOT_STATUS = 0,
   VT_RT_SLOT_ACCEL_LO = 1,
   VT_RT_SLOT_ACCEL_HI = 2,
   VT_RT_SLOT_TRACE_META0 = 3,
   VT_RT_SLOT_TRACE_META1 = 4,
   VT_RT_SLOT_ORIGIN_X = 5,
   VT_RT_SLOT_ORIGIN_Y = 6,
   VT_RT_SLOT_ORIGIN_Z = 7,
   VT_RT_SLOT_TMIN = 8,
   VT_RT_SLOT_DIRECTION_X = 9,
   VT_RT_SLOT_DIRECTION_Y = 10,
   VT_RT_SLOT_DIRECTION_Z = 11,
   VT_RT_SLOT_TMAX = 12,
   /* Retained for the legacy wavefront mailbox path. */
   VT_RT_SLOT_PAYLOAD_PTR_LO = 13,
   VT_RT_SLOT_PAYLOAD_PTR_HI = 14,
   VT_RT_SLOT_LAUNCH_ID_X = 15,
   VT_RT_SLOT_LAUNCH_ID_Y = 16,
   VT_RT_SLOT_LAUNCH_ID_Z = 17,
   VT_RT_SLOT_WORD_COUNT = 18,
};

/* trace_meta0 packs bounded TraceRay inputs. */
#define VT_RT_TRACE_META0_FLAGS_SHIFT 0
#define VT_RT_TRACE_META0_FLAGS_BITS 8
#define VT_RT_TRACE_META0_CULL_MASK_SHIFT 8
#define VT_RT_TRACE_META0_CULL_MASK_BITS 8
#define VT_RT_TRACE_META0_SBT_OFFSET_SHIFT 16
#define VT_RT_TRACE_META0_SBT_OFFSET_BITS 4
#define VT_RT_TRACE_META0_SBT_STRIDE_SHIFT 20
#define VT_RT_TRACE_META0_SBT_STRIDE_BITS 4

/* trace_meta1 also replaces the former CPS active-level PDS header word. */
#define VT_RT_TRACE_META1_MISS_INDEX_SHIFT 0
#define VT_RT_TRACE_META1_MISS_INDEX_BITS 16
#define VT_RT_TRACE_META1_ACTIVE_LEVEL_SHIFT 16
#define VT_RT_TRACE_META1_ACTIVE_LEVEL_BITS 8
#define VT_RT_TRACE_META1_MAX_RECURSION_DEPTH \
   (1u << VT_RT_TRACE_META1_ACTIVE_LEVEL_BITS)

/* Status is a lifecycle handshake, not a traversal result. */
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
   VT_RT_HIT_RECORD_META0 = 0,
   VT_RT_HIT_RECORD_META1 = 1,
   VT_RT_HIT_RECORD_HIT_T = 2,
   VT_RT_HIT_RECORD_BARYCENTRICS_X = 3,
   VT_RT_HIT_RECORD_BARYCENTRICS_Y = 4,
   VT_RT_HIT_RECORD_PRIMITIVE_ID = 5,
   VT_RT_HIT_RECORD_INSTANCE_ID = 6,
   VT_RT_HIT_RECORD_GEOMETRY_ID = 7,
   VT_RT_HIT_RECORD_HIT_KIND = 8,
   VT_RT_HIT_RECORD_WORD_COUNT = 9,
};

/* meta0 is deliberately packed because these values are all shader-visible
 * but each has a much smaller Vulkan-defined range than one full word. */
#define VT_RT_HIT_RECORD_META0_VALID_SHIFT 0
#define VT_RT_HIT_RECORD_META0_STATUS_SHIFT 1
#define VT_RT_HIT_RECORD_META0_STATUS_BITS 3
#define VT_RT_HIT_RECORD_META0_FRONT_FACE_SHIFT 4
#define VT_RT_HIT_RECORD_META0_OPAQUE_SHIFT 5
#define VT_RT_HIT_RECORD_META0_NEED_SOFTWARE_OPACITY_TEST_SHIFT 6
#define VT_RT_HIT_RECORD_META0_INSTANCE_CUSTOM_INDEX_SHIFT 7
#define VT_RT_HIT_RECORD_META0_INSTANCE_CUSTOM_INDEX_BITS 24

#define VT_RT_HIT_RECORD_META1_INSTANCE_SBT_RECORD_OFFSET_SHIFT 0
#define VT_RT_HIT_RECORD_META1_INSTANCE_SBT_RECORD_OFFSET_BITS 24

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
#define VT_RT_ABI_CONTROL_BASE_BYTES VT_RT_ABI_RAY_STATE_SIZE_BYTES
#define VT_RT_ABI_CONTROL_SIZE_BYTES \
   (VT_RT_CONTROL_WORD_COUNT * VT_RT_ABI_WORD_BYTES)

#define VT_RT_ABI_CANDIDATE_HIT_RECORD_BASE_BYTES 76
#define VT_RT_ABI_CANDIDATE_HIT_RECORD_SIZE_BYTES 36

#define VT_RT_ABI_COMMITTED_HIT_RECORD_BASE_BYTES 112
#define VT_RT_ABI_COMMITTED_HIT_RECORD_SIZE_BYTES 36

/*
 * Fixed-header physical PDS layout.
 *
 * Logical ABI offsets above remain lane-private byte offsets used by shader
 * lowering.  The physical PDS mapping intentionally changes at the hit
 * records: ray state and control are field-major, whereas each candidate or
 * committed record is a contiguous lane-major nine-word object.  For global
 * PDS thread ID t and resident PDS-thread count T, physical word indices are:
 *
 *   ray/control word w:             w * T + t
 *   candidate record word r:        prefix_words * T + t * record_words + r
 *   committed record word r:        (prefix_words + record_words) * T
 *                                   + t * record_words + r
 *
 * This permits an RTcore intersection unit to write a completed lane's whole
 * record without a lane-by-field transpose crossbar.  Payload and ordinary
 * shader-private storage remain outside this fixed-header mapping.
 */
#define VT_RT_ABI_PDS_FIELD_MAJOR_PREFIX_SIZE_BYTES \
   VT_RT_ABI_CANDIDATE_HIT_RECORD_BASE_BYTES
#define VT_RT_ABI_PDS_FIELD_MAJOR_PREFIX_WORD_COUNT 19
#define VT_RT_ABI_PDS_LANE_MAJOR_HIT_RECORD_WORD_COUNT \
   VT_RT_HIT_RECORD_WORD_COUNT
#define VT_RT_ABI_PDS_LANE_MAJOR_HIT_RECORD_COUNT 2

/* Compiler-to-LSU private-address mode tag.  It is never part of a logical
 * ABI offset: dedicated NIR hit-record accesses set it so the PDS boundary
 * applies the lane-major record mapping rather than normal AS5 field-major
 * addressing. */
#define VT_RT_ABI_HIT_RECORD_ADDR_TAG (-2147483647 - 1)

#define VT_RT_ABI_HIT_ATTRIB_BASE_BYTES 148
/* The payload starts after candidate and committed non-barycentric hit
 * attributes.  Intersection/any-hit write the candidate half; an accepted
 * report snapshots it into the committed half before closest-hit runs.
 * The two triangle barycentric words live in hit-record words 3 and 4 and
 * are therefore not part of this variable-sized region. */
#define VT_RT_ABI_HIT_ATTRIB_SIZE_BYTES 0

#define VT_RT_ABI_PAYLOAD_BASE_BYTES_FOR_HIT_ATTRIB_SIZE(hit_attrib_size_bytes) \
   (VT_RT_ABI_HIT_ATTRIB_BASE_BYTES + 2 * (hit_attrib_size_bytes))

#define VT_RT_ABI_SHADER_REGION_BASE_BYTES VT_RT_ABI_HIT_ATTRIB_BASE_BYTES
#define VT_RT_ABI_PAYLOAD_BASE_BYTES VT_RT_ABI_SHADER_REGION_BASE_BYTES
/* The linked RT pipeline determines P before lowering:
 * ALIGN16(max(location * 16 + natural_size)).  Megakernel payload storage is
 * D consecutive frames: frame d is [payload_base + d * P,
 * payload_base + (d + 1) * P).  The fixed Header ends at payload_base; P and
 * D are pipeline properties, not fixed RTcore Header constants. */
#define VT_RT_ABI_DEFAULT_PAYLOAD_SIZE_BYTES 0
#define VT_RT_ABI_REGION_SIZE_BYTES VT_RT_ABI_SHADER_REGION_BASE_BYTES
#define VT_RT_ABI_FIXED_HEADER_SIZE_BYTES VT_RT_ABI_SHADER_REGION_BASE_BYTES

#endif
