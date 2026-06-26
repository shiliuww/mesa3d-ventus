/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VT_NIR_H
#define VT_NIR_H

#include <stdbool.h>
#include <stdint.h>

#include "nir/nir.h"
#include "nir/nir_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

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
};

struct vt_nir_rt_config;

struct vt_nir_rt_trace_params {
   nir_def *accel_struct;
   nir_def *flags;
   nir_def *cull_mask;
   nir_def *sbt_offset;
   nir_def *sbt_stride;
   nir_def *miss_index;
   nir_def *origin;
   nir_def *tmin;
   nir_def *direction;
   nir_def *tmax;
   nir_def *payload;
};

struct vt_nir_rt_result {
   nir_def *hit;
   nir_def *hit_t;
   nir_def *sbt_index;
};

typedef void (*vt_nir_rt_emit_shader_cb)(nir_builder *b,
                                         const struct vt_nir_rt_trace_params *params,
                                         const struct vt_nir_rt_result *result,
                                         void *data);

struct vt_nir_rt_config {
   uint32_t slot_stride_bytes;
   uint32_t slot_base_bytes;
   bool enable_monolithic_shader_callbacks;
   vt_nir_rt_emit_shader_cb emit_closest_hit;
   vt_nir_rt_emit_shader_cb emit_miss;
   void *callback_data;
};

struct vt_nir_rt_config vt_nir_rt_default_config(void);

struct vt_nir_rt_result
vt_nir_build_traversal(nir_builder *b,
                       const struct vt_nir_rt_trace_params *params,
                       const struct vt_nir_rt_config *config);

bool vt_nir_lower_rt_io_monolithic(nir_shader *shader,
                                   const struct vt_nir_rt_config *config);

bool vt_nir_lower_rt_abi(nir_shader *shader,
                         const struct vt_nir_rt_config *config);

#ifdef __cplusplus
}
#endif

#endif
