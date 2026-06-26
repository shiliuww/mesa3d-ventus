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

enum vt_rt_status {
   VT_RT_STATUS_IDLE = 0,
   VT_RT_STATUS_TRACE_REQUEST = 1,
   VT_RT_STATUS_HIT = 2,
   VT_RT_STATUS_MISS = 3,
   VT_RT_STATUS_DONE = 4,
};

enum vt_rt_execution_model {
   VT_RT_EXECUTION_ITERATIVE_RAYGEN_LOOP = 0,
};

enum vt_rt_shader_layout {
   VT_RT_LAYOUT_RAYGEN = 0,
   VT_RT_LAYOUT_CLOSEST_HIT = 1,
   VT_RT_LAYOUT_MISS = 2,
   VT_RT_LAYOUT_COUNT = 3,
};

enum vt_rt_payload_word {
   VT_RT_PAYLOAD_DEPTH = 0,
   VT_RT_PAYLOAD_WEIGHT = 1,
   VT_RT_PAYLOAD_COLOR_X = 2,
   VT_RT_PAYLOAD_COLOR_Y = 3,
   VT_RT_PAYLOAD_COLOR_Z = 4,
   VT_RT_PAYLOAD_RAY_ORIGIN_X = 5,
   VT_RT_PAYLOAD_RAY_ORIGIN_Y = 6,
   VT_RT_PAYLOAD_RAY_ORIGIN_Z = 7,
   VT_RT_PAYLOAD_RAY_DIRECTION_X = 8,
   VT_RT_PAYLOAD_RAY_DIRECTION_Y = 9,
   VT_RT_PAYLOAD_RAY_DIRECTION_Z = 10,
};

struct vt_nir_scratchpad_layout {
   uint32_t base_bytes;
   uint32_t stride_bytes;
   uint32_t ray_state_base_bytes;
   uint32_t payload_base_bytes;
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

struct vt_nir_pipeline_scratch_info {
   uint32_t per_lane_scratch_size;
   uint32_t shader_scratch_size[VT_RT_LAYOUT_COUNT];
};

typedef void (*vt_nir_rt_emit_shader_cb)(nir_builder *b,
                                         const struct vt_nir_rt_trace_params *params,
                                         const struct vt_nir_rt_result *result,
                                         void *data);

struct vt_nir_rt_config {
   enum vt_rt_execution_model execution_model;
   struct vt_nir_scratchpad_layout scratchpad[VT_RT_LAYOUT_COUNT];
   uint32_t miss_depth_value;
   bool enable_monolithic_shader_callbacks;
   vt_nir_rt_emit_shader_cb emit_closest_hit;
   vt_nir_rt_emit_shader_cb emit_miss;
   void *callback_data;
};

struct vt_nir_rt_config vt_nir_rt_default_config(void);

enum vt_rt_shader_layout vt_nir_layout_for_stage(mesa_shader_stage stage);

nir_def *vt_nir_scratchpad_base(nir_builder *b,
                                const struct vt_nir_rt_config *config,
                                enum vt_rt_shader_layout layout);

nir_def *vt_nir_scratchpad_addr(nir_builder *b,
                                const struct vt_nir_rt_config *config,
                                enum vt_rt_shader_layout layout,
                                uint32_t byte_offset);

nir_def *vt_nir_payload_addr(nir_builder *b,
                             const struct vt_nir_rt_config *config,
                             enum vt_rt_shader_layout layout,
                             enum vt_rt_payload_word word);

nir_def *vt_nir_load_payload_word(nir_builder *b,
                                  const struct vt_nir_rt_config *config,
                                  enum vt_rt_shader_layout layout,
                                  enum vt_rt_payload_word word,
                                  uint32_t bit_size);

void vt_nir_store_payload_word(nir_builder *b,
                               const struct vt_nir_rt_config *config,
                               enum vt_rt_shader_layout layout,
                               enum vt_rt_payload_word word,
                               nir_def *value);

struct vt_nir_rt_result
vt_nir_build_traversal(nir_builder *b,
                       const struct vt_nir_rt_trace_params *params,
                       const struct vt_nir_rt_config *config);

bool vt_nir_lower_rt_io_monolithic(nir_shader *shader,
                                   const struct vt_nir_rt_config *config);

bool vt_nir_lower_rt_abi(nir_shader *shader,
                         const struct vt_nir_rt_config *config);

bool vt_nir_lower_shader_scratch(nir_shader *shader,
                                 const struct vt_nir_rt_config *config);

void vt_nir_pipeline_scratch_init(struct vt_nir_pipeline_scratch_info *info);

void vt_nir_pipeline_scratch_include_shader(struct vt_nir_pipeline_scratch_info *info,
                                            const nir_shader *shader);

uint32_t vt_nir_pipeline_compute_scratch(nir_shader **shaders,
                                         unsigned shader_count,
                                         struct vt_nir_pipeline_scratch_info *info);

uint32_t vt_nir_pipeline_scratch_size(const struct vt_nir_pipeline_scratch_info *info);

#ifdef __cplusplus
}
#endif

#endif
