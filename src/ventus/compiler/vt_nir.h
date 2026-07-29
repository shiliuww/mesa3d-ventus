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
#include "ventus/compiler/vt_rt_global_abi.h"
#include "ventus/compiler/vt_rt_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

enum vt_rt_execution_model {
   VT_RT_EXECUTION_ITERATIVE_RAYGEN_LOOP = 0,
   VT_RT_EXECUTION_WAVEFRONT_GLOBAL_QUEUE = 1,
};

enum vt_rt_shader_layout {
   VT_RT_LAYOUT_RAYGEN = 0,
   VT_RT_LAYOUT_CLOSEST_HIT = 1,
   VT_RT_LAYOUT_MISS = 2,
   VT_RT_LAYOUT_ANY_HIT = 3,
   VT_RT_LAYOUT_INTERSECTION = 4,
   VT_RT_LAYOUT_COUNT = 5,
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
   uint32_t control_base_bytes;
   uint32_t candidate_hit_record_base_bytes;
   uint32_t hit_record_base_bytes;
   uint32_t hit_attrib_base_bytes;
   uint32_t hit_attrib_size_bytes;
   uint32_t total_size_bytes;
};

struct vt_nir_rt_abi_layout {
   uint32_t ray_state_base_bytes;
   uint32_t ray_state_size_bytes;
   uint32_t payload_base_bytes;
   uint32_t payload_size_bytes;
   uint32_t control_base_bytes;
   uint32_t control_size_bytes;
   uint32_t candidate_hit_record_base_bytes;
   uint32_t candidate_hit_record_size_bytes;
   uint32_t hit_record_base_bytes;
   uint32_t hit_record_size_bytes;
   uint32_t hit_attrib_base_bytes;
   uint32_t hit_attrib_size_bytes;
   uint32_t total_size_bytes;
};

struct vt_nir_rt_config;

struct vt_nir_rt_trace_params {
   nir_def *accel_struct;
   nir_def *flags;
   nir_def *cull_mask;
   nir_def *sbt_base;
   nir_def *sbt_stride_bytes;
   nir_def *miss_sbt_base;
   nir_def *miss_sbt_stride_bytes;
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
   nir_def *slot_base;
   nir_def *status;
   nir_def *hit;
   nir_def *hit_t;
   nir_def *sbt_index;
   nir_def *instance_sbt_record_offset;
   nir_def *geometry_index;
   nir_def *hit_shader_record_ptr;
   nir_def *miss_shader_record_ptr;
   nir_def *shader_record_ptr;
};

struct vt_nir_rt_global_queue_args {
   /*
    * Low 32-bit Ventus global address of a RayQueueHeader followed by
    * field-major planes.  Vulkan device-address values are normalized to this
    * target address domain by the driver before reaching NIR.
    */
   nir_def *base_addr;
   nir_def *field_stride_bytes;
};

struct vt_nir_rt_submission_mailbox_args {
   /* Low 32-bit global address of this lane's reusable submit mailbox. */
   nir_def *base_addr;
   nir_def *generation;
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
   uint64_t hit_sbt_base_addr;
   uint32_t hit_sbt_record_stride;
   uint32_t shader_group_handle_size;
   uint32_t miss_depth_value;
   bool enable_monolithic_shader_callbacks;
   vt_nir_rt_emit_shader_cb emit_closest_hit;
   vt_nir_rt_emit_shader_cb emit_miss;
   vt_nir_rt_emit_shader_cb emit_any_hit;
   vt_nir_rt_emit_shader_cb emit_intersection;
   void *callback_data;
};

struct vt_nir_rt_config vt_nir_rt_default_config(void);

struct vt_nir_rt_abi_layout vt_nir_rt_default_abi_layout(void);

struct vt_rt_global_queue_layout
vt_nir_rt_global_queue_layout(uint32_t capacity_rays);

nir_def *vt_nir_rt_global_queue_field_addr(
   nir_builder *b, const struct vt_nir_rt_global_queue_args *queue,
   enum vt_rt_global_trace_field field, nir_def *ray_ref);

void vt_nir_rt_store_submission_mailbox(
   nir_builder *b, const struct vt_nir_rt_submission_mailbox_args *mailbox,
   const struct vt_nir_rt_trace_params *params, nir_def *parent_frame,
   nir_def *phase_generation, nir_def *depth);

void vt_nir_rt_config_apply_abi_layout(struct vt_nir_rt_config *config,
                                       const struct vt_nir_rt_abi_layout *layout);

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

nir_def *vt_nir_hit_record_addr(nir_builder *b,
                                const struct vt_nir_rt_config *config,
                                enum vt_rt_shader_layout layout,
                                enum vt_rt_hit_record_word word);

nir_def *vt_nir_candidate_hit_record_addr(nir_builder *b,
                                          const struct vt_nir_rt_config *config,
                                          enum vt_rt_shader_layout layout,
                                          enum vt_rt_hit_record_word word);

nir_def *vt_nir_control_addr(nir_builder *b,
                             const struct vt_nir_rt_config *config,
                             enum vt_rt_shader_layout layout,
                             enum vt_rt_control_word word);

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

nir_def *vt_nir_load_hit_record_word(nir_builder *b,
                                     const struct vt_nir_rt_config *config,
                                     enum vt_rt_shader_layout layout,
                                     enum vt_rt_hit_record_word word,
                                     uint32_t bit_size);

void vt_nir_store_hit_record_word(nir_builder *b,
                                  const struct vt_nir_rt_config *config,
                                  enum vt_rt_shader_layout layout,
                                  enum vt_rt_hit_record_word word,
                                  nir_def *value);

nir_def *vt_nir_load_candidate_hit_record_word(nir_builder *b,
                                               const struct vt_nir_rt_config *config,
                                               enum vt_rt_shader_layout layout,
                                               enum vt_rt_hit_record_word word,
                                               uint32_t bit_size);

void vt_nir_store_candidate_hit_record_word(nir_builder *b,
                                            const struct vt_nir_rt_config *config,
                                            enum vt_rt_shader_layout layout,
                                            enum vt_rt_hit_record_word word,
                                            nir_def *value);

nir_def *vt_nir_load_control_word(nir_builder *b,
                                  const struct vt_nir_rt_config *config,
                                  enum vt_rt_shader_layout layout,
                                  enum vt_rt_control_word word,
                                  uint32_t bit_size);

void vt_nir_store_control_word(nir_builder *b,
                               const struct vt_nir_rt_config *config,
                               enum vt_rt_shader_layout layout,
                               enum vt_rt_control_word word,
                               nir_def *value);

struct vt_nir_rt_result
vt_nir_build_traversal(nir_builder *b,
                       const struct vt_nir_rt_trace_params *params,
                       const struct vt_nir_rt_config *config);

bool vt_nir_lower_rt_io_monolithic(nir_shader *shader,
                                   const struct vt_nir_rt_config *config);

bool vt_nir_lower_rt_abi(nir_shader *shader,
                         const struct vt_nir_rt_config *config);

bool vt_nir_lower_rt_payload(nir_shader *shader,
                             const struct vt_nir_rt_config *config);

bool vt_nir_lower_rt_hit_record(nir_shader *shader,
                                const struct vt_nir_rt_config *config);

bool vt_nir_lower_shader_record_ptr(nir_shader *shader,
                                    const struct vt_nir_rt_config *config);

bool vt_nir_lower_rt_control(nir_shader *shader,
                             const struct vt_nir_rt_config *config);

bool vt_nir_lower_shader_scratch(nir_shader *shader,
                                 const struct vt_nir_rt_config *config);

bool vt_nir_lower_descriptors(nir_shader *shader);

/*
 * CPS preparation for raygen traceRayEXT calls.  Mesa owns the exact live SSA
 * spill layout; Ventus only consumes the resulting frame size and resume
 * shaders.  The caller owns mem_ctx and therefore the returned shader array.
 */
struct vt_nir_rt_cps_info {
   nir_shader **resume_shaders;
   uint32_t num_resume_shaders;
   uint32_t frame_size_bytes;
};

bool vt_nir_lower_rt_cps(nir_shader *shader,
                         struct vt_nir_rt_cps_info *info,
                         void *mem_ctx);

enum vt_nir_rt_canonicalize_status {
   VT_NIR_RT_CANONICALIZE_SUCCESS = 0,
   VT_NIR_RT_CANONICALIZE_RESIDUAL_FUNCTION_TEMP,
};

struct vt_nir_rt_canonicalize_error {
   const char *variable_name;
};

enum vt_nir_rt_canonicalize_status
vt_nir_canonicalize_rt_for_llvm(
   nir_shader *shader,
   struct vt_nir_rt_canonicalize_error *error);

void vt_nir_pipeline_scratch_init(struct vt_nir_pipeline_scratch_info *info);

void vt_nir_pipeline_scratch_include_shader(struct vt_nir_pipeline_scratch_info *info,
                                            const nir_shader *shader);

uint32_t vt_nir_pipeline_compute_scratch(nir_shader **shaders,
                                         unsigned shader_count,
                                         struct vt_nir_pipeline_scratch_info *info);

uint32_t vt_nir_pipeline_scratch_size(const struct vt_nir_pipeline_scratch_info *info);

enum vt_nir_rt_dispatch_arg {
   VT_NIR_RT_DISPATCH_ARG_TLAS_ADDR = 0,
   VT_NIR_RT_DISPATCH_ARG_IMAGE_ADDR = 8,
   VT_NIR_RT_DISPATCH_ARG_CAMERA_ADDR = 16,
   VT_NIR_RT_DISPATCH_ARG_RAYGEN_SBT_ADDR = 24,
   VT_NIR_RT_DISPATCH_ARG_RAYGEN_SBT_SIZE = 32,
   VT_NIR_RT_DISPATCH_ARG_RAYGEN_SBT_STRIDE = 40,
   VT_NIR_RT_DISPATCH_ARG_MISS_SBT_ADDR = 48,
   VT_NIR_RT_DISPATCH_ARG_MISS_SBT_SIZE = 56,
   VT_NIR_RT_DISPATCH_ARG_MISS_SBT_STRIDE = 64,
   VT_NIR_RT_DISPATCH_ARG_HIT_SBT_ADDR = 72,
   VT_NIR_RT_DISPATCH_ARG_HIT_SBT_SIZE = 80,
   VT_NIR_RT_DISPATCH_ARG_HIT_SBT_STRIDE = 88,
   VT_NIR_RT_DISPATCH_ARG_CALLABLE_SBT_ADDR = 96,
   VT_NIR_RT_DISPATCH_ARG_CALLABLE_SBT_SIZE = 104,
   VT_NIR_RT_DISPATCH_ARG_CALLABLE_SBT_STRIDE = 112,
   VT_NIR_RT_DISPATCH_ARG_WIDTH = 120,
   VT_NIR_RT_DISPATCH_ARG_HEIGHT = 124,
   VT_NIR_RT_DISPATCH_ARG_DEPTH = 128,
   VT_NIR_RT_DISPATCH_ARG_IMAGE_FORMAT = 132,
   VT_NIR_RT_DISPATCH_ARG_IMAGE_STRIDE = 136,
   VT_NIR_RT_DISPATCH_ARG_CAMERA_SIZE = 140,
   VT_NIR_RT_DISPATCH_ARG_SCRATCH_STRIDE = 144,
   VT_NIR_RT_DISPATCH_ARG_DESCRIPTOR_TABLE_ADDR = 152,
   VT_NIR_RT_DISPATCH_ARG_DESCRIPTOR_COUNT = 160,
   VT_NIR_RT_DISPATCH_ARG_DESCRIPTOR_RECORD_STRIDE = 164,
   /* Driver-private CPS spill arena.  It is unused by the legacy path. */
   VT_NIR_RT_DISPATCH_ARG_CONTINUATION_ARENA_ADDR = 168,
   VT_NIR_RT_DISPATCH_ARG_CONTINUATION_FRAME_STRIDE = 176,
   VT_NIR_RT_DISPATCH_ARG_CONTINUATION_CONTEXT_COUNT = 180,
   /* Transitional wavefront handoff pool; legacy traversal ignores it. */
   VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_BASE = 184,
   VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_STRIDE = 192,
   VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_GENERATION = 196,
   VT_NIR_RT_DISPATCH_ARG_SIZE = 200,
};

enum vt_nir_rt_descriptor_record {
   VT_NIR_RT_DESCRIPTOR_RECORD_TYPE = 0,
   VT_NIR_RT_DESCRIPTOR_RECORD_SET = 4,
   VT_NIR_RT_DESCRIPTOR_RECORD_BINDING = 8,
   VT_NIR_RT_DESCRIPTOR_RECORD_FORMAT = 12,
   VT_NIR_RT_DESCRIPTOR_RECORD_DEVICE_ADDR = 16,
   VT_NIR_RT_DESCRIPTOR_RECORD_BYTE_SIZE = 24,
   VT_NIR_RT_DESCRIPTOR_RECORD_STRIDE = 32,
   VT_NIR_RT_DESCRIPTOR_RECORD_AUX = 36,
   VT_NIR_RT_DESCRIPTOR_RECORD_SIZE = 40,
};

#ifdef __cplusplus
}
#endif

#endif
