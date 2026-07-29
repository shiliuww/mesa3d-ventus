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
   /* Compatibility mirror: it publishes a global mailbox but still executes
    * the legacy PDS traversal loop. */
   VT_RT_EXECUTION_WAVEFRONT_GLOBAL_QUEUE = 1,
   /* PDS-free profile.  Raygen publishes level-zero requests in global
    * mailboxes; RTcore compacts them into, then consumes, the queue. */
   VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER = 2,
   /* Terminal miss/closest-hit execution over a compact global ray_ref list. */
   VT_RT_EXECUTION_WAVEFRONT_GLOBAL_CALLBACK = 3,
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
   /* PDS-free static RT state: ray, hit records, attributes, and payload. */
   uint32_t region_size_bytes;
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
   nir_def *abi_tag;
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
   /* Global address of a RayQueueHeader followed by field-major planes.
    * It remains u64 in the NIR/dispatch ABI; an RV32 execution submission is
    * accepted only after the driver has checked the complete allocation range.
    */
   nir_def *base_addr;
   nir_def *field_stride_bytes;
};

struct vt_nir_rt_submission_mailbox_args {
   /* Global address of this lane's reusable submit mailbox; see the RV32
    * submission range rule above. */
   nir_def *base_addr;
   nir_def *generation;
   /* Queue identity is carried in the producer mailbox, not inferred from
    * PDS or from a physical lane. */
   nir_def *queue_base_addr;
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
   /* Keep reportIntersectionEXT in an intersection shader intact until the
    * per-hit-group callback builder can inline that group's any-hit code at
    * the report site. */
   bool defer_intersection_reports;
   bool enable_monolithic_shader_callbacks;
   vt_nir_rt_emit_shader_cb emit_closest_hit;
   vt_nir_rt_emit_shader_cb emit_miss;
   vt_nir_rt_emit_shader_cb emit_any_hit;
   vt_nir_rt_emit_shader_cb emit_intersection;
   void *callback_data;
};

struct vt_nir_rt_config vt_nir_rt_default_config(void);

struct vt_nir_rt_abi_layout vt_nir_rt_default_abi_layout(void);

/* Rebind the common RT regions into a PDS-free global CPS frame. */
void vt_nir_rt_config_apply_global_cps_layout(
   struct vt_nir_rt_config *config,
   const struct vt_nir_rt_abi_layout *layout);

struct vt_rt_global_queue_layout
vt_nir_rt_global_queue_layout(uint32_t capacity_rays);

struct vt_rt_global_completion_layout
vt_nir_rt_global_completion_layout(uint32_t capacity_rays);

bool vt_nir_rt_global_queue_capacity(uint32_t width, uint32_t height,
                                     uint32_t depth,
                                     uint32_t max_ray_recursion_depth,
                                     uint32_t children_per_ray,
                                     uint32_t *capacity_rays);

/* Number of global mailbox records for all resident physical worker lanes.
 * The record kind is PRIMARY, shadow, reflection, or refraction. */
bool vt_nir_rt_submission_mailbox_capacity(uint32_t resident_workers,
                                           uint32_t lanes_per_worker,
                                           uint64_t *mailbox_records);

nir_def *vt_nir_rt_global_queue_field_addr(
   nir_builder *b, const struct vt_nir_rt_global_queue_args *queue,
   enum vt_rt_global_trace_field field, nir_def *ray_ref);

void vt_nir_rt_store_submission_mailbox(
   nir_builder *b, const struct vt_nir_rt_submission_mailbox_args *mailbox,
   const struct vt_nir_rt_trace_params *params, nir_def *parent_frame,
   nir_def *cps_frame, nir_def *continuation_id, nir_def *cps_stack_size,
   nir_def *launch_id, nir_def *phase_generation, nir_def *depth);

/* Late lowering for the Mesa-private enqueue pseudo.  The result contains
 * only ordinary global loads/stores and atomics understood by the backend. */
bool vt_nir_lower_rt_enqueue(nir_shader *shader);

/* Computes the global staging address owned by this physical worker lane.
 * worker_id is globally unique across resident SM/warp slots. */
nir_def *vt_nir_rt_submission_mailbox_addr(
   nir_builder *b, nir_def *mailbox_pool_addr, nir_def *mailbox_stride_bytes,
   nir_def *worker_id, nir_def *lane_id,
   enum vt_rt_submission_mailbox_child_kind kind);

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

nir_def *vt_nir_load_ray_state_word(nir_builder *b,
                                    const struct vt_nir_rt_config *config,
                                    enum vt_rt_shader_layout layout,
                                    enum vt_rt_slot_word word);

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

/* Insert terminal miss/closest-hit dispatch after each traceRayEXT without
 * consuming the trace instruction.  CPS splitting subsequently moves that
 * dispatch into the root continuation, where the global completion plane is
 * the source of the terminal result. */
bool vt_nir_lower_rt_global_root_continuations(
   nir_shader *shader, const struct vt_nir_rt_config *config);

/* PDS-free root raygen producer.  Call vt_nir_lower_rt_cps() first, then pass
 * its resulting initial/resume shaders through these lowerings.  The MVP owns
 * exactly one root CPS frame per launch invocation; callback/child traces are
 * deliberately outside this profile. */
bool vt_nir_lower_rt_global_raygen(nir_shader *shader,
                                   const struct vt_nir_rt_config *config);

bool vt_nir_lower_rt_global_resume(nir_shader *shader,
                                   const struct vt_nir_rt_config *config);

/* Materialize Vulkan push-constant loads through the frozen RT dispatch
 * snapshot.  This is shared by compatibility and global execution profiles. */
bool vt_nir_lower_rt_push_constants(nir_shader *shader);

/* Rewrites a terminal miss/closest-hit shader's legacy RT slot accesses to
 * the global queue/completion planes selected by the compact dispatch list. */
bool vt_nir_lower_rt_global_callback(nir_shader *shader,
                                     const struct vt_nir_rt_config *config);

/* Compact callback helpers for driver-side report-site lowering. */
nir_def *vt_nir_rt_global_callback_ray_ref(nir_builder *b);
nir_def *vt_nir_rt_global_load_completion_addr(
   nir_builder *b, nir_def *ray_ref, enum vt_rt_global_completion_field lo);

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
   VT_NIR_RT_DISPATCH_ARG_ABI_TAG = 148,
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
   /* Target wavefront ABI.  These remain zero until worker dispatch lands. */
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_QUEUE_BASE = 200,
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_QUEUE_CAPACITY = 208,
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_QUEUE_GENERATION = 212,
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_CPS_ARENA_ADDR = 216,
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_CPS_FRAME_STRIDE = 224,
   VT_NIR_RT_DISPATCH_ARG_MAX_RAY_RECURSION_DEPTH = 228,
   /* Separate ray_ref-indexed completion/result plane. */
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_BASE = 232,
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_CAPACITY = 240,
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_GENERATION = 244,
   /* Compact worker/callback/resume dispatch: ray_ref_list[index]. */
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_RAY_REF_LIST_BASE = 248,
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_RAY_REF_COUNT = 256,
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_RAY_REF_GENERATION = 260,
   VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_FIELD_STRIDE = 264,
   /* Immutable Vulkan push-constant snapshot shared by all RT ELFs. */
   VT_NIR_RT_DISPATCH_ARG_PUSH_CONSTANTS_ADDR = 272,
   VT_NIR_RT_DISPATCH_ARG_PUSH_CONSTANTS_SIZE = 280,
   VT_NIR_RT_DISPATCH_ARG_SIZE = 288,
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
   /* Sampler state for combined image samplers.  Keep this in the software
    * descriptor ABI rather than baking a default sampler into the LLVM
    * lowering: different Vulkan descriptors may share one image. */
   VT_NIR_RT_DESCRIPTOR_RECORD_SAMPLER_MAG_FILTER = 40,
   VT_NIR_RT_DESCRIPTOR_RECORD_SAMPLER_MIN_FILTER = 44,
   VT_NIR_RT_DESCRIPTOR_RECORD_SAMPLER_ADDRESS_MODE_U = 48,
   VT_NIR_RT_DESCRIPTOR_RECORD_SAMPLER_ADDRESS_MODE_V = 52,
   VT_NIR_RT_DESCRIPTOR_RECORD_SIZE = 56,
};

#ifdef __cplusplus
}
#endif

#endif
