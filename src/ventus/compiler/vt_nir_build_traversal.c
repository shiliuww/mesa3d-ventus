/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

struct vt_nir_rt_config
vt_nir_rt_default_config(void)
{
   struct vt_nir_rt_abi_layout abi = vt_nir_rt_default_abi_layout();
   struct vt_nir_rt_config config = {
      .execution_model = VT_RT_EXECUTION_ITERATIVE_RAYGEN_LOOP,
      .scratchpad = {
         [VT_RT_LAYOUT_RAYGEN] = {
            .base_bytes = 0,
            .stride_bytes = 256,
         },
         [VT_RT_LAYOUT_CLOSEST_HIT] = {
            .base_bytes = 0,
            .stride_bytes = 256,
         },
         [VT_RT_LAYOUT_MISS] = {
            .base_bytes = 0,
            .stride_bytes = 256,
         },
      },
      .hit_sbt_base_addr = 0,
      .hit_sbt_record_stride = 64,
      .shader_group_handle_size = 32,
      .miss_depth_value = 0x7fffffff,
      .enable_monolithic_shader_callbacks = false,
      .emit_closest_hit = NULL,
      .emit_miss = NULL,
      .emit_any_hit = NULL,
      .emit_intersection = NULL,
      .callback_data = NULL,
   };
   vt_nir_rt_config_apply_abi_layout(&config, &abi);
   return config;
}

enum vt_rt_shader_layout
vt_nir_layout_for_stage(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_ANY_HIT:
      return VT_RT_LAYOUT_ANY_HIT;
   case MESA_SHADER_CLOSEST_HIT:
      return VT_RT_LAYOUT_CLOSEST_HIT;
   case MESA_SHADER_INTERSECTION:
      return VT_RT_LAYOUT_INTERSECTION;
   case MESA_SHADER_MISS:
      return VT_RT_LAYOUT_MISS;
   case MESA_SHADER_RAYGEN:
   default:
      return VT_RT_LAYOUT_RAYGEN;
   }
}

nir_def *
vt_nir_scratchpad_base(nir_builder *b,
                       const struct vt_nir_rt_config *config,
                       enum vt_rt_shader_layout layout)
{
   const struct vt_nir_scratchpad_layout *sp = &config->scratchpad[layout];
   return nir_imm_int(b, sp->base_bytes);
}

nir_def *
vt_nir_scratchpad_addr(nir_builder *b,
                       const struct vt_nir_rt_config *config,
                       enum vt_rt_shader_layout layout,
                       uint32_t byte_offset)
{
   return nir_iadd_imm(b, vt_nir_scratchpad_base(b, config, layout), byte_offset);
}

nir_def *
vt_nir_payload_addr(nir_builder *b,
                    const struct vt_nir_rt_config *config,
                    enum vt_rt_shader_layout layout,
                    enum vt_rt_payload_word word)
{
   const struct vt_nir_scratchpad_layout *sp = &config->scratchpad[layout];
   return vt_nir_scratchpad_addr(b, config, layout,
                                 sp->payload_base_bytes + word * 4);
}

nir_def *
vt_nir_hit_record_addr(nir_builder *b,
                       const struct vt_nir_rt_config *config,
                       enum vt_rt_shader_layout layout,
                       enum vt_rt_hit_record_word word)
{
   const struct vt_nir_scratchpad_layout *sp = &config->scratchpad[layout];
   return vt_nir_scratchpad_addr(b, config, layout,
                                 sp->hit_record_base_bytes + word * 4);
}

nir_def *
vt_nir_candidate_hit_record_addr(nir_builder *b,
                                 const struct vt_nir_rt_config *config,
                                 enum vt_rt_shader_layout layout,
                                 enum vt_rt_hit_record_word word)
{
   const struct vt_nir_scratchpad_layout *sp = &config->scratchpad[layout];
   return vt_nir_scratchpad_addr(b, config, layout,
                                 sp->candidate_hit_record_base_bytes + word * 4);
}

nir_def *
vt_nir_control_addr(nir_builder *b,
                    const struct vt_nir_rt_config *config,
                    enum vt_rt_shader_layout layout,
                    enum vt_rt_control_word word)
{
   const struct vt_nir_scratchpad_layout *sp = &config->scratchpad[layout];
   return vt_nir_scratchpad_addr(b, config, layout,
                                 sp->control_base_bytes + word * 4);
}

nir_def *
vt_nir_load_payload_word(nir_builder *b,
                         const struct vt_nir_rt_config *config,
                         enum vt_rt_shader_layout layout,
                         enum vt_rt_payload_word word,
                         uint32_t bit_size)
{
   return nir_ventus_rt_load_slot(b, 1, bit_size,
                                  vt_nir_payload_addr(b, config, layout, word));
}

void
vt_nir_store_payload_word(nir_builder *b,
                          const struct vt_nir_rt_config *config,
                          enum vt_rt_shader_layout layout,
                          enum vt_rt_payload_word word,
                          nir_def *value)
{
   nir_ventus_rt_store_slot(b, value,
                            vt_nir_payload_addr(b, config, layout, word));
}

nir_def *
vt_nir_load_hit_record_word(nir_builder *b,
                            const struct vt_nir_rt_config *config,
                            enum vt_rt_shader_layout layout,
                            enum vt_rt_hit_record_word word,
                            uint32_t bit_size)
{
   return nir_ventus_rt_load_slot(b, 1, bit_size,
                                  vt_nir_hit_record_addr(b, config, layout, word));
}

void
vt_nir_store_hit_record_word(nir_builder *b,
                             const struct vt_nir_rt_config *config,
                             enum vt_rt_shader_layout layout,
                             enum vt_rt_hit_record_word word,
                             nir_def *value)
{
   if (value->bit_size == 1)
      value = nir_b2i32(b, value);
   nir_ventus_rt_store_slot(b, value,
                            vt_nir_hit_record_addr(b, config, layout, word));
}

nir_def *
vt_nir_load_candidate_hit_record_word(nir_builder *b,
                                      const struct vt_nir_rt_config *config,
                                      enum vt_rt_shader_layout layout,
                                      enum vt_rt_hit_record_word word,
                                      uint32_t bit_size)
{
   return nir_ventus_rt_load_slot(
      b, 1, bit_size,
      vt_nir_candidate_hit_record_addr(b, config, layout, word));
}

void
vt_nir_store_candidate_hit_record_word(nir_builder *b,
                                       const struct vt_nir_rt_config *config,
                                       enum vt_rt_shader_layout layout,
                                       enum vt_rt_hit_record_word word,
                                       nir_def *value)
{
   if (value->bit_size == 1)
      value = nir_b2i32(b, value);
   nir_ventus_rt_store_slot(
      b, value, vt_nir_candidate_hit_record_addr(b, config, layout, word));
}

nir_def *
vt_nir_load_control_word(nir_builder *b,
                         const struct vt_nir_rt_config *config,
                         enum vt_rt_shader_layout layout,
                         enum vt_rt_control_word word,
                         uint32_t bit_size)
{
   return nir_ventus_rt_load_slot(b, 1, bit_size,
                                  vt_nir_control_addr(b, config, layout, word));
}

void
vt_nir_store_control_word(nir_builder *b,
                          const struct vt_nir_rt_config *config,
                          enum vt_rt_shader_layout layout,
                          enum vt_rt_control_word word,
                          nir_def *value)
{
   if (value->bit_size == 1)
      value = nir_b2i32(b, value);
   nir_ventus_rt_store_slot(b, value,
                            vt_nir_control_addr(b, config, layout, word));
}

static nir_def *
slot_offset(nir_builder *b, nir_def *base, enum vt_rt_slot_word word)
{
   return nir_iadd_imm(b, base, word * 4);
}

static void
store_word(nir_builder *b, nir_def *base, enum vt_rt_slot_word word, nir_def *value)
{
   if (!value)
      return;

   nir_ventus_rt_store_slot(b, value, slot_offset(b, base, word));
}

static nir_def *
load_word(nir_builder *b, nir_def *base, enum vt_rt_slot_word word)
{
   return nir_ventus_rt_load_slot(b, 1, 32, slot_offset(b, base, word));
}

static void
store_vec3(nir_builder *b, nir_def *base, enum vt_rt_slot_word first_word, nir_def *value)
{
   if (!value)
      return;

   store_word(b, base, first_word, nir_channel(b, value, 0));
   store_word(b, base, first_word + 1, nir_channel(b, value, 1));
   store_word(b, base, first_word + 2, nir_channel(b, value, 2));
}

static void
store_u64_words(nir_builder *b, nir_def *base, enum vt_rt_slot_word lo_word,
                nir_def *value)
{
   if (!value)
      return;

   if (value->bit_size == 64) {
      store_word(b, base, lo_word, nir_unpack_64_2x32_split_x(b, value));
      store_word(b, base, lo_word + 1, nir_unpack_64_2x32_split_y(b, value));
   } else {
      store_word(b, base, lo_word, value);
      store_word(b, base, lo_word + 1, nir_imm_int(b, 0));
   }
}

static nir_def *
build_hit_shader_record_ptr(nir_builder *b,
                            const struct vt_nir_rt_config *config,
                            const struct vt_nir_rt_trace_params *params,
                            nir_def *record_base,
                            nir_def *instance_sbt_record_offset,
                            nir_def *geometry_index)
{
   if (!record_base)
      record_base = nir_imm_int64(b, 0);

   nir_def *trace_stride =
      params->sbt_stride ? nir_u2u64(b, params->sbt_stride) : nir_imm_int64(b, 0);
   nir_def *trace_offset =
      params->sbt_offset ? nir_u2u64(b, params->sbt_offset) : nir_imm_int64(b, 0);
   nir_def *record_index =
      nir_iadd(b,
               nir_iadd(b, nir_u2u64(b, instance_sbt_record_offset),
                        nir_imul(b, nir_u2u64(b, geometry_index), trace_stride)),
               trace_offset);
   nir_def *record_stride = params->sbt_stride_bytes ?
      nir_u2u64(b, params->sbt_stride_bytes) : nir_imm_int64(b, 0);
   nir_def *record_offset = nir_imul(b, record_index, record_stride);
   record_offset = nir_iadd_imm(b, record_offset, config->shader_group_handle_size);

   return nir_iadd(b, nir_u2u64(b, record_base), record_offset);
}

static nir_def *
build_miss_shader_record_ptr(nir_builder *b,
                             const struct vt_nir_rt_config *config,
                             const struct vt_nir_rt_trace_params *params)
{
   nir_def *record_base = params->miss_sbt_base ?
      params->miss_sbt_base : nir_imm_int64(b, 0);
   nir_def *record_index = params->miss_index ?
      nir_u2u64(b, params->miss_index) : nir_imm_int64(b, 0);
   nir_def *record_stride = params->miss_sbt_stride_bytes ?
      nir_u2u64(b, params->miss_sbt_stride_bytes) : nir_imm_int64(b, 0);
   nir_def *record_offset = nir_imul(b, record_index, record_stride);
   record_offset = nir_iadd_imm(b, record_offset, config->shader_group_handle_size);
   return nir_iadd(b, nir_u2u64(b, record_base), record_offset);
}

struct vt_nir_rt_result
vt_nir_build_traversal(nir_builder *b,
                       const struct vt_nir_rt_trace_params *params,
                       const struct vt_nir_rt_config *config)
{
   struct vt_nir_rt_config default_config;
   if (!config) {
      default_config = vt_nir_rt_default_config();
      config = &default_config;
   }

   const struct vt_nir_scratchpad_layout *raygen =
      &config->scratchpad[VT_RT_LAYOUT_RAYGEN];
   nir_def *base =
      vt_nir_scratchpad_addr(b, config, VT_RT_LAYOUT_RAYGEN,
                             raygen->ray_state_base_bytes);

   store_word(b, base, VT_RT_SLOT_STATUS, nir_imm_int(b, VT_RT_STATUS_TRACE_REQUEST));
   store_u64_words(b, base, VT_RT_SLOT_ACCEL_LO, params->accel_struct);
   store_word(b, base, VT_RT_SLOT_FLAGS, params->flags);
   store_word(b, base, VT_RT_SLOT_CULL_MASK, params->cull_mask);
   store_word(b, base, VT_RT_SLOT_SBT_OFFSET, params->sbt_offset);
   store_word(b, base, VT_RT_SLOT_SBT_STRIDE, params->sbt_stride);
   store_word(b, base, VT_RT_SLOT_MISS_INDEX, params->miss_index);
   store_vec3(b, base, VT_RT_SLOT_ORIGIN_X, params->origin);
   store_word(b, base, VT_RT_SLOT_TMIN, params->tmin);
   store_vec3(b, base, VT_RT_SLOT_DIRECTION_X, params->direction);
   store_word(b, base, VT_RT_SLOT_TMAX, params->tmax);
   store_u64_words(b, base, VT_RT_SLOT_PAYLOAD_PTR_LO, params->payload);

   nir_def *status = nir_ventus_rt_traverse(b, base);
   nir_def *hit_t = load_word(b, base, VT_RT_SLOT_HIT_T);
   nir_def *sbt_index = load_word(b, base, VT_RT_SLOT_SBT_INDEX);
   nir_def *candidate_non_opaque_triangle =
      nir_ieq_imm(b, status, VT_RT_TRAVERSAL_CANDIDATE_NON_OPAQUE_TRIANGLE);
   nir_def *candidate_procedural_aabb =
      nir_ieq_imm(b, status, VT_RT_TRAVERSAL_CANDIDATE_PROCEDURAL_AABB);
   nir_def *is_candidate =
      nir_ior(b, candidate_non_opaque_triangle, candidate_procedural_aabb);
   nir_def *candidate_instance_sbt_record_offset =
      vt_nir_load_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                            VT_RT_HIT_RECORD_INSTANCE_SBT_RECORD_OFFSET,
                                            32);
   nir_def *committed_instance_sbt_record_offset =
      vt_nir_load_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                  VT_RT_HIT_RECORD_INSTANCE_SBT_RECORD_OFFSET, 32);
   nir_def *instance_sbt_record_offset =
      nir_bcsel(b, is_candidate, candidate_instance_sbt_record_offset,
                committed_instance_sbt_record_offset);
   nir_def *candidate_geometry_index =
      vt_nir_load_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                            VT_RT_HIT_RECORD_GEOMETRY_ID, 32);
   nir_def *committed_geometry_index =
      vt_nir_load_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                  VT_RT_HIT_RECORD_GEOMETRY_ID, 32);
   nir_def *geometry_index =
      nir_bcsel(b, is_candidate, candidate_geometry_index, committed_geometry_index);
   nir_def *hit_shader_record_ptr =
      build_hit_shader_record_ptr(b, config, params, params->sbt_base,
                                  instance_sbt_record_offset, geometry_index);
   nir_def *miss_shader_record_ptr = build_miss_shader_record_ptr(b, config, params);
   nir_def *complete_hit = nir_ieq_imm(b, status, VT_RT_TRAVERSAL_COMPLETE_HIT);
   nir_def *complete_miss = nir_ieq_imm(b, status,
                                        VT_RT_TRAVERSAL_COMPLETE_MISS);
   nir_def *shader_record_ptr =
      nir_bcsel(b, is_candidate, hit_shader_record_ptr,
                nir_bcsel(b, complete_hit, hit_shader_record_ptr,
                          nir_bcsel(b, complete_miss, miss_shader_record_ptr,
                                    nir_imm_int64(b, 0))));
   nir_def *opaque = nir_inot(b, candidate_non_opaque_triangle);
   nir_def *need_software_opacity =
      candidate_non_opaque_triangle;

   vt_nir_store_control_word(b, config, VT_RT_LAYOUT_RAYGEN,
                             VT_RT_CONTROL_DONE,
                             nir_ior(b,
                                     nir_ieq_imm(b, status,
                                                VT_RT_TRAVERSAL_COMPLETE_MISS),
                                     nir_ieq_imm(b, status,
                                                VT_RT_TRAVERSAL_COMPLETE_HIT)));
   vt_nir_store_control_word(b, config, VT_RT_LAYOUT_RAYGEN,
                             VT_RT_CONTROL_INCOMPLETE,
                             nir_ior(b,
                                     nir_ieq_imm(b, status,
                                                VT_RT_TRAVERSAL_CANDIDATE_NON_OPAQUE_TRIANGLE),
                                     nir_ieq_imm(b, status,
                                                VT_RT_TRAVERSAL_CANDIDATE_PROCEDURAL_AABB)));
   vt_nir_store_control_word(b, config, VT_RT_LAYOUT_RAYGEN,
                             VT_RT_CONTROL_ACCEPT_HIT,
                             nir_ieq_imm(b, status,
                                         VT_RT_TRAVERSAL_COMPLETE_HIT));
   vt_nir_store_control_word(b, config, VT_RT_LAYOUT_RAYGEN,
                             VT_RT_CONTROL_IGNORE_HIT, nir_imm_false(b));
   vt_nir_store_control_word(b, config, VT_RT_LAYOUT_RAYGEN,
                             VT_RT_CONTROL_TERMINATE_RAY, nir_imm_false(b));
   vt_nir_store_control_word(b, config, VT_RT_LAYOUT_RAYGEN,
                             VT_RT_CONTROL_SKIP_CLOSEST_HIT, nir_imm_false(b));

   vt_nir_store_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                          VT_RT_HIT_RECORD_STATUS, status);
   vt_nir_store_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                          VT_RT_HIT_RECORD_HIT_T, hit_t);
   vt_nir_store_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                          VT_RT_HIT_RECORD_SBT_INDEX, sbt_index);
   vt_nir_store_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                          VT_RT_HIT_RECORD_SHADER_RECORD_PTR_LO,
                                          nir_unpack_64_2x32_split_x(b, shader_record_ptr));
   vt_nir_store_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                          VT_RT_HIT_RECORD_SHADER_RECORD_PTR_HI,
                                          nir_unpack_64_2x32_split_y(b, shader_record_ptr));
   vt_nir_store_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                          VT_RT_HIT_RECORD_FRONT_FACE, nir_imm_true(b));
   vt_nir_store_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                          VT_RT_HIT_RECORD_OPAQUE, opaque);
   vt_nir_store_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                          VT_RT_HIT_RECORD_NEED_SOFTWARE_OPACITY_TEST,
                                          need_software_opacity);

   vt_nir_store_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                VT_RT_HIT_RECORD_STATUS, status);
   vt_nir_store_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                VT_RT_HIT_RECORD_HIT_T, hit_t);
   vt_nir_store_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                VT_RT_HIT_RECORD_SBT_INDEX, sbt_index);
   vt_nir_store_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                VT_RT_HIT_RECORD_SHADER_RECORD_PTR_LO,
                                nir_unpack_64_2x32_split_x(b, shader_record_ptr));
   vt_nir_store_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                VT_RT_HIT_RECORD_SHADER_RECORD_PTR_HI,
                                nir_unpack_64_2x32_split_y(b, shader_record_ptr));
   vt_nir_store_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                VT_RT_HIT_RECORD_FRONT_FACE, nir_imm_true(b));
   vt_nir_store_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                VT_RT_HIT_RECORD_OPAQUE, opaque);
   vt_nir_store_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                VT_RT_HIT_RECORD_NEED_SOFTWARE_OPACITY_TEST,
                                need_software_opacity);

   struct vt_nir_rt_result result = {
      .slot_base = base,
      .status = status,
      .hit = nir_ieq_imm(b, status, VT_RT_TRAVERSAL_COMPLETE_HIT),
      .hit_t = hit_t,
      .sbt_index = sbt_index,
      .instance_sbt_record_offset = instance_sbt_record_offset,
      .geometry_index = geometry_index,
      .shader_record_ptr = shader_record_ptr,
   };
   return result;
}
