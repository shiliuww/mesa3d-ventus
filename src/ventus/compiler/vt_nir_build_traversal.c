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
      .defer_intersection_reports = false,
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
vt_nir_load_active_level(nir_builder *b,
                         const struct vt_nir_rt_config *config,
                         enum vt_rt_shader_layout layout)
{
   return vt_nir_load_trace_meta_field(
      b, config, layout, VT_RT_SLOT_TRACE_META1,
      VT_RT_TRACE_META1_ACTIVE_LEVEL_SHIFT,
      VT_RT_TRACE_META1_ACTIVE_LEVEL_BITS);
}

void
vt_nir_store_active_level(nir_builder *b,
                          const struct vt_nir_rt_config *config,
                          enum vt_rt_shader_layout layout,
                          nir_def *level)
{
   vt_nir_store_trace_meta_field(
      b, config, layout, VT_RT_SLOT_TRACE_META1,
      VT_RT_TRACE_META1_ACTIVE_LEVEL_SHIFT,
      VT_RT_TRACE_META1_ACTIVE_LEVEL_BITS, level);
}

nir_def *
vt_nir_payload_addr(nir_builder *b,
                    const struct vt_nir_rt_config *config,
                    enum vt_rt_shader_layout layout,
                    enum vt_rt_payload_word word)
{
   const struct vt_nir_scratchpad_layout *sp = &config->scratchpad[layout];
   const uint32_t payload_size =
      sp->total_size_bytes - sp->payload_spill_size_bytes -
      sp->payload_base_bytes;
   nir_def *depth = vt_nir_load_active_level(b, config, layout);
   nir_def *depth_offset = nir_imul_imm(b, depth, payload_size);
   return nir_iadd(b,
                   vt_nir_scratchpad_addr(b, config, layout,
                                          sp->payload_base_bytes + word * 4),
                   depth_offset);
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

static nir_def *
vt_nir_hit_kind_addr(nir_builder *b,
                     const struct vt_nir_rt_config *config,
                     enum vt_rt_shader_layout layout, bool candidate)
{
   /* hit_kind is a standalone field-major plane, not word eight of the
    * lane-major record body.  Use the physical field-major word directly so
    * its ordinary AS5 access needs no layout tag. */
   return vt_nir_scratchpad_addr(
      b, config, layout,
      (candidate ? VT_RT_ABI_PDS_CANDIDATE_HIT_KIND_BASE_WORD_COUNT :
                   VT_RT_ABI_PDS_COMMITTED_HIT_KIND_BASE_WORD_COUNT) *
         sizeof(uint32_t));
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
   assert(bit_size == 32);
   return nir_ventus_rt_hit_record_load(
      b, 1, vt_nir_hit_record_addr(b, config, layout, word));
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
   nir_ventus_rt_hit_record_store(
      b, value, vt_nir_hit_record_addr(b, config, layout, word));
}

nir_def *
vt_nir_load_candidate_hit_record_word(nir_builder *b,
                                      const struct vt_nir_rt_config *config,
                                      enum vt_rt_shader_layout layout,
                                      enum vt_rt_hit_record_word word,
                                      uint32_t bit_size)
{
   assert(bit_size == 32);
   return nir_ventus_rt_hit_record_load(
      b, 1, vt_nir_candidate_hit_record_addr(b, config, layout, word));
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
   nir_ventus_rt_hit_record_store(
      b, value, vt_nir_candidate_hit_record_addr(b, config, layout, word));
}

nir_def *
vt_nir_load_committed_hit_kind(nir_builder *b,
                               const struct vt_nir_rt_config *config,
                               enum vt_rt_shader_layout layout)
{
   return nir_ventus_rt_hit_kind_load(
      b, 1, vt_nir_hit_kind_addr(b, config, layout, false));
}

void
vt_nir_store_committed_hit_kind(nir_builder *b,
                                const struct vt_nir_rt_config *config,
                                enum vt_rt_shader_layout layout,
                                nir_def *value)
{
   if (value->bit_size == 1)
      value = nir_b2i32(b, value);
   nir_ventus_rt_hit_kind_store(
      b, value, vt_nir_hit_kind_addr(b, config, layout, false));
}

nir_def *
vt_nir_load_candidate_hit_kind(
   nir_builder *b, const struct vt_nir_rt_config *config,
   enum vt_rt_shader_layout layout)
{
   return nir_ventus_rt_hit_kind_load(
      b, 1, vt_nir_hit_kind_addr(b, config, layout, true));
}

void
vt_nir_store_candidate_hit_kind(
   nir_builder *b, const struct vt_nir_rt_config *config,
   enum vt_rt_shader_layout layout, nir_def *value)
{
   if (value->bit_size == 1)
      value = nir_b2i32(b, value);
   nir_ventus_rt_hit_kind_store(
      b, value, vt_nir_hit_kind_addr(b, config, layout, true));
}

static nir_def *
load_hit_record_meta_field(nir_builder *b, nir_def *meta,
                           uint32_t shift, uint32_t bits)
{
   assert(bits > 0 && bits <= 32);
   if (shift)
      meta = nir_ushr_imm(b, meta, shift);
   return bits == 32 ? meta : nir_iand_imm(b, meta, (1u << bits) - 1);
}

static nir_def *
update_hit_record_meta_field(nir_builder *b, nir_def *old_meta,
                             uint32_t shift, uint32_t bits, nir_def *value)
{
   assert(bits > 0 && bits <= 32);
   const uint32_t value_mask = bits == 32 ? UINT32_MAX : (1u << bits) - 1;
   const uint32_t field_mask = value_mask << shift;
   if (value->bit_size == 1)
      value = nir_b2i32(b, value);
   value = nir_iand_imm(b, value, value_mask);
   if (shift)
      value = nir_ishl_imm(b, value, shift);
   return nir_ior(b, nir_iand_imm(b, old_meta, ~field_mask), value);
}

nir_def *
vt_nir_load_hit_record_meta_field(nir_builder *b,
                                  const struct vt_nir_rt_config *config,
                                  enum vt_rt_shader_layout layout,
                                  enum vt_rt_hit_record_word meta_word,
                                  uint32_t shift, uint32_t bits)
{
   return load_hit_record_meta_field(
      b, vt_nir_load_hit_record_word(b, config, layout, meta_word, 32),
      shift, bits);
}

nir_def *
vt_nir_load_candidate_hit_record_meta_field(
   nir_builder *b, const struct vt_nir_rt_config *config,
   enum vt_rt_shader_layout layout, enum vt_rt_hit_record_word meta_word,
   uint32_t shift, uint32_t bits)
{
   return load_hit_record_meta_field(
      b, vt_nir_load_candidate_hit_record_word(b, config, layout, meta_word, 32),
      shift, bits);
}

void
vt_nir_store_hit_record_meta_field(nir_builder *b,
                                   const struct vt_nir_rt_config *config,
                                   enum vt_rt_shader_layout layout,
                                   enum vt_rt_hit_record_word meta_word,
                                   uint32_t shift, uint32_t bits,
                                   nir_def *value)
{
   nir_def *old_meta =
      vt_nir_load_hit_record_word(b, config, layout, meta_word, 32);
   vt_nir_store_hit_record_word(
      b, config, layout, meta_word,
      update_hit_record_meta_field(b, old_meta, shift, bits, value));
}

void
vt_nir_store_candidate_hit_record_meta_field(
   nir_builder *b, const struct vt_nir_rt_config *config,
   enum vt_rt_shader_layout layout, enum vt_rt_hit_record_word meta_word,
   uint32_t shift, uint32_t bits, nir_def *value)
{
   nir_def *old_meta =
      vt_nir_load_candidate_hit_record_word(b, config, layout, meta_word, 32);
   vt_nir_store_candidate_hit_record_word(
      b, config, layout, meta_word,
      update_hit_record_meta_field(b, old_meta, shift, bits, value));
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

nir_def *
vt_nir_build_hit_shader_record_ptr(
   nir_builder *b, const struct vt_nir_rt_config *config, nir_def *record_base,
   nir_def *record_stride_bytes, nir_def *trace_offset, nir_def *trace_stride,
   nir_def *instance_sbt_record_offset, nir_def *geometry_index)
{
   if (!record_base)
      record_base = nir_imm_int64(b, 0);

   trace_stride = trace_stride ? nir_u2u64(b, trace_stride) : nir_imm_int64(b, 0);
   trace_offset = trace_offset ? nir_u2u64(b, trace_offset) : nir_imm_int64(b, 0);
   nir_def *record_index =
      nir_iadd(b,
               nir_iadd(b, nir_u2u64(b, instance_sbt_record_offset),
                        nir_imul(b, nir_u2u64(b, geometry_index), trace_stride)),
               trace_offset);
   nir_def *record_stride = record_stride_bytes ?
      nir_u2u64(b, record_stride_bytes) : nir_imm_int64(b, 0);
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

   store_u64_words(b, base, VT_RT_SLOT_ACCEL_LO, params->accel_struct);
   vt_nir_store_trace_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_SLOT_TRACE_META0,
      VT_RT_TRACE_META0_FLAGS_SHIFT, VT_RT_TRACE_META0_FLAGS_BITS, params->flags);
   vt_nir_store_trace_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_SLOT_TRACE_META0,
      VT_RT_TRACE_META0_CULL_MASK_SHIFT, VT_RT_TRACE_META0_CULL_MASK_BITS,
      params->cull_mask);
   vt_nir_store_trace_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_SLOT_TRACE_META0,
      VT_RT_TRACE_META0_SBT_OFFSET_SHIFT, VT_RT_TRACE_META0_SBT_OFFSET_BITS,
      params->sbt_offset);
   vt_nir_store_trace_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_SLOT_TRACE_META0,
      VT_RT_TRACE_META0_SBT_STRIDE_SHIFT, VT_RT_TRACE_META0_SBT_STRIDE_BITS,
      params->sbt_stride);
   vt_nir_store_trace_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_SLOT_TRACE_META1,
      VT_RT_TRACE_META1_MISS_INDEX_SHIFT, VT_RT_TRACE_META1_MISS_INDEX_BITS,
      params->miss_index);
   store_vec3(b, base, VT_RT_SLOT_ORIGIN_X, params->origin);
   store_word(b, base, VT_RT_SLOT_TMIN, params->tmin);
   store_vec3(b, base, VT_RT_SLOT_DIRECTION_X, params->direction);
   store_word(b, base, VT_RT_SLOT_TMAX, params->tmax);
   store_u64_words(b, base, VT_RT_SLOT_PAYLOAD_PTR_LO, params->payload);
   /* Publish the slot only after its complete ray header is visible. */
   store_word(b, base, VT_RT_SLOT_STATUS,
              nir_imm_int(b, VT_RT_SLOT_STATUS_TRACE_REQUEST));

   /* The fixed PDS header itself starts at byte offset zero.  The RTcore
    * receives the issuing warp's PDS thread-ID base and derives the selected
    * lane's header as pdsWarpTidBase + lane. */
   nir_def *pds_warp_tid_base = nir_ventus_rt_pds_warp_tid_base(b);
   nir_def *status = nir_ventus_rt_traverse(b, pds_warp_tid_base);
   nir_def *candidate_non_opaque_triangle =
      nir_ieq_imm(b, status, VT_RT_TRAVERSAL_CANDIDATE_NON_OPAQUE_TRIANGLE);
   nir_def *candidate_procedural_aabb =
      nir_ieq_imm(b, status, VT_RT_TRAVERSAL_CANDIDATE_PROCEDURAL_AABB);
   nir_def *is_candidate =
      nir_ior(b, candidate_non_opaque_triangle, candidate_procedural_aabb);
   nir_def *candidate_hit_t =
      vt_nir_load_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                            VT_RT_HIT_RECORD_HIT_T, 32);
   nir_def *committed_hit_t = vt_nir_load_hit_record_word(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_HIT_T, 32);
   nir_def *hit_t = nir_bcsel(b, is_candidate, candidate_hit_t, committed_hit_t);
   nir_def *candidate_instance_sbt_record_offset =
      vt_nir_load_candidate_hit_record_meta_field(
         b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META1,
         VT_RT_HIT_RECORD_META1_INSTANCE_SBT_RECORD_OFFSET_SHIFT,
         VT_RT_HIT_RECORD_META1_INSTANCE_SBT_RECORD_OFFSET_BITS);
   nir_def *committed_instance_sbt_record_offset =
      vt_nir_load_hit_record_meta_field(
         b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META1,
         VT_RT_HIT_RECORD_META1_INSTANCE_SBT_RECORD_OFFSET_SHIFT,
         VT_RT_HIT_RECORD_META1_INSTANCE_SBT_RECORD_OFFSET_BITS);
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
   /* shader_record_ptr is derived when needed, never stored in the PDS ABI. */
   nir_def *hit_shader_record_ptr =
      vt_nir_build_hit_shader_record_ptr(
         b, config, params->sbt_base, params->sbt_stride_bytes,
         params->sbt_offset, params->sbt_stride,
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
   nir_def *hit_record_valid = nir_ior(b, complete_hit, is_candidate);
   vt_nir_store_candidate_hit_record_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META0,
      VT_RT_HIT_RECORD_META0_VALID_SHIFT, 1, hit_record_valid);
   vt_nir_store_candidate_hit_record_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META0,
      VT_RT_HIT_RECORD_META0_STATUS_SHIFT, VT_RT_HIT_RECORD_META0_STATUS_BITS,
      status);
   vt_nir_store_candidate_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                          VT_RT_HIT_RECORD_HIT_T, hit_t);
   vt_nir_store_candidate_hit_record_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META0,
      VT_RT_HIT_RECORD_META0_FRONT_FACE_SHIFT, 1, nir_imm_true(b));
   nir_push_if(b, complete_hit);
   vt_nir_store_hit_record_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META0,
      VT_RT_HIT_RECORD_META0_VALID_SHIFT, 1, nir_imm_true(b));
   vt_nir_store_hit_record_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META0,
      VT_RT_HIT_RECORD_META0_STATUS_SHIFT, VT_RT_HIT_RECORD_META0_STATUS_BITS,
      status);
   vt_nir_store_hit_record_word(b, config, VT_RT_LAYOUT_RAYGEN,
                                VT_RT_HIT_RECORD_HIT_T, hit_t);
   vt_nir_store_hit_record_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META0,
      VT_RT_HIT_RECORD_META0_FRONT_FACE_SHIFT, 1, nir_imm_true(b));
   nir_push_else(b, NULL);
   nir_push_if(b, complete_miss);
   vt_nir_store_hit_record_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META0,
      VT_RT_HIT_RECORD_META0_VALID_SHIFT, 1, nir_imm_false(b));
   vt_nir_store_hit_record_meta_field(
      b, config, VT_RT_LAYOUT_RAYGEN, VT_RT_HIT_RECORD_META0,
      VT_RT_HIT_RECORD_META0_STATUS_SHIFT, VT_RT_HIT_RECORD_META0_STATUS_BITS,
      status);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);

   struct vt_nir_rt_result result = {
      .slot_base = base,
      .status = status,
      .hit = nir_ieq_imm(b, status, VT_RT_TRAVERSAL_COMPLETE_HIT),
      .hit_t = hit_t,
      .instance_sbt_record_offset = instance_sbt_record_offset,
      .geometry_index = geometry_index,
      .hit_shader_record_ptr = hit_shader_record_ptr,
      .miss_shader_record_ptr = miss_shader_record_ptr,
      .shader_record_ptr = shader_record_ptr,
   };
   return result;
}
