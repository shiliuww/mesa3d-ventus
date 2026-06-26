/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

struct vt_nir_rt_config
vt_nir_rt_default_config(void)
{
   struct vt_nir_rt_config config = {
      .execution_model = VT_RT_EXECUTION_ITERATIVE_RAYGEN_LOOP,
      .scratchpad = {
         [VT_RT_LAYOUT_RAYGEN] = {
            .base_bytes = 0,
            .stride_bytes = 256,
            .ray_state_base_bytes = 0,
            .payload_base_bytes = 128,
         },
         [VT_RT_LAYOUT_CLOSEST_HIT] = {
            .base_bytes = 0,
            .stride_bytes = 256,
            .ray_state_base_bytes = 0,
            .payload_base_bytes = 128,
         },
         [VT_RT_LAYOUT_MISS] = {
            .base_bytes = 0,
            .stride_bytes = 256,
            .ray_state_base_bytes = 0,
            .payload_base_bytes = 128,
         },
      },
      .miss_depth_value = 0x7fffffff,
      .enable_monolithic_shader_callbacks = false,
      .emit_closest_hit = NULL,
      .emit_miss = NULL,
      .callback_data = NULL,
   };
   return config;
}

enum vt_rt_shader_layout
vt_nir_layout_for_stage(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_CLOSEST_HIT:
      return VT_RT_LAYOUT_CLOSEST_HIT;
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
   nir_def *launch_id = nir_load_ray_launch_id(b);
   nir_def *launch_size = nir_load_ray_launch_size(b);
   nir_def *x = nir_channel(b, launch_id, 0);
   nir_def *y = nir_channel(b, launch_id, 1);
   nir_def *width = nir_channel(b, launch_size, 0);
   nir_def *linear_id = nir_iadd(b, x, nir_imul(b, y, width));

   const struct vt_nir_scratchpad_layout *sp = &config->scratchpad[layout];
   return nir_iadd_imm(b, nir_imul_imm(b, linear_id, sp->stride_bytes),
                       sp->base_bytes);
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
vt_nir_load_payload_word(nir_builder *b,
                         const struct vt_nir_rt_config *config,
                         enum vt_rt_shader_layout layout,
                         enum vt_rt_payload_word word,
                         uint32_t bit_size)
{
   return nir_load_shared(b, 1, bit_size,
                          vt_nir_payload_addr(b, config, layout, word),
                          .align_mul = 4);
}

void
vt_nir_store_payload_word(nir_builder *b,
                          const struct vt_nir_rt_config *config,
                          enum vt_rt_shader_layout layout,
                          enum vt_rt_payload_word word,
                          nir_def *value)
{
   nir_store_shared(b, value, vt_nir_payload_addr(b, config, layout, word),
                    .align_mul = 4);
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

   nir_store_shared(b, value, slot_offset(b, base, word), .align_mul = 4);
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

   nir_def *status =
      nir_load_shared(b, 1, 32, slot_offset(b, base, VT_RT_SLOT_STATUS),
                      .align_mul = 4);
   nir_def *hit_t =
      nir_load_shared(b, 1, 32, slot_offset(b, base, VT_RT_SLOT_HIT_T),
                      .align_mul = 4);
   nir_def *sbt_index =
      nir_load_shared(b, 1, 32, slot_offset(b, base, VT_RT_SLOT_SBT_INDEX),
                      .align_mul = 4);

   struct vt_nir_rt_result result = {
      .hit = nir_ieq_imm(b, status, VT_RT_STATUS_HIT),
      .hit_t = hit_t,
      .sbt_index = sbt_index,
   };
   return result;
}
