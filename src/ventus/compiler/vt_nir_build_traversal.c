/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

struct vt_nir_rt_config
vt_nir_rt_default_config(void)
{
   struct vt_nir_rt_config config = {
      .slot_stride_bytes = 256,
      .slot_base_bytes = 0,
      .enable_monolithic_shader_callbacks = true,
      .emit_closest_hit = NULL,
      .emit_miss = NULL,
      .callback_data = NULL,
   };
   return config;
}

static nir_def *
slot_base(nir_builder *b, const struct vt_nir_rt_config *config)
{
   nir_def *launch_id = nir_load_ray_launch_id(b);
   nir_def *launch_size = nir_load_ray_launch_size(b);
   nir_def *x = nir_channel(b, launch_id, 0);
   nir_def *y = nir_channel(b, launch_id, 1);
   nir_def *width = nir_channel(b, launch_size, 0);
   nir_def *linear_id = nir_iadd(b, x, nir_imul(b, y, width));

   return nir_iadd_imm(b, nir_imul_imm(b, linear_id, config->slot_stride_bytes),
                       config->slot_base_bytes);
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

   nir_def *base = slot_base(b, config);

   store_word(b, base, VT_RT_SLOT_STATUS, nir_imm_int(b, 1));
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
      .hit = nir_ine_imm(b, status, 0),
      .hit_t = hit_t,
      .sbt_index = sbt_index,
   };
   return result;
}
