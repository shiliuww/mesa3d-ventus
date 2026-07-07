/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

#include "compiler/glsl_types.h"
#include "nir/nir_deref.h"

struct vt_lower_rt_var_state {
   const struct vt_nir_rt_config *config;
   nir_variable_mode mode;
   uint32_t base_offset;
   bool require_explicit_location;
};

static enum vt_rt_shader_layout
current_layout(nir_builder *b)
{
   return vt_nir_layout_for_stage(b->shader->info.stage);
}

static nir_def *
load_rt_slot_word(nir_builder *b, const struct vt_nir_rt_config *config,
                  enum vt_rt_shader_layout layout, uint32_t byte_offset)
{
   return nir_ventus_rt_load_slot(
      b, 1, 32, vt_nir_scratchpad_addr(b, config, layout, byte_offset));
}

static nir_def *
load_ray_state_word(nir_builder *b, const struct vt_nir_rt_config *config,
                    enum vt_rt_shader_layout layout, enum vt_rt_slot_word word)
{
   const struct vt_nir_scratchpad_layout *sp = &config->scratchpad[layout];
   return load_rt_slot_word(b, config, layout,
                            sp->ray_state_base_bytes + word * 4);
}

static nir_def *
load_ray_state_vec3(nir_builder *b, const struct vt_nir_rt_config *config,
                    enum vt_rt_shader_layout layout, enum vt_rt_slot_word first_word)
{
   return nir_vec3(b,
                   load_ray_state_word(b, config, layout, first_word),
                   load_ray_state_word(b, config, layout, first_word + 1),
                   load_ray_state_word(b, config, layout, first_word + 2));
}

static void
store_rt_slot_word(nir_builder *b, const struct vt_nir_rt_config *config,
                   enum vt_rt_shader_layout layout, uint32_t byte_offset,
                   nir_def *value)
{
   nir_ventus_rt_store_slot(
      b, value, vt_nir_scratchpad_addr(b, config, layout, byte_offset));
}

static nir_def *
load_rt_var_word(nir_builder *b, const struct vt_lower_rt_var_state *state,
                 uint32_t word)
{
   return load_rt_slot_word(b, state->config, current_layout(b),
                            state->base_offset + word * 4);
}

static void
store_rt_var_word(nir_builder *b, const struct vt_lower_rt_var_state *state,
                  uint32_t word, nir_def *value)
{
   store_rt_slot_word(b, state->config, current_layout(b),
                      state->base_offset + word * 4, value);
}

static bool
lower_rt_var_deref(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   if (intrin->intrinsic != nir_intrinsic_load_deref &&
       intrin->intrinsic != nir_intrinsic_store_deref)
      return false;

   struct vt_lower_rt_var_state *state = data;
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   if (!deref || !nir_deref_mode_is(deref, state->mode))
      return false;
   if (deref->deref_type == nir_deref_type_cast)
      return false;

   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (!var)
      return false;
   if (state->require_explicit_location && !var->data.explicit_location)
      return false;

   b->cursor = nir_after_instr(instr);

   uint32_t location =
      state->base_offset + var->data.driver_location +
      nir_deref_instr_get_const_offset(deref, glsl_get_natural_size_align_bytes);

   if (intrin->intrinsic == nir_intrinsic_load_deref) {
      uint32_t num_components = intrin->def.num_components;
      uint32_t bit_size = intrin->def.bit_size;
      nir_def *components[NIR_MAX_VEC_COMPONENTS];

      for (uint32_t comp = 0; comp < num_components; comp++) {
         uint32_t offset = location + comp * DIV_ROUND_UP(bit_size, 8);
         uint32_t word = offset / 4;
         uint32_t comp_offset = offset % 4;

         if (bit_size == 64) {
            components[comp] =
               nir_pack_64_2x32_split(b, load_rt_var_word(b, state, word),
                                      load_rt_var_word(b, state, word + 1));
         } else if (bit_size == 32) {
            components[comp] = load_rt_var_word(b, state, word);
         } else if (bit_size == 16) {
            components[comp] =
               nir_channel(b, nir_unpack_32_2x16(b,
                  load_rt_var_word(b, state, word)), comp_offset / 2);
         } else if (bit_size == 8) {
            components[comp] =
               nir_channel(b, nir_unpack_bits(b,
                  load_rt_var_word(b, state, word), 8), comp_offset);
         } else {
            assert(bit_size == 1);
            components[comp] = nir_i2b(b, load_rt_var_word(b, state, word));
         }
      }

      nir_def_rewrite_uses(&intrin->def, nir_vec(b, components, num_components));
   } else {
      nir_def *value = intrin->src[1].ssa;
      uint32_t num_components = value->num_components;
      uint32_t bit_size = value->bit_size;

      for (uint32_t comp = 0; comp < num_components; comp++) {
         uint32_t offset = location + comp * DIV_ROUND_UP(bit_size, 8);
         uint32_t word = offset / 4;
         uint32_t comp_offset = offset % 4;
         nir_def *component = nir_channel(b, value, comp);

         if (bit_size == 64) {
            store_rt_var_word(b, state, word,
                              nir_unpack_64_2x32_split_x(b, component));
            store_rt_var_word(b, state, word + 1,
                              nir_unpack_64_2x32_split_y(b, component));
         } else if (bit_size == 32) {
            store_rt_var_word(b, state, word, component);
         } else if (bit_size == 16) {
            nir_def *prev =
               nir_unpack_32_2x16(b, load_rt_var_word(b, state, word));
            nir_def *components[2];
            for (uint32_t i = 0; i < 2; i++) {
               components[i] = i == comp_offset / 2 ? component :
                               nir_channel(b, prev, i);
            }
            store_rt_var_word(b, state, word,
                              nir_pack_32_2x16(b, nir_vec(b, components, 2)));
         } else if (bit_size == 8) {
            nir_def *prev =
               nir_unpack_bits(b, load_rt_var_word(b, state, word), 8);
            nir_def *components[4];
            for (uint32_t i = 0; i < 4; i++) {
               components[i] = i == comp_offset ? component :
                               nir_channel(b, prev, i);
            }
            store_rt_var_word(b, state, word,
                              nir_pack_32_4x8(b, nir_vec(b, components, 4)));
         } else {
            assert(bit_size == 1);
            store_rt_var_word(b, state, word, nir_b2i32(b, component));
         }
      }
   }

   nir_instr_remove(instr);
   return true;
}

static bool
lower_rt_vars(nir_shader *shader, nir_variable_mode mode,
              const struct vt_nir_rt_config *config, uint32_t base_offset,
              bool require_explicit_location)
{
   bool progress = false;

   progress |= nir_lower_vars_to_explicit_types(shader, mode,
                                                glsl_get_natural_size_align_bytes);
   progress |= nir_lower_indirect_derefs_to_if_else_trees(shader, mode,
                                                          UINT32_MAX);

   struct vt_lower_rt_var_state state = {
      .config = config,
      .mode = mode,
      .base_offset = base_offset,
      .require_explicit_location = require_explicit_location,
   };
   progress |= nir_shader_instructions_pass(shader, lower_rt_var_deref,
                                            nir_metadata_control_flow, &state);

   if (progress) {
      nir_remove_dead_derefs(shader);
      nir_remove_dead_variables(shader, mode, NULL);
   }

   return progress;
}

bool
vt_nir_lower_rt_payload(nir_shader *shader,
                        const struct vt_nir_rt_config *config)
{
   struct vt_nir_rt_config default_config;
   if (!config) {
      default_config = vt_nir_rt_default_config();
      config = &default_config;
   }

   enum vt_rt_shader_layout layout = vt_nir_layout_for_stage(shader->info.stage);
   uint32_t payload_base = config->scratchpad[layout].payload_base_bytes;

   bool progress = false;
   if (shader->info.stage == MESA_SHADER_RAYGEN)
      progress |= lower_rt_vars(shader, nir_var_shader_temp, config,
                                payload_base, false);
   progress |= lower_rt_vars(shader, nir_var_function_temp, config,
                             payload_base, true);
   progress |= lower_rt_vars(shader, nir_var_shader_call_data, config,
                             payload_base, false);
   return progress;
}

static nir_def *
load_hit_record_u64(nir_builder *b, const struct vt_nir_rt_config *config,
                    enum vt_rt_shader_layout layout,
                    enum vt_rt_hit_record_word lo_word)
{
   return nir_pack_64_2x32_split(b,
      vt_nir_load_hit_record_word(b, config, layout, lo_word, 32),
      vt_nir_load_hit_record_word(b, config, layout, lo_word + 1, 32));
}

static bool
uses_candidate_hit_record(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_ANY_HIT || stage == MESA_SHADER_INTERSECTION;
}

static nir_def *
load_active_hit_record_word(nir_builder *b, const struct vt_nir_rt_config *config,
                            enum vt_rt_shader_layout layout,
                            enum vt_rt_hit_record_word word,
                            uint32_t bit_size)
{
   if (uses_candidate_hit_record(b->shader->info.stage)) {
      return vt_nir_load_candidate_hit_record_word(b, config, layout, word,
                                                   bit_size);
   }

   return vt_nir_load_hit_record_word(b, config, layout, word, bit_size);
}

static bool
lower_hit_record_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin,
                           void *data)
{
   const struct vt_nir_rt_config *config = data;
   enum vt_rt_shader_layout layout = vt_nir_layout_for_stage(b->shader->info.stage);
   nir_def *replacement = NULL;

   b->cursor = nir_before_instr(&intrin->instr);

   switch (intrin->intrinsic) {
   case nir_intrinsic_load_ray_world_origin:
   case nir_intrinsic_load_ray_object_origin:
      replacement = load_ray_state_vec3(b, config, layout, VT_RT_SLOT_ORIGIN_X);
      break;
   case nir_intrinsic_load_ray_world_direction:
   case nir_intrinsic_load_ray_object_direction:
      replacement = load_ray_state_vec3(b, config, layout,
                                        VT_RT_SLOT_DIRECTION_X);
      break;
   case nir_intrinsic_load_ray_t_max:
      replacement = load_active_hit_record_word(b, config, layout,
                                                VT_RT_HIT_RECORD_HIT_T, 32);
      break;
   case nir_intrinsic_load_ray_t_min:
      replacement = load_ray_state_word(b, config, layout, VT_RT_SLOT_TMIN);
      break;
   case nir_intrinsic_load_ray_flags:
      replacement = load_ray_state_word(b, config, layout, VT_RT_SLOT_FLAGS);
      break;
   case nir_intrinsic_load_cull_mask:
      replacement = load_ray_state_word(b, config, layout, VT_RT_SLOT_CULL_MASK);
      break;
   case nir_intrinsic_load_primitive_id:
      replacement = load_active_hit_record_word(b, config, layout,
                                                VT_RT_HIT_RECORD_PRIMITIVE_ID, 32);
      break;
   case nir_intrinsic_load_instance_id:
      replacement = load_active_hit_record_word(b, config, layout,
                                                VT_RT_HIT_RECORD_INSTANCE_ID, 32);
      break;
   case nir_intrinsic_load_ray_geometry_index:
      replacement = load_active_hit_record_word(b, config, layout,
                                                VT_RT_HIT_RECORD_GEOMETRY_ID, 32);
      break;
   case nir_intrinsic_load_ray_hit_kind:
      replacement = load_active_hit_record_word(b, config, layout,
                                                VT_RT_HIT_RECORD_HIT_KIND, 32);
      break;
   case nir_intrinsic_load_front_face:
      replacement =
         nir_i2b(b, load_active_hit_record_word(b, config, layout,
                                                VT_RT_HIT_RECORD_FRONT_FACE, 32));
      break;
   case nir_intrinsic_load_front_face_fsign:
      replacement =
         nir_bcsel(b, nir_i2b(b, load_active_hit_record_word(b, config, layout,
                                                            VT_RT_HIT_RECORD_FRONT_FACE, 32)),
                   nir_imm_float(b, 1.0f), nir_imm_float(b, -1.0f));
      break;
   default:
      return false;
   }

   nir_def_rewrite_uses(&intrin->def, replacement);
   nir_instr_remove(&intrin->instr);
   return true;
}

bool
vt_nir_lower_rt_hit_record(nir_shader *shader,
                           const struct vt_nir_rt_config *config)
{
   struct vt_nir_rt_config default_config;
   if (!config) {
      default_config = vt_nir_rt_default_config();
      config = &default_config;
   }

   return nir_shader_intrinsics_pass(shader, lower_hit_record_intrinsic,
                                     nir_metadata_control_flow,
                                     (void *)config);
}

static bool
lower_rt_control_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin,
                           void *data)
{
   const struct vt_nir_rt_config *config = data;
   enum vt_rt_shader_layout layout = vt_nir_layout_for_stage(b->shader->info.stage);

   b->cursor = nir_before_instr(&intrin->instr);

   switch (intrin->intrinsic) {
   case nir_intrinsic_ignore_ray_intersection:
      vt_nir_store_control_word(b, config, layout, VT_RT_CONTROL_ACCEPT_HIT,
                                nir_imm_false(b));
      vt_nir_store_control_word(b, config, layout, VT_RT_CONTROL_IGNORE_HIT,
                                nir_imm_true(b));
      vt_nir_store_control_word(b, config, layout, VT_RT_CONTROL_TERMINATE_RAY,
                                nir_imm_false(b));
      nir_instr_remove(&intrin->instr);
      return true;
   case nir_intrinsic_terminate_ray:
      vt_nir_store_control_word(b, config, layout, VT_RT_CONTROL_ACCEPT_HIT,
                                nir_imm_true(b));
      vt_nir_store_control_word(b, config, layout, VT_RT_CONTROL_IGNORE_HIT,
                                nir_imm_false(b));
      vt_nir_store_control_word(b, config, layout, VT_RT_CONTROL_TERMINATE_RAY,
                                nir_imm_true(b));
      nir_instr_remove(&intrin->instr);
      return true;
   case nir_intrinsic_report_ray_intersection: {
      nir_def *hit_t = intrin->src[0].ssa;
      nir_def *hit_kind = intrin->src[1].ssa;
      nir_def *current_tmax =
         vt_nir_load_hit_record_word(b, config, layout,
                                     VT_RT_HIT_RECORD_HIT_T, 32);
      nir_def *tmin = load_ray_state_word(b, config, layout, VT_RT_SLOT_TMIN);
      nir_def *accepted =
         nir_iand(b, nir_fge(b, current_tmax, hit_t),
                  nir_fge(b, hit_t, tmin));

      vt_nir_store_control_word(b, config, layout, VT_RT_CONTROL_ACCEPT_HIT,
                                accepted);
      vt_nir_store_control_word(b, config, layout, VT_RT_CONTROL_IGNORE_HIT,
                                nir_inot(b, accepted));
      vt_nir_store_candidate_hit_record_word(b, config, layout,
                                             VT_RT_HIT_RECORD_STATUS,
                                             nir_bcsel(b, accepted,
                                                       nir_imm_int(b, VT_RT_STATUS_HIT),
                                                       nir_imm_int(b, VT_RT_STATUS_IDLE)));
      vt_nir_store_candidate_hit_record_word(b, config, layout,
                                             VT_RT_HIT_RECORD_HIT_T, hit_t);
      vt_nir_store_candidate_hit_record_word(b, config, layout,
                                             VT_RT_HIT_RECORD_HIT_KIND, hit_kind);
      vt_nir_store_candidate_hit_record_word(b, config, layout,
                                             VT_RT_HIT_RECORD_FRONT_FACE,
                                             nir_ieq_imm(b, hit_kind, 0xfe));
      vt_nir_store_candidate_hit_record_word(b, config, layout,
                                             VT_RT_HIT_RECORD_OPAQUE,
                                             nir_imm_false(b));
      vt_nir_store_candidate_hit_record_word(b, config, layout,
                                             VT_RT_HIT_RECORD_NEED_SOFTWARE_OPACITY_TEST,
                                             nir_imm_true(b));

      nir_def_rewrite_uses(&intrin->def, accepted);
      nir_instr_remove(&intrin->instr);
      return true;
   }
   default:
      return false;
   }
}

bool
vt_nir_lower_rt_control(nir_shader *shader,
                        const struct vt_nir_rt_config *config)
{
   struct vt_nir_rt_config default_config;
   if (!config) {
      default_config = vt_nir_rt_default_config();
      config = &default_config;
   }

   bool progress = false;
   enum vt_rt_shader_layout layout = vt_nir_layout_for_stage(shader->info.stage);
   uint32_t hit_attrib_base =
      config->scratchpad[layout].hit_attrib_base_bytes;

   progress |= lower_rt_vars(shader, nir_var_ray_hit_attrib, config,
                             hit_attrib_base, false);
   progress |= nir_shader_intrinsics_pass(shader, lower_rt_control_intrinsic,
                                          nir_metadata_control_flow,
                                          (void *)config);
   return progress;
}

static bool
lower_shader_record_ptr_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin,
                                  void *data)
{
   if (intrin->intrinsic != nir_intrinsic_load_shader_record_ptr)
      return false;

   const struct vt_nir_rt_config *config = data;
   enum vt_rt_shader_layout layout = vt_nir_layout_for_stage(b->shader->info.stage);

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *ptr = load_hit_record_u64(b, config, layout,
                                      VT_RT_HIT_RECORD_SHADER_RECORD_PTR_LO);
   nir_def_rewrite_uses(&intrin->def, ptr);
   nir_instr_remove(&intrin->instr);
   return true;
}

bool
vt_nir_lower_shader_record_ptr(nir_shader *shader,
                               const struct vt_nir_rt_config *config)
{
   struct vt_nir_rt_config default_config;
   if (!config) {
      default_config = vt_nir_rt_default_config();
      config = &default_config;
   }

   return nir_shader_intrinsics_pass(shader, lower_shader_record_ptr_intrinsic,
                                     nir_metadata_control_flow,
                                     (void *)config);
}
