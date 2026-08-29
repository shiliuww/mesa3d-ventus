/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

#include "compiler/glsl_types.h"
#include "nir/nir_deref.h"

#include <limits.h>
#include <stdlib.h>

#define VT_RT_MAX_PAYLOAD_LOCATIONS 32

struct vt_payload_location {
   int location;
   uint32_t offset;
   uint32_t size;
   uint32_t align;
};

struct vt_payload_location_map {
   struct vt_payload_location entries[VT_RT_MAX_PAYLOAD_LOCATIONS];
   uint32_t count;
};

/* nir_lower_vars_to_explicit_types assigns driver locations for ordinary
 * private storage and clears data.location.  A trace_ray dereference remains
 * in the callback until it is inlined into raygen, so retain the Vulkan
 * payload location on that variable for the later per-depth-frame binding and
 * Header payload-pointer emission.  The actual load/store dereferences are
 * still lowered before this metadata is restored. */
struct vt_payload_var_location {
   nir_variable *var;
   int location;
};

static struct vt_payload_var_location *
save_payload_var_locations(nir_shader *shader, uint32_t *count)
{
   *count = 0;
   nir_foreach_variable_in_shader(var, shader) {
      if ((var->data.mode == nir_var_shader_temp ||
           var->data.mode == nir_var_shader_call_data) &&
          var->data.explicit_location && var->data.location >= 0)
         (*count)++;
   }

   if (!*count)
      return NULL;

   struct vt_payload_var_location *locations =
      calloc(*count, sizeof(*locations));
   if (!locations) {
      *count = 0;
      return NULL;
   }

   uint32_t index = 0;
   nir_foreach_variable_in_shader(var, shader) {
      if ((var->data.mode != nir_var_shader_temp &&
           var->data.mode != nir_var_shader_call_data) ||
          !var->data.explicit_location || var->data.location < 0)
         continue;
      locations[index++] = (struct vt_payload_var_location) {
         .var = var,
         .location = var->data.location,
      };
   }
   assert(index == *count);
   return locations;
}

static void
restore_payload_var_locations(const struct vt_payload_var_location *locations,
                              uint32_t count)
{
   for (uint32_t i = 0; i < count; i++) {
      locations[i].var->data.location = locations[i].location;
      locations[i].var->data.explicit_location = true;
   }
}

struct vt_lower_rt_var_state {
   const struct vt_nir_rt_config *config;
   nir_variable_mode mode;
   uint32_t base_offset;
   bool require_explicit_location;
   const struct vt_payload_location_map *payload_locations;
};

static bool
is_barycentric_hit_attribute(const struct vt_lower_rt_var_state *state,
                             nir_variable *var, uint32_t location)
{
   if (!state || state->mode != nir_var_ray_hit_attrib ||
       !state->config->scratchpad[VT_RT_LAYOUT_ANY_HIT].hit_attrib_barycentrics)
      return false;

   unsigned size = 0, align = 0;
   glsl_get_natural_size_align_bytes(var->type, &size, &align);
   return location == 0 && size == 8 && align >= 4;
}

static enum vt_rt_shader_layout
current_layout(nir_builder *b)
{
   return vt_nir_layout_for_stage(b->shader->info.stage);
}

static nir_def *
load_dispatch_arg(nir_builder *b, uint32_t offset, unsigned bit_size)
{
   nir_intrinsic_instr *intrin =
      nir_intrinsic_instr_create(b->shader, nir_intrinsic_ventus_rt_dispatch_arg);
   intrin->src[0] = nir_src_for_ssa(nir_imm_int(b, offset));

   nir_def_init(&intrin->instr, &intrin->def, 1, bit_size);
   nir_builder_instr_insert(b, &intrin->instr);
   return &intrin->def;
}

static nir_def *
load_rt_slot_word(nir_builder *b, const struct vt_nir_rt_config *config,
                  enum vt_rt_shader_layout layout, uint32_t byte_offset)
{
   return nir_ventus_rt_load_slot(
      b, 1, 32, vt_nir_scratchpad_addr(b, config, layout, byte_offset));
}

nir_def *
vt_nir_load_ray_state_word(nir_builder *b,
                           const struct vt_nir_rt_config *config,
                           enum vt_rt_shader_layout layout,
                           enum vt_rt_slot_word word)
{
   const struct vt_nir_scratchpad_layout *sp = &config->scratchpad[layout];
   return load_rt_slot_word(b, config, layout,
                            sp->ray_state_base_bytes + word * 4);
}

static nir_def *
load_packed_u32_field(nir_builder *b, nir_def *word, uint32_t shift,
                      uint32_t bits)
{
   assert(bits > 0 && bits <= 32);
   if (shift)
      word = nir_ushr_imm(b, word, shift);
   return bits == 32 ? word : nir_iand_imm(b, word, (1u << bits) - 1);
}

static nir_def *
update_packed_u32_field(nir_builder *b, nir_def *old_word, uint32_t shift,
                        uint32_t bits, nir_def *value)
{
   assert(bits > 0 && bits <= 32);
   const uint32_t value_mask = bits == 32 ? UINT32_MAX : (1u << bits) - 1;
   const uint32_t field_mask = value_mask << shift;
   if (value->bit_size == 1)
      value = nir_b2i32(b, value);
   value = nir_iand_imm(b, value, value_mask);
   if (shift)
      value = nir_ishl_imm(b, value, shift);
   return nir_ior(b, nir_iand_imm(b, old_word, ~field_mask), value);
}

nir_def *
vt_nir_load_trace_meta_field(nir_builder *b,
                             const struct vt_nir_rt_config *config,
                             enum vt_rt_shader_layout layout,
                             enum vt_rt_slot_word meta_word,
                             uint32_t shift, uint32_t bits)
{
   return load_packed_u32_field(
      b, vt_nir_load_ray_state_word(b, config, layout, meta_word), shift, bits);
}

void
vt_nir_store_trace_meta_field(nir_builder *b,
                              const struct vt_nir_rt_config *config,
                              enum vt_rt_shader_layout layout,
                              enum vt_rt_slot_word meta_word,
                              uint32_t shift, uint32_t bits, nir_def *value)
{
   nir_def *old_word = vt_nir_load_ray_state_word(b, config, layout, meta_word);
   nir_ventus_rt_store_slot(
      b, update_packed_u32_field(b, old_word, shift, bits, value),
      vt_nir_scratchpad_addr(
         b, config, layout,
         config->scratchpad[layout].ray_state_base_bytes + meta_word * 4));
}

static nir_def *
load_ray_state_vec3(nir_builder *b, const struct vt_nir_rt_config *config,
                    enum vt_rt_shader_layout layout, enum vt_rt_slot_word first_word)
{
   return nir_vec3(b,
                   vt_nir_load_ray_state_word(b, config, layout, first_word),
                   vt_nir_load_ray_state_word(b, config, layout, first_word + 1),
                   vt_nir_load_ray_state_word(b, config, layout, first_word + 2));
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
                 uint32_t word, bool depth_indexed)
{
   enum vt_rt_shader_layout layout = current_layout(b);
   const struct vt_nir_scratchpad_layout *sp =
      &state->config->scratchpad[layout];
   const uint32_t payload_size =
      sp->total_size_bytes - sp->payload_spill_size_bytes -
      sp->payload_base_bytes;
   nir_def *addr = vt_nir_scratchpad_addr(
      b, state->config, layout, state->base_offset + word * 4);
   /* Megakernel payload interface variables use a distinct P-byte frame for
    * every static TraceRay depth.  This makes the PDS location
    * payload_base + depth * P + location * 16 the sole payload storage; no
    * private-pointer mirror is involved. */
   if (depth_indexed && payload_size) {
      nir_def *depth = vt_nir_load_active_level(b, state->config, layout);
      addr = nir_iadd(b, addr,
                      nir_imul_imm(b, depth, payload_size));
   }
   return nir_ventus_rt_load_slot(b, 1, 32, addr);
}

static void
store_rt_var_word(nir_builder *b, const struct vt_lower_rt_var_state *state,
                  uint32_t word, nir_def *value, bool depth_indexed)
{
   enum vt_rt_shader_layout layout = current_layout(b);
   const struct vt_nir_scratchpad_layout *sp =
      &state->config->scratchpad[layout];
   const uint32_t payload_size =
      sp->total_size_bytes - sp->payload_spill_size_bytes -
      sp->payload_base_bytes;
   nir_def *addr = vt_nir_scratchpad_addr(
      b, state->config, layout, state->base_offset + word * 4);
   if (depth_indexed && payload_size) {
      nir_def *depth = vt_nir_load_active_level(b, state->config, layout);
      addr = nir_iadd(b, addr,
                      nir_imul_imm(b, depth, payload_size));
   }
   nir_ventus_rt_store_slot(b, value, addr);
}

static struct vt_payload_location *
find_payload_location(struct vt_payload_location_map *map, int location)
{
   for (uint32_t i = 0; i < map->count; ++i) {
      if (map->entries[i].location == location)
         return &map->entries[i];
   }

   return NULL;
}

static const struct vt_payload_location *
lookup_payload_location(const struct vt_payload_location_map *map, int location)
{
   for (uint32_t i = 0; map && i < map->count; ++i) {
      if (map->entries[i].location == location)
         return &map->entries[i];
   }

   return NULL;
}

static void
add_payload_location_var(struct vt_payload_location_map *map,
                         nir_variable *var)
{
   if (!var->data.explicit_location || var->data.location < 0)
      return;

   struct vt_payload_location *entry =
      find_payload_location(map, var->data.location);
   if (!entry) {
      if (map->count >= VT_RT_MAX_PAYLOAD_LOCATIONS) {
         assert(!"too many RT payload locations");
         return;
      }

      entry = &map->entries[map->count++];
      *entry = (struct vt_payload_location) {
         .location = var->data.location,
      };
   }

   unsigned size = 0;
   unsigned align = 0;
   glsl_get_natural_size_align_bytes(var->type, &size, &align);
   entry->size = MAX2(entry->size, size);
   entry->align = MAX2(entry->align, align);
}

static void
collect_payload_locations_from_list(struct vt_payload_location_map *map,
                                    struct exec_list *vars,
                                    nir_variable_mode mode)
{
   nir_foreach_variable_in_list(var, vars) {
      if (var->data.mode == mode)
         add_payload_location_var(map, var);
   }
}

static void
assign_payload_location_offsets(struct vt_payload_location_map *map)
{
   for (uint32_t i = 0; i < map->count; ++i) {
      if (map->entries[i].location >= 0)
         map->entries[i].offset =
            map->entries[i].location * VT_RT_ABI_ALIGNMENT_BYTES;
   }
}

static struct vt_payload_location_map
build_payload_location_map(nir_shader *shader, nir_variable_mode modes)
{
   struct vt_payload_location_map map = {0};

   if (modes & nir_var_shader_temp)
      collect_payload_locations_from_list(&map, &shader->variables,
                                          nir_var_shader_temp);
   if (modes & nir_var_shader_call_data)
      collect_payload_locations_from_list(&map, &shader->variables,
                                          nir_var_shader_call_data);

   if (modes & nir_var_function_temp) {
      nir_foreach_function_impl(impl, shader) {
         collect_payload_locations_from_list(&map, &impl->locals,
                                             nir_var_function_temp);
      }
   }

   assign_payload_location_offsets(&map);
   return map;
}

bool
vt_nir_rt_compute_pipeline_payload_size(nir_shader **shaders,
                                        unsigned shader_count,
                                        uint32_t *payload_size_bytes)
{
   if (!payload_size_bytes)
      return false;

   uint64_t end = 0;
   for (unsigned s = 0; s < shader_count; ++s) {
      nir_shader *shader = shaders[s];
      if (!shader)
         continue;

      const nir_variable_mode payload_modes =
         nir_var_function_temp | nir_var_shader_call_data | nir_var_shader_temp;
      struct vt_payload_location_map map =
         build_payload_location_map(shader, payload_modes);
      for (uint32_t i = 0; i < map.count; ++i) {
         const struct vt_payload_location *entry = &map.entries[i];
         const uint64_t location_offset =
            (uint64_t)entry->location * VT_RT_ABI_ALIGNMENT_BYTES;
         const uint64_t entry_end = location_offset + entry->size;
         if (entry_end > UINT32_MAX)
            return false;
         end = MAX2(end, entry_end);
      }
   }

   if (end > UINT32_MAX - (VT_RT_ABI_ALIGNMENT_BYTES - 1))
      return false;
   *payload_size_bytes = ALIGN_POT((uint32_t)end,
                                   VT_RT_ABI_ALIGNMENT_BYTES);
   return true;
}

static uint32_t
rt_var_driver_location(const struct vt_lower_rt_var_state *state,
                       nir_variable *var)
{
   if (var->data.explicit_location) {
      const struct vt_payload_location *entry =
         lookup_payload_location(state->payload_locations, var->data.location);
      if (entry)
         return entry->offset;
   }

   return var->data.driver_location;
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
   /* A callback payload without a SPIR-V Location decoration is still a
    * valid single payload interface in the global CPS ABI.  Its driver
    * location is the canonical fallback; only callers that explicitly
    * require a Location must reject it. */
   if (state->require_explicit_location && var->data.location < 0)
      return false;

   b->cursor = nir_after_instr(instr);

   uint32_t location =
      rt_var_driver_location(state, var) +
      nir_deref_instr_get_const_offset(deref, glsl_get_natural_size_align_bytes);
   const bool compiler_private =
      state->mode == nir_var_shader_temp && !var->data.explicit_location;
   const bool megakernel_payload =
      state->config->execution_model == VT_RT_EXECUTION_ITERATIVE_RAYGEN_LOOP &&
      (state->mode == nir_var_shader_call_data ||
       (state->mode == nir_var_shader_temp && var->data.explicit_location));
   const bool depth_indexed = compiler_private || megakernel_payload;

   const bool barycentrics = is_barycentric_hit_attribute(state, var, location);
   const bool candidate = b->shader->info.stage == MESA_SHADER_ANY_HIT ||
                         b->shader->info.stage == MESA_SHADER_INTERSECTION;
   if (barycentrics && intrin->def.bit_size == 32 &&
       intrin->def.num_components == 2 &&
       nir_deref_instr_get_const_offset(deref,
                                        glsl_get_natural_size_align_bytes) == 0) {
      nir_def *components[2];
      for (unsigned comp = 0; comp < 2; comp++) {
         const enum vt_rt_hit_record_word word =
            (enum vt_rt_hit_record_word)(VT_RT_HIT_RECORD_BARYCENTRICS_X + comp);
         components[comp] = candidate ?
            vt_nir_load_candidate_hit_record_word(b, state->config,
                                                  current_layout(b), word, 32) :
            vt_nir_load_hit_record_word(b, state->config, current_layout(b),
                                        word, 32);
      }
      if (intrin->intrinsic == nir_intrinsic_load_deref) {
         nir_def_rewrite_uses(&intrin->def, nir_vec(b, components, 2));
      } else {
         nir_def *value = intrin->src[1].ssa;
         for (unsigned comp = 0; comp < 2; comp++) {
            const enum vt_rt_hit_record_word word =
               (enum vt_rt_hit_record_word)(VT_RT_HIT_RECORD_BARYCENTRICS_X + comp);
            if (candidate)
               vt_nir_store_candidate_hit_record_word(
                  b, state->config, current_layout(b), word,
                  nir_channel(b, value, comp));
            else
               vt_nir_store_hit_record_word(
                  b, state->config, current_layout(b), word,
                  nir_channel(b, value, comp));
         }
      }
      nir_instr_remove(instr);
      return true;
   }

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
               nir_pack_64_2x32_split(b,
                  load_rt_var_word(b, state, word, depth_indexed),
                  load_rt_var_word(b, state, word + 1, depth_indexed));
         } else if (bit_size == 32) {
            components[comp] = load_rt_var_word(b, state, word, depth_indexed);
         } else if (bit_size == 16) {
            components[comp] =
               nir_channel(b, nir_unpack_32_2x16(b,
                  load_rt_var_word(b, state, word, depth_indexed)),
                           comp_offset / 2);
         } else if (bit_size == 8) {
            components[comp] =
               nir_channel(b, nir_unpack_bits(b,
                  load_rt_var_word(b, state, word, depth_indexed), 8),
                           comp_offset);
         } else {
            assert(bit_size == 1);
            components[comp] = nir_i2b(b, load_rt_var_word(b, state, word,
                                                              depth_indexed));
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
                              nir_unpack_64_2x32_split_x(b, component),
                              depth_indexed);
            store_rt_var_word(b, state, word + 1,
                              nir_unpack_64_2x32_split_y(b, component),
                              depth_indexed);
         } else if (bit_size == 32) {
            store_rt_var_word(b, state, word, component, depth_indexed);
         } else if (bit_size == 16) {
            nir_def *prev =
               nir_unpack_32_2x16(b, load_rt_var_word(b, state, word,
                                                      depth_indexed));
            nir_def *components[2];
            for (uint32_t i = 0; i < 2; i++) {
               components[i] = i == comp_offset / 2 ? component :
                               nir_channel(b, prev, i);
            }
            store_rt_var_word(b, state, word,
                              nir_pack_32_2x16(b, nir_vec(b, components, 2)),
                              depth_indexed);
         } else if (bit_size == 8) {
            nir_def *prev =
               nir_unpack_bits(b, load_rt_var_word(b, state, word,
                                                   depth_indexed), 8);
            nir_def *components[4];
            for (uint32_t i = 0; i < 4; i++) {
               components[i] = i == comp_offset ? component :
                               nir_channel(b, prev, i);
            }
            store_rt_var_word(b, state, word,
                              nir_pack_32_4x8(b, nir_vec(b, components, 4)),
                              depth_indexed);
         } else {
            assert(bit_size == 1);
            store_rt_var_word(b, state, word, nir_b2i32(b, component),
                              depth_indexed);
         }
      }
   }

   nir_instr_remove(instr);
   return true;
}

static bool
lower_rt_vars(nir_shader *shader, nir_variable_mode mode,
              const struct vt_nir_rt_config *config, uint32_t base_offset,
              bool require_explicit_location,
              const struct vt_payload_location_map *payload_locations)
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
      .payload_locations = payload_locations,
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
   nir_variable_mode payload_modes =
      nir_var_function_temp | nir_var_shader_call_data | nir_var_shader_temp;
   struct vt_payload_location_map payload_locations =
      build_payload_location_map(shader, payload_modes);
   uint32_t saved_location_count = 0;
   struct vt_payload_var_location *saved_locations =
      save_payload_var_locations(shader, &saved_location_count);

   bool progress = false;
   /* Global CPS frames preserve ray payload state without allocating any PDS.
    * Raygen payload interfaces may be shader_temp variables without an
    * explicit Vulkan location, so use their lowered driver location here.
    * Keep ordinary function temporaries as CPS SSA spills. */
   const bool global_cps_payload =
      config->execution_model == VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER;
   progress |= lower_rt_vars(shader, nir_var_shader_temp, config,
                             payload_base, !global_cps_payload,
                             &payload_locations);
   progress |= lower_rt_vars(shader, nir_var_function_temp, config,
                             payload_base, true, &payload_locations);
   progress |= lower_rt_vars(shader, nir_var_shader_call_data, config,
                             payload_base, false, &payload_locations);
   restore_payload_var_locations(saved_locations, saved_location_count);
   free(saved_locations);
   return progress;
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

static nir_def *
load_active_hit_record_meta_field(
   nir_builder *b, const struct vt_nir_rt_config *config,
   enum vt_rt_shader_layout layout, enum vt_rt_hit_record_word meta_word,
   uint32_t shift, uint32_t bits)
{
   if (uses_candidate_hit_record(b->shader->info.stage)) {
      return vt_nir_load_candidate_hit_record_meta_field(
         b, config, layout, meta_word, shift, bits);
   }

   return vt_nir_load_hit_record_meta_field(b, config, layout, meta_word,
                                            shift, bits);
}

/* A procedural report is evaluated before any committed hit exists.  Reading
 * committed HIT_T unconditionally makes the first report compare against the
 * zero-filled record and reject every positive intersection. */
static nir_def *
load_committed_tmax(nir_builder *b, const struct vt_nir_rt_config *config,
                   enum vt_rt_shader_layout layout)
{
   nir_def *committed_valid = nir_ieq_imm(
      b, vt_nir_load_hit_record_meta_field(
            b, config, layout, VT_RT_HIT_RECORD_META0,
            VT_RT_HIT_RECORD_META0_VALID_SHIFT, 1),
      VT_RT_HIT_RECORD_STATUS_VALID);
   nir_def *committed_t = vt_nir_load_hit_record_word(
      b, config, layout, VT_RT_HIT_RECORD_HIT_T, 32);
   nir_def *ray_tmax = vt_nir_load_ray_state_word(
      b, config, layout, VT_RT_SLOT_TMAX);
   return nir_bcsel(b, committed_valid, committed_t, ray_tmax);
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
      replacement = vt_nir_load_ray_state_word(b, config, layout, VT_RT_SLOT_TMIN);
      break;
   case nir_intrinsic_load_ray_flags:
      replacement = vt_nir_load_trace_meta_field(
         b, config, layout, VT_RT_SLOT_TRACE_META0,
         VT_RT_TRACE_META0_FLAGS_SHIFT, VT_RT_TRACE_META0_FLAGS_BITS);
      break;
   case nir_intrinsic_load_cull_mask:
      replacement = vt_nir_load_trace_meta_field(
         b, config, layout, VT_RT_SLOT_TRACE_META0,
         VT_RT_TRACE_META0_CULL_MASK_SHIFT, VT_RT_TRACE_META0_CULL_MASK_BITS);
      break;
   case nir_intrinsic_load_primitive_id:
      replacement = load_active_hit_record_word(b, config, layout,
                                                VT_RT_HIT_RECORD_PRIMITIVE_ID, 32);
      break;
   case nir_intrinsic_load_instance_id:
      replacement = load_active_hit_record_word(b, config, layout,
                                                VT_RT_HIT_RECORD_INSTANCE_ID, 32);
      break;
   case nir_intrinsic_load_ray_instance_custom_index:
      replacement = load_active_hit_record_meta_field(
         b, config, layout, VT_RT_HIT_RECORD_META0,
         VT_RT_HIT_RECORD_META0_INSTANCE_CUSTOM_INDEX_SHIFT,
         VT_RT_HIT_RECORD_META0_INSTANCE_CUSTOM_INDEX_BITS);
      break;
   case nir_intrinsic_load_ray_geometry_index:
      replacement = load_active_hit_record_word(b, config, layout,
                                                VT_RT_HIT_RECORD_GEOMETRY_ID, 32);
      break;
   case nir_intrinsic_load_ray_hit_kind:
      replacement = uses_candidate_hit_record(b->shader->info.stage) ?
         vt_nir_load_candidate_hit_kind(b, config, layout) :
         vt_nir_load_committed_hit_kind(b, config, layout);
      break;
   case nir_intrinsic_load_front_face:
      replacement =
         nir_i2b(b, load_active_hit_record_meta_field(
            b, config, layout, VT_RT_HIT_RECORD_META0,
            VT_RT_HIT_RECORD_META0_FRONT_FACE_SHIFT, 1));
      break;
   case nir_intrinsic_load_front_face_fsign:
      replacement =
         nir_bcsel(b, nir_i2b(b, load_active_hit_record_meta_field(
                      b, config, layout, VT_RT_HIT_RECORD_META0,
                      VT_RT_HIT_RECORD_META0_FRONT_FACE_SHIFT, 1)),
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
      vt_nir_store_control_word(b, config, layout,
                                VT_RT_CONTROL_CALLBACK_DECISION,
                                nir_imm_int(b, VT_RT_CALLBACK_IGNORE));
      nir_instr_remove(&intrin->instr);
      return true;
   case nir_intrinsic_terminate_ray:
      vt_nir_store_control_word(b, config, layout,
                                VT_RT_CONTROL_CALLBACK_DECISION,
                                nir_imm_int(b, VT_RT_CALLBACK_TERMINATE));
      nir_instr_remove(&intrin->instr);
      return true;
   case nir_intrinsic_report_ray_intersection: {
      if (config->defer_intersection_reports &&
          b->shader->info.stage == MESA_SHADER_INTERSECTION)
         return false;
      nir_def *hit_t = intrin->src[0].ssa;
      nir_def *hit_kind = intrin->src[1].ssa;
      nir_def *current_tmax = load_committed_tmax(b, config, layout);
      nir_def *tmin = vt_nir_load_ray_state_word(b, config, layout, VT_RT_SLOT_TMIN);
      nir_def *accepted =
         nir_iand(b, nir_flt(b, hit_t, current_tmax),
                  nir_fge(b, hit_t, tmin));

      vt_nir_store_control_word(b, config, layout,
                                VT_RT_CONTROL_CALLBACK_DECISION,
                                nir_bcsel(b, accepted,
                                          nir_imm_int(b, VT_RT_CALLBACK_ACCEPT),
                                          nir_imm_int(b, VT_RT_CALLBACK_IGNORE)));
      vt_nir_store_candidate_hit_record_meta_field(
         b, config, layout, VT_RT_HIT_RECORD_META0,
         VT_RT_HIT_RECORD_META0_VALID_SHIFT, 1, accepted);
      vt_nir_store_candidate_hit_record_meta_field(
         b, config, layout, VT_RT_HIT_RECORD_META0,
         VT_RT_HIT_RECORD_META0_STATUS_SHIFT,
         VT_RT_HIT_RECORD_META0_STATUS_BITS,
         nir_imm_int(b, VT_RT_TRAVERSAL_CANDIDATE_PROCEDURAL_AABB));
      vt_nir_store_candidate_hit_record_word(b, config, layout,
                                             VT_RT_HIT_RECORD_HIT_T, hit_t);
      vt_nir_store_candidate_hit_kind(b, config, layout, hit_kind);
      vt_nir_store_candidate_hit_record_meta_field(
         b, config, layout, VT_RT_HIT_RECORD_META0,
         VT_RT_HIT_RECORD_META0_FRONT_FACE_SHIFT, 1,
         nir_ieq_imm(b, hit_kind, 0xfe));

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
   const bool candidate_attributes =
      shader->info.stage == MESA_SHADER_ANY_HIT ||
      shader->info.stage == MESA_SHADER_INTERSECTION;
   uint32_t hit_attrib_base = candidate_attributes ?
      config->scratchpad[layout].hit_attrib_base_bytes :
      config->scratchpad[layout].committed_hit_attrib_base_bytes;

   progress |= lower_rt_vars(shader, nir_var_ray_hit_attrib, config,
                             hit_attrib_base, false, NULL);
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
   nir_def *ptr;
   if (layout == VT_RT_LAYOUT_RAYGEN) {
      ptr = nir_iadd_imm(
         b, load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_RAYGEN_SBT_ADDR, 64),
         config->shader_group_handle_size);
   } else if (layout == VT_RT_LAYOUT_MISS) {
      nir_def *miss_index = nir_u2u64(b, vt_nir_load_trace_meta_field(
         b, config, layout, VT_RT_SLOT_TRACE_META1,
         VT_RT_TRACE_META1_MISS_INDEX_SHIFT,
         VT_RT_TRACE_META1_MISS_INDEX_BITS));
      nir_def *miss_stride =
         load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_MISS_SBT_STRIDE, 64);
      nir_def *miss_offset = nir_iadd_imm(
         b, nir_imul(b, miss_index, miss_stride),
         config->shader_group_handle_size);
      ptr = nir_iadd(b,
                     load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_MISS_SBT_ADDR,
                                       64),
                     miss_offset);
   } else {
      nir_def *instance_sbt_record_offset =
         load_active_hit_record_meta_field(
            b, config, layout, VT_RT_HIT_RECORD_META1,
            VT_RT_HIT_RECORD_META1_INSTANCE_SBT_RECORD_OFFSET_SHIFT,
            VT_RT_HIT_RECORD_META1_INSTANCE_SBT_RECORD_OFFSET_BITS);
      nir_def *geometry_index = load_active_hit_record_word(
         b, config, layout, VT_RT_HIT_RECORD_GEOMETRY_ID, 32);
      ptr = vt_nir_build_hit_shader_record_ptr(
         b, config,
         load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HIT_SBT_ADDR, 64),
         load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HIT_SBT_STRIDE, 64),
         vt_nir_load_trace_meta_field(
            b, config, layout, VT_RT_SLOT_TRACE_META0,
            VT_RT_TRACE_META0_SBT_OFFSET_SHIFT,
            VT_RT_TRACE_META0_SBT_OFFSET_BITS),
         vt_nir_load_trace_meta_field(
            b, config, layout, VT_RT_SLOT_TRACE_META0,
            VT_RT_TRACE_META0_SBT_STRIDE_SHIFT,
            VT_RT_TRACE_META0_SBT_STRIDE_BITS),
         instance_sbt_record_offset, geometry_index);
   }
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
