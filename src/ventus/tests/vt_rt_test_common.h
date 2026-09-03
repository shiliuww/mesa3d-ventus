/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VT_RT_TEST_COMMON_H
#define VT_RT_TEST_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "compiler/glsl_types.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "spirv/nir_spirv.h"
#include "ventus/compiler/vt_nir.h"

#define VT_RT_CHECK(cond)                                                       \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "CHECK failed: %s at %s:%d\n", #cond, __FILE__,      \
                 __LINE__);                                                    \
         abort();                                                              \
      }                                                                        \
   } while (0)

struct vt_rt_intrinsic_counts {
   unsigned trace_ray;
   unsigned jump_break;
   unsigned load_shared;
   unsigned store_shared;
   unsigned ventus_rt_load_slot;
   unsigned ventus_rt_store_slot;
   unsigned ventus_rt_local_load;
   unsigned ventus_rt_local_store;
   unsigned ventus_rt_hit_record_load;
   unsigned ventus_rt_hit_record_store;
   unsigned ventus_rt_hit_kind_load;
   unsigned ventus_rt_hit_kind_store;
   unsigned load_deref;
   unsigned store_deref;
   unsigned load_shader_record_ptr;
   unsigned load_ray_t_max;
   unsigned load_primitive_id;
   unsigned load_instance_id;
   unsigned load_ray_geometry_index;
   unsigned load_ray_hit_kind;
   unsigned load_front_face;
   unsigned load_ray_t_min;
   unsigned load_ray_flags;
   unsigned load_cull_mask;
   unsigned report_ray_intersection;
   unsigned ignore_ray_intersection;
   unsigned terminate_ray;
   unsigned ventus_rt_warp_first_tid;
   unsigned ventus_rt_traverse;
   unsigned ventus_rt_release;
   unsigned ventus_rt_enqueue;
};

static inline unsigned
vt_rt_count_loops_in_cf_list(struct exec_list *list)
{
   unsigned loops = 0;

   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         loops += vt_rt_count_loops_in_cf_list(&nif->then_list);
         loops += vt_rt_count_loops_in_cf_list(&nif->else_list);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         loops++;
         loops += vt_rt_count_loops_in_cf_list(&loop->body);
         loops += vt_rt_count_loops_in_cf_list(&loop->continue_list);
         break;
      }
      default:
         break;
      }
   }

   return loops;
}

static inline unsigned
vt_rt_count_loops(nir_shader *shader)
{
   unsigned loops = 0;

   nir_foreach_function_impl(impl, shader)
      loops += vt_rt_count_loops_in_cf_list(&impl->body);

   return loops;
}

static inline bool
vt_rt_shader_has_imul_any(nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;

            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op != nir_op_imul)
               continue;
            return true;
         }
      }
   }

   return false;
}

static inline bool
vt_rt_all_releases_are_outside_loops(nir_shader *shader)
{
   unsigned releases = 0;
   unsigned outside_loop_releases = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != nir_intrinsic_ventus_rt_release)
               continue;

            releases++;
            bool inside_loop = false;
            for (nir_cf_node *parent = block->cf_node.parent; parent;
                 parent = parent->parent) {
               if (parent->type == nir_cf_node_loop) {
                  inside_loop = true;
                  break;
               }
            }
            if (!inside_loop)
               outside_loop_releases++;
         }
      }
   }

   return releases > 0 && releases == outside_loop_releases;
}

/* A raygen release must publish TERMINATED in the same block first.  This
 * guards the lifecycle ordering without treating compiler private spills as
 * RT-owned PDS state. */
static inline bool
vt_rt_releases_follow_terminated_store(nir_shader *shader)
{
   bool saw_release = false;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         bool saw_terminated_store = false;
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic == nir_intrinsic_ventus_rt_local_store &&
                nir_src_is_const(intrin->src[0]) &&
                nir_src_as_uint(intrin->src[0]) == VT_RT_SLOT_STATUS_TERMINATED)
               saw_terminated_store = true;
            if (intrin->intrinsic != nir_intrinsic_ventus_rt_release)
               continue;
            if (!saw_terminated_store)
               return false;
            saw_release = true;
         }
      }
   }
   return saw_release;
}

static inline void
vt_rt_count_intrinsics(nir_shader *shader, struct vt_rt_intrinsic_counts *counts)
{
   *counts = (struct vt_rt_intrinsic_counts){0};

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_jump) {
               if (nir_instr_as_jump(instr)->type == nir_jump_break)
                  counts->jump_break++;
               continue;
            }

            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            switch (intrin->intrinsic) {
            case nir_intrinsic_trace_ray:
               counts->trace_ray++;
               break;
            case nir_intrinsic_load_shared:
               counts->load_shared++;
               break;
            case nir_intrinsic_store_shared:
               counts->store_shared++;
               break;
            case nir_intrinsic_ventus_rt_load_slot:
               counts->ventus_rt_load_slot++;
               break;
            case nir_intrinsic_ventus_rt_store_slot:
               counts->ventus_rt_store_slot++;
               break;
            case nir_intrinsic_ventus_rt_local_load:
               counts->ventus_rt_local_load++;
               break;
            case nir_intrinsic_ventus_rt_local_store:
               counts->ventus_rt_local_store++;
               break;
            case nir_intrinsic_ventus_rt_hit_record_load:
               counts->ventus_rt_hit_record_load++;
               break;
            case nir_intrinsic_ventus_rt_hit_record_store:
               counts->ventus_rt_hit_record_store++;
               break;
            case nir_intrinsic_ventus_rt_hit_kind_load:
               counts->ventus_rt_hit_kind_load++;
               break;
            case nir_intrinsic_ventus_rt_hit_kind_store:
               counts->ventus_rt_hit_kind_store++;
               break;
            case nir_intrinsic_load_deref:
               counts->load_deref++;
               break;
            case nir_intrinsic_store_deref:
               counts->store_deref++;
               break;
            case nir_intrinsic_load_shader_record_ptr:
               counts->load_shader_record_ptr++;
               break;
            case nir_intrinsic_load_ray_t_max:
               counts->load_ray_t_max++;
               break;
            case nir_intrinsic_load_primitive_id:
               counts->load_primitive_id++;
               break;
            case nir_intrinsic_load_instance_id:
               counts->load_instance_id++;
               break;
            case nir_intrinsic_load_ray_geometry_index:
               counts->load_ray_geometry_index++;
               break;
            case nir_intrinsic_load_ray_hit_kind:
               counts->load_ray_hit_kind++;
               break;
            case nir_intrinsic_load_front_face:
               counts->load_front_face++;
               break;
            case nir_intrinsic_load_ray_t_min:
               counts->load_ray_t_min++;
               break;
            case nir_intrinsic_load_ray_flags:
               counts->load_ray_flags++;
               break;
            case nir_intrinsic_load_cull_mask:
               counts->load_cull_mask++;
               break;
            case nir_intrinsic_report_ray_intersection:
               counts->report_ray_intersection++;
               break;
            case nir_intrinsic_ignore_ray_intersection:
               counts->ignore_ray_intersection++;
               break;
            case nir_intrinsic_terminate_ray:
               counts->terminate_ray++;
               break;
            case nir_intrinsic_ventus_rt_warp_first_tid:
               counts->ventus_rt_warp_first_tid++;
               break;
            case nir_intrinsic_ventus_rt_traverse:
               counts->ventus_rt_traverse++;
               break;
            case nir_intrinsic_ventus_rt_release:
               counts->ventus_rt_release++;
               break;
            case nir_intrinsic_ventus_rt_enqueue:
               counts->ventus_rt_enqueue++;
               break;
            default:
               break;
            }
         }
      }
   }
}

static inline nir_shader *
vt_rt_load_raygen_spirv(const char *path,
                        const nir_shader_compiler_options *options)
{
   FILE *fp = fopen(path, "rb");
   VT_RT_CHECK(fp != NULL);

   VT_RT_CHECK(fseek(fp, 0, SEEK_END) == 0);
   long size = ftell(fp);
   VT_RT_CHECK(size > 0 && (size % 4) == 0);
   VT_RT_CHECK(fseek(fp, 0, SEEK_SET) == 0);

   uint32_t *words = malloc((size_t)size);
   VT_RT_CHECK(words != NULL);
   VT_RT_CHECK(fread(words, 1, (size_t)size, fp) == (size_t)size);
   VT_RT_CHECK(fclose(fp) == 0);

   const struct spirv_to_nir_options spirv_options = {
      .environment = NIR_SPIRV_VULKAN,
      .ubo_addr_format = nir_address_format_vec2_index_32bit_offset,
      .ssbo_addr_format = nir_address_format_vec2_index_32bit_offset,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .push_const_addr_format = nir_address_format_logical,
      .shared_addr_format = nir_address_format_32bit_offset,
      .constant_addr_format = nir_address_format_64bit_global,
   };

   nir_shader *shader =
      spirv_to_nir(words, (size_t)size / 4, NULL, MESA_SHADER_RAYGEN, "main",
                   &spirv_options, options);
   free(words);

   VT_RT_CHECK(shader != NULL);
   VT_RT_CHECK(shader->info.stage == MESA_SHADER_RAYGEN);
   return shader;
}

static inline nir_shader *
vt_rt_build_empty_shader(mesa_shader_stage stage,
                         const nir_shader_compiler_options *options,
                         const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(stage, options, "%s", name);
   return b.shader;
}

#endif
