/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>

#include "compiler/glsl_types.h"
#include "nir/nir.h"
#include "spirv/nir_spirv.h"
#include "ventus/compiler/vt_nir.h"

struct intrinsic_counts {
   unsigned trace_ray;
   unsigned load_shared;
   unsigned store_shared;
};

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "CHECK failed: %s at %s:%d\n", #cond, __FILE__,      \
                 __LINE__);                                                    \
         abort();                                                              \
      }                                                                        \
   } while (0)

static void
count_intrinsics(nir_shader *shader, struct intrinsic_counts *counts)
{
   *counts = (struct intrinsic_counts){0};

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
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
            default:
               break;
            }
         }
      }
   }
}

static nir_shader *
load_raygen_spirv(const char *path, const nir_shader_compiler_options *options)
{
   FILE *fp = fopen(path, "rb");
   CHECK(fp != NULL);

   CHECK(fseek(fp, 0, SEEK_END) == 0);
   long size = ftell(fp);
   CHECK(size > 0 && (size % 4) == 0);
   CHECK(fseek(fp, 0, SEEK_SET) == 0);

   uint32_t *words = malloc((size_t)size);
   CHECK(words != NULL);
   CHECK(fread(words, 1, (size_t)size, fp) == (size_t)size);
   CHECK(fclose(fp) == 0);

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

   CHECK(shader != NULL);
   CHECK(shader->info.stage == MESA_SHADER_RAYGEN);
   return shader;
}

static nir_shader *
build_empty_shader(mesa_shader_stage stage, const nir_shader_compiler_options *options,
                   const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(stage, options, "%s", name);
   return b.shader;
}

int
main(int argc, char **argv)
{
   CHECK(argc == 2);
   glsl_type_singleton_init_or_ref();

   const nir_shader_compiler_options options = {0};
   struct vt_nir_rt_config config = vt_nir_rt_default_config();
   nir_shader *raygen = load_raygen_spirv(argv[1], &options);
   nir_shader *miss = build_empty_shader(MESA_SHADER_MISS, &options, "vt_rt_minimal_miss");
   nir_shader *closest_hit =
      build_empty_shader(MESA_SHADER_CLOSEST_HIT, &options, "vt_rt_minimal_closest_hit");

   struct intrinsic_counts before;
   count_intrinsics(raygen, &before);
   CHECK(before.trace_ray == 1);

   bool progress = vt_nir_lower_rt_io_monolithic(raygen, &config);
   CHECK(progress);

   progress = vt_nir_lower_rt_abi(raygen, &config);
   CHECK(!progress || raygen->scratch_size >= 0);

   struct intrinsic_counts after;
   count_intrinsics(raygen, &after);
   CHECK(after.trace_ray == 0);
   CHECK(after.store_shared > 0);
   CHECK(after.load_shared > 0);

   raygen->scratch_size = 48;
   miss->scratch_size = 16;
   closest_hit->scratch_size = 80;

   nir_shader *pipeline[] = {raygen, miss, closest_hit};
   struct vt_nir_pipeline_scratch_info scratch_info;
   uint32_t per_lane_scratch =
      vt_nir_pipeline_compute_scratch(pipeline, 3, &scratch_info);

   CHECK(per_lane_scratch == 80);
   CHECK(scratch_info.shader_scratch_size[VT_RT_LAYOUT_RAYGEN] == 48);
   CHECK(scratch_info.shader_scratch_size[VT_RT_LAYOUT_MISS] == 16);
   CHECK(scratch_info.shader_scratch_size[VT_RT_LAYOUT_CLOSEST_HIT] == 80);

   printf("PASS vt_rt_minimal_loop spirv_input=1 trace_ray_lowered=1 shared_store=%u "
          "shared_load=%u per_lane_scratch=%u\n",
          after.store_shared, after.load_shared, per_lane_scratch);

   ralloc_free(raygen);
   ralloc_free(miss);
   ralloc_free(closest_hit);
   glsl_type_singleton_decref();
   return 0;
}
