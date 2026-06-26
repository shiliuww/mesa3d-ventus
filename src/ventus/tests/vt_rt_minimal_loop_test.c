/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>

#include "compiler/glsl_types.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
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
build_minimal_raygen(const nir_shader_compiler_options *options)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_RAYGEN, options,
                                     "vt_rt_minimal_raygen");

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *one = nir_imm_int(&b, 1);
   nir_def *accel = nir_imm_int64(&b, 0x1000);
   nir_def *flags = zero;
   nir_def *cull_mask = nir_imm_int(&b, 0xff);
   nir_def *sbt_offset = zero;
   nir_def *sbt_stride = one;
   nir_def *miss_index = zero;
   nir_def *origin =
      nir_vec3(&b, nir_imm_float(&b, 0.0f),
                  nir_imm_float(&b, 0.0f),
                  nir_imm_float(&b, 0.0f));
   nir_def *tmin = nir_imm_float(&b, 0.001f);
   nir_def *direction =
      nir_vec3(&b, nir_imm_float(&b, 0.0f),
                  nir_imm_float(&b, 0.0f),
                  nir_imm_float(&b, -1.0f));
   nir_def *tmax = nir_imm_float(&b, 1000.0f);
   nir_def *payload = nir_undef(&b, 1, 32);

   CHECK(accel && flags && cull_mask && sbt_offset && sbt_stride &&
         miss_index && origin && tmin && direction && tmax && payload);

   nir_intrinsic_instr *trace =
      nir_intrinsic_instr_create(b.shader, nir_intrinsic_trace_ray);
   nir_def *srcs[] = {
      accel, flags, cull_mask, sbt_offset, sbt_stride, miss_index,
      origin, tmin, direction, tmax, payload,
   };
   CHECK(nir_intrinsic_infos[nir_intrinsic_trace_ray].num_srcs == 11);
   for (unsigned i = 0; i < 11; i++) {
      CHECK(srcs[i] != NULL);
      trace->src[i] = nir_src_for_ssa(srcs[i]);
      CHECK(trace->src[i].ssa != NULL);
   }
   nir_builder_instr_insert(&b, &trace->instr);

   return b.shader;
}

static nir_shader *
build_empty_shader(mesa_shader_stage stage, const nir_shader_compiler_options *options,
                   const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(stage, options, "%s", name);
   return b.shader;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   const nir_shader_compiler_options options = {0};
   struct vt_nir_rt_config config = vt_nir_rt_default_config();
   nir_shader *raygen = build_minimal_raygen(&options);
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

   printf("PASS vt_rt_minimal_loop trace_ray_lowered=1 shared_store=%u "
          "shared_load=%u per_lane_scratch=%u\n",
          after.store_shared, after.load_shared, per_lane_scratch);

   ralloc_free(raygen);
   ralloc_free(miss);
   ralloc_free(closest_hit);
   glsl_type_singleton_decref();
   return 0;
}
