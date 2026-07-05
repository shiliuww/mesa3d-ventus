/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "vt_rt_test_common.h"

#define VT_RT_FLAGGED_RAY_FLAGS 0x16u
#define VT_RT_FLAGGED_CULL_MASK 0x7fu
#define VT_RT_FLAGGED_SBT_OFFSET 3u
#define VT_RT_FLAGGED_SBT_STRIDE 5u
#define VT_RT_FLAGGED_MISS_INDEX 2u

struct vec3 {
   float x;
   float y;
   float z;
};

static struct vec3
v3(float x, float y, float z)
{
   return (struct vec3){x, y, z};
}

static struct vec3
v3_add(struct vec3 a, struct vec3 b)
{
   return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static struct vec3
v3_sub(struct vec3 a, struct vec3 b)
{
   return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static struct vec3
v3_mul(struct vec3 a, float s)
{
   return v3(a.x * s, a.y * s, a.z * s);
}

static float
v3_dot(struct vec3 a, struct vec3 b)
{
   return a.x * b.x + a.y * b.y + a.z * b.z;
}

static struct vec3
v3_norm(struct vec3 a)
{
   float len = sqrtf(v3_dot(a, a));
   if (len <= 0.0f)
      return v3(0.0f, 0.0f, 0.0f);
   return v3_mul(a, 1.0f / len);
}

static struct vec3
v3_lerp(struct vec3 a, struct vec3 b, float t)
{
   return v3_add(v3_mul(a, 1.0f - t), v3_mul(b, t));
}

static float
clampf(float x, float lo, float hi)
{
   if (x < lo)
      return lo;
   if (x > hi)
      return hi;
   return x;
}

static uint8_t
to_srgb8(float x)
{
   x = clampf(x, 0.0f, 1.0f);
   return (uint8_t)(255.0f * sqrtf(x) + 0.5f);
}

static bool
intersect_sphere(struct vec3 origin, struct vec3 direction, float *hit_t,
                 bool *front_face)
{
   struct vec3 center = v3(0.0f, 0.0f, 0.0f);
   float radius = 0.72f;
   struct vec3 oc = v3_sub(origin, center);
   float a = v3_dot(direction, direction);
   float half_b = v3_dot(oc, direction);
   float c = v3_dot(oc, oc) - radius * radius;
   float discriminant = half_b * half_b - a * c;
   if (discriminant < 0.0f)
      return false;

   float root = sqrtf(discriminant);
   float t = (-half_b - root) / a;
   if (t < 0.125f) {
      t = (-half_b + root) / a;
      if (t < 0.125f)
         return false;
   }

   struct vec3 normal = v3_norm(v3_add(origin, v3_mul(direction, t)));
   *front_face = v3_dot(direction, normal) < 0.0f;
   *hit_t = t;
   return true;
}

static struct vec3
sky_color(struct vec3 direction)
{
   float t = 0.5f * (direction.y + 1.0f);
   return v3_lerp(v3(0.95f, 0.98f, 1.0f), v3(0.20f, 0.42f, 0.82f), t);
}

static bool
trace_ray_has_flagged_params(nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != nir_intrinsic_trace_ray)
               continue;

            return nir_src_is_const(intrin->src[1]) &&
                   nir_src_as_uint(intrin->src[1]) == VT_RT_FLAGGED_RAY_FLAGS &&
                   nir_src_is_const(intrin->src[2]) &&
                   nir_src_as_uint(intrin->src[2]) == VT_RT_FLAGGED_CULL_MASK &&
                   nir_src_is_const(intrin->src[3]) &&
                   nir_src_as_uint(intrin->src[3]) == VT_RT_FLAGGED_SBT_OFFSET &&
                   nir_src_is_const(intrin->src[4]) &&
                   nir_src_as_uint(intrin->src[4]) == VT_RT_FLAGGED_SBT_STRIDE &&
                   nir_src_is_const(intrin->src[5]) &&
                   nir_src_as_uint(intrin->src[5]) == VT_RT_FLAGGED_MISS_INDEX;
         }
      }
   }

   return false;
}

static nir_shader *
build_flag_probe_any_hit(const nir_shader_compiler_options *options)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_ANY_HIT, options,
                                     "vt_rt_flag_probe_any_hit");

   nir_def *front_face = nir_load_front_face(&b, 1);
   nir_def *ray_flags = nir_load_ray_flags(&b);
   nir_def *cull_mask = nir_load_cull_mask(&b);
   nir_def *hit_kind = nir_load_ray_hit_kind(&b);

   nir_variable *sink =
      nir_variable_create(b.shader, nir_var_shader_temp,
                          glsl_uvec4_type(), "flag_probe_sink");
   nir_store_var(&b, sink,
                 nir_vec4(&b, nir_b2i32(&b, front_face), ray_flags, cull_mask,
                          hit_kind),
                 0xf);
   nir_ignore_ray_intersection(&b);

   return b.shader;
}

static nir_shader *
build_flag_probe_intersection(const nir_shader_compiler_options *options)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_INTERSECTION, options,
                                     "vt_rt_flag_probe_intersection");

   nir_def *t = nir_fadd_imm(&b, nir_load_ray_t_min(&b), 0.5f);
   nir_def *accepted =
      nir_report_ray_intersection(&b, 1, t, nir_imm_int(&b, 0xff));

   nir_variable *sink =
      nir_variable_create(b.shader, nir_var_shader_temp,
                          glsl_bool_type(), "intersection_probe_sink");
   nir_store_var(&b, sink, accepted, 0x1);

   return b.shader;
}

static struct vec3
trace_flagged_pixel(struct vec3 origin, struct vec3 direction,
                    unsigned *software_opacity_pixels)
{
   float hit_t = 0.0f;
   bool front_face = false;
   if (!intersect_sphere(origin, direction, &hit_t, &front_face))
      return sky_color(direction);

   if ((VT_RT_FLAGGED_RAY_FLAGS & 0x10u) && !front_face)
      return sky_color(direction);

   struct vec3 hit_pos = v3_add(origin, v3_mul(direction, hit_t));
   struct vec3 normal = v3_norm(hit_pos);
   float stripes = sinf(24.0f * hit_pos.x) * sinf(18.0f * hit_pos.y);
   bool need_sw_opacity_test =
      (VT_RT_FLAGGED_RAY_FLAGS & 0x2u) && stripes > 0.18f;

   if (need_sw_opacity_test) {
      (*software_opacity_pixels)++;
      return v3_lerp(sky_color(direction), v3(0.90f, 0.95f, 1.0f), 0.35f);
   }

   struct vec3 light = v3_norm(v3(-0.5f, 0.8f, 0.35f));
   float diffuse = clampf(v3_dot(normal, light), 0.0f, 1.0f);
   float face_term = front_face ? 1.0f : 0.35f;
   struct vec3 base = v3(0.95f, 0.28f, 0.10f);
   struct vec3 flag_tint = v3(0.12f, 0.34f, 0.95f);
   return v3_add(v3_mul(base, face_term * (0.20f + 0.80f * diffuse)),
                 v3_mul(flag_tint, 0.10f));
}

static void
write_flagged_ppm(const char *path, unsigned width, unsigned height,
                  unsigned *software_opacity_pixels)
{
   FILE *fp = fopen(path, "wb");
   VT_RT_CHECK(fp != NULL);
   VT_RT_CHECK(fprintf(fp, "P6\n%u %u\n255\n", width, height) > 0);

   for (unsigned y = 0; y < height; y++) {
      for (unsigned x = 0; x < width; x++) {
         float u = ((float)x + 0.5f) / (float)width;
         float v = ((float)y + 0.5f) / (float)height;
         float aspect = (float)width / (float)height;
         struct vec3 origin = v3(0.0f, 0.0f, 2.6f);
         struct vec3 direction =
            v3_norm(v3((2.0f * u - 1.0f) * aspect, 1.0f - 2.0f * v, -1.65f));
         struct vec3 color =
            trace_flagged_pixel(origin, direction, software_opacity_pixels);
         uint8_t rgb[3] = {
            to_srgb8(color.x),
            to_srgb8(color.y),
            to_srgb8(color.z),
         };
         VT_RT_CHECK(fwrite(rgb, 1, sizeof(rgb), fp) == sizeof(rgb));
      }
   }

   VT_RT_CHECK(fclose(fp) == 0);
}

int
main(int argc, char **argv)
{
   VT_RT_CHECK(argc == 3);
   glsl_type_singleton_init_or_ref();

   const nir_shader_compiler_options options = {0};
   struct vt_nir_rt_config config = vt_nir_rt_default_config();
   config.hit_sbt_base_addr = 0x200000000ull;
   config.hit_sbt_record_stride = 96;
   config.shader_group_handle_size = 32;

   nir_shader *raygen = vt_rt_load_raygen_spirv(argv[1], &options);
   VT_RT_CHECK(trace_ray_has_flagged_params(raygen));

   struct vt_rt_intrinsic_counts before;
   vt_rt_count_intrinsics(raygen, &before);
   VT_RT_CHECK(before.trace_ray == 1);

   bool progress = vt_nir_lower_rt_io_monolithic(raygen, &config);
   VT_RT_CHECK(progress);
   progress = vt_nir_lower_rt_abi(raygen, &config);
   VT_RT_CHECK(!progress || raygen->scratch_size >= 0);
   VT_RT_CHECK(raygen->scratch_size >= vt_nir_rt_default_abi_layout().total_size_bytes);

   struct vt_rt_intrinsic_counts after;
   vt_rt_count_intrinsics(raygen, &after);
   VT_RT_CHECK(after.trace_ray == 0);
   VT_RT_CHECK(after.ventus_rt_store_slot > before.ventus_rt_store_slot);
   VT_RT_CHECK(after.ventus_rt_load_slot > before.ventus_rt_load_slot);
   VT_RT_CHECK(vt_rt_shader_has_imul_any(raygen));

   nir_shader *any_hit = build_flag_probe_any_hit(&options);
   struct vt_rt_intrinsic_counts any_before;
   vt_rt_count_intrinsics(any_hit, &any_before);
   VT_RT_CHECK(any_before.load_front_face == 1);
   VT_RT_CHECK(any_before.load_ray_flags == 1);
   VT_RT_CHECK(any_before.load_cull_mask == 1);
   VT_RT_CHECK(any_before.ignore_ray_intersection == 1);

   progress = vt_nir_lower_rt_abi(any_hit, &config);
   VT_RT_CHECK(progress);
   struct vt_rt_intrinsic_counts any_after;
   vt_rt_count_intrinsics(any_hit, &any_after);
   VT_RT_CHECK(any_after.load_front_face == 0);
   VT_RT_CHECK(any_after.load_ray_flags == 0);
   VT_RT_CHECK(any_after.load_cull_mask == 0);
   VT_RT_CHECK(any_after.ignore_ray_intersection == 0);
   VT_RT_CHECK(any_after.ventus_rt_load_slot > any_before.ventus_rt_load_slot);
   VT_RT_CHECK(any_after.ventus_rt_store_slot > any_before.ventus_rt_store_slot);

   nir_shader *intersection = build_flag_probe_intersection(&options);
   struct vt_rt_intrinsic_counts intersection_before;
   vt_rt_count_intrinsics(intersection, &intersection_before);
   VT_RT_CHECK(intersection_before.load_ray_t_min == 1);
   VT_RT_CHECK(intersection_before.report_ray_intersection == 1);

   progress = vt_nir_lower_rt_abi(intersection, &config);
   VT_RT_CHECK(progress);
   struct vt_rt_intrinsic_counts intersection_after;
   vt_rt_count_intrinsics(intersection, &intersection_after);
   VT_RT_CHECK(intersection_after.load_ray_t_min == 0);
   VT_RT_CHECK(intersection_after.report_ray_intersection == 0);
   VT_RT_CHECK(intersection_after.ventus_rt_load_slot >
               intersection_before.ventus_rt_load_slot);
   VT_RT_CHECK(intersection_after.ventus_rt_store_slot >
               intersection_before.ventus_rt_store_slot);

   unsigned software_opacity_pixels = 0;
   write_flagged_ppm(argv[2], 192, 108, &software_opacity_pixels);
   VT_RT_CHECK(software_opacity_pixels > 0);

   printf("PASS vt_rt_flagged_image_smoke spirv_input=1 flagged_trace=1 "
          "flag_lowering=1 anyhit_flags=1 intersection_report=1 "
          "image=%s width=192 height=108 software_opacity_pixels=%u "
          "ray_flags=0x%x cull_mask=0x%x sbt_offset=%u sbt_stride=%u "
          "miss_index=%u\n",
          argv[2], software_opacity_pixels, VT_RT_FLAGGED_RAY_FLAGS,
          VT_RT_FLAGGED_CULL_MASK, VT_RT_FLAGGED_SBT_OFFSET,
          VT_RT_FLAGGED_SBT_STRIDE, VT_RT_FLAGGED_MISS_INDEX);

   ralloc_free(raygen);
   ralloc_free(any_hit);
   ralloc_free(intersection);
   glsl_type_singleton_decref();
   return 0;
}
