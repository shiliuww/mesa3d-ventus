/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "vt_rt_test_common.h"

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
v3_reflect(struct vec3 incident, struct vec3 normal)
{
   return v3_sub(incident, v3_mul(normal, 2.0f * v3_dot(incident, normal)));
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
intersect_sphere(struct vec3 origin, struct vec3 direction, float *hit_t)
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
   if (t < 0.001f) {
      t = (-half_b + root) / a;
      if (t < 0.001f)
         return false;
   }

   *hit_t = t;
   return true;
}

static struct vec3
sky_color(struct vec3 direction)
{
   float t = 0.5f * (direction.y + 1.0f);
   return v3_lerp(v3(0.95f, 0.98f, 1.0f), v3(0.26f, 0.50f, 0.86f), t);
}

static struct vec3
trace_iterative(struct vec3 origin, struct vec3 direction)
{
   struct vec3 accumulated = v3(0.0f, 0.0f, 0.0f);
   float weight = 1.0f;

   for (int depth = 0; depth < 3 && weight > 0.01f; depth++) {
      float previous_weight = weight;
      float hit_t = 0.0f;

      if (intersect_sphere(origin, direction, &hit_t)) {
         struct vec3 hit_pos = v3_add(origin, v3_mul(direction, hit_t));
         struct vec3 normal = v3_norm(hit_pos);
         struct vec3 light = v3_norm(v3(-0.5f, 0.8f, 0.35f));
         float diffuse = clampf(v3_dot(normal, light), 0.0f, 1.0f);
         float rim = powf(clampf(1.0f + v3_dot(direction, normal), 0.0f, 1.0f), 2.0f);
         struct vec3 local = v3_add(v3_mul(v3(0.95f, 0.25f, 0.12f), 0.18f + 0.82f * diffuse),
                                    v3_mul(v3(0.35f, 0.55f, 1.0f), 0.18f * rim));

         accumulated = v3_add(accumulated, v3_mul(local, previous_weight));
         origin = v3_add(hit_pos, v3_mul(normal, 0.002f));
         direction = v3_norm(v3_reflect(direction, normal));
         weight *= 0.32f;
      } else {
         accumulated = v3_add(accumulated, v3_mul(sky_color(direction), previous_weight));
         break;
      }
   }

   return accumulated;
}

static void
write_ppm(const char *path, unsigned width, unsigned height)
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
         struct vec3 color = trace_iterative(origin, direction);
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
   nir_shader *raygen = vt_rt_load_raygen_spirv(argv[1], &options);

   struct vt_rt_intrinsic_counts before;
   vt_rt_count_intrinsics(raygen, &before);
   VT_RT_CHECK(before.trace_ray == 1);

   bool progress = vt_nir_lower_rt_io_monolithic(raygen, &config);
   VT_RT_CHECK(progress);
   progress = vt_nir_lower_rt_abi(raygen, &config);
   VT_RT_CHECK(!progress || raygen->scratch_size >= 0);

   struct vt_rt_intrinsic_counts after;
   vt_rt_count_intrinsics(raygen, &after);
   VT_RT_CHECK(after.trace_ray == 0);
   VT_RT_CHECK(vt_rt_count_loops(raygen) >= 1);
   VT_RT_CHECK(after.jump_break >= 1);
   VT_RT_CHECK(after.ventus_rt_local_store > 0);
   VT_RT_CHECK(after.ventus_rt_local_load > 0);

   write_ppm(argv[2], 160, 96);
   printf("PASS vt_rt_image_smoke spirv_input=1 image=%s width=160 height=96\n",
          argv[2]);

   ralloc_free(raygen);
   glsl_type_singleton_decref();
   return 0;
}
