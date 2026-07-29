/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>

#include "ventus/compiler/vt_nir.h"

#define CHECK(condition) do { \
   if (!(condition)) { \
      fprintf(stderr, "CHECK failed: %s\n", #condition); \
      return 1; \
   } \
} while (0)

int
main(void)
{
   const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_RAYGEN, &options,
                                                   "vt_rt_global_queue_nir");
   const struct vt_nir_rt_submission_mailbox_args mailbox = {
      .base_addr = nir_imm_int64(&b, 0x1000),
      .generation = nir_imm_int(&b, 7),
      .queue_base_addr = nir_imm_int64(&b, 0x4000),
   };
   const struct vt_nir_rt_trace_params params = {
      .accel_struct = nir_imm_int64(&b, 0x2000),
      .flags = nir_imm_int(&b, 1),
      .cull_mask = nir_imm_int(&b, 0xff),
      .sbt_offset = nir_imm_int(&b, 2),
      .sbt_stride = nir_imm_int(&b, 3),
      .miss_index = nir_imm_int(&b, 4),
      .origin = nir_vec3(&b, nir_imm_float(&b, 1.0), nir_imm_float(&b, 2.0),
                         nir_imm_float(&b, 3.0)),
      .tmin = nir_imm_float(&b, 0.1),
      .direction = nir_vec3(&b, nir_imm_float(&b, 4.0), nir_imm_float(&b, 5.0),
                            nir_imm_float(&b, 6.0)),
      .tmax = nir_imm_float(&b, 100.0),
      .payload = nir_imm_int64(&b, 0x3000),
   };

   vt_nir_rt_store_submission_mailbox(&b, &mailbox, &params,
                                       nir_imm_int(&b, 8),
                                       nir_imm_int(&b, 9),
                                       nir_imm_int(&b, 10),
                                       nir_imm_int(&b, 11),
                                       nir_vec3(&b, nir_imm_int(&b, 12),
                                                nir_imm_int(&b, 13),
                                                nir_imm_int(&b, 14)),
                                       nir_imm_int(&b, 9),
                                       nir_imm_int(&b, 10));
   nir_ventus_rt_enqueue(&b, mailbox.base_addr);
   nir_validate_shader(b.shader, "vt_rt_global_queue_nir");

   unsigned stores = 0;
   unsigned enqueue = 0;
   nir_intrinsic_instr *last_store = NULL;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic == nir_intrinsic_store_global) {
               stores++;
               last_store = intrin;
            }
            if (intrin->intrinsic == nir_intrinsic_ventus_rt_enqueue)
               enqueue++;
         }
      }
   }
   /* Record fields plus mailbox magic/version/generation/queue-base/valid. */
   CHECK(stores == VT_RT_GLOBAL_TRACE_READY + 6);
   CHECK(last_store && nir_src_is_const(last_store->src[0]));
   CHECK(nir_src_as_uint(last_store->src[0]) == 1);
   CHECK(enqueue == 1);

   CHECK(vt_nir_lower_rt_enqueue(b.shader));
   nir_validate_shader(b.shader, "vt_rt_global_queue_nir after enqueue lowering");

   enqueue = 0;
   unsigned atomics = 0;
   unsigned queue_stores = 0;
   unsigned stride_left_shifts = 0;
   unsigned stride_right_shifts = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               const nir_op op = nir_instr_as_alu(instr)->op;
               stride_left_shifts += op == nir_op_ishl;
               stride_right_shifts += op == nir_op_ushr;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            enqueue += intrin->intrinsic == nir_intrinsic_ventus_rt_enqueue;
            atomics += intrin->intrinsic == nir_intrinsic_global_atomic;
            queue_stores += intrin->intrinsic == nir_intrinsic_store_global;
         }
      }
   }
   CHECK(enqueue == 0);
   CHECK(atomics >= 4);
   CHECK(queue_stores >= stores + VT_RT_GLOBAL_TRACE_READY);
   CHECK(stride_left_shifts >= 1);
   CHECK(stride_right_shifts >= 1);
   printf("PASS vt_rt_global_queue_nir stores=%u atomics=%u stride_shifts=%u/%u\n",
          queue_stores, atomics, stride_left_shifts, stride_right_shifts);
   ralloc_free(b.shader);
   return 0;
}
