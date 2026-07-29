/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

/*
 * This is deliberately separate from vt_nir_lower_rt_io.c.  That file owns
 * the proven monolithic/PDS compatibility path.  This file owns only the
 * PDS-free producer boundary: a terminal primary traceRayEXT becomes a
 * global mailbox publication followed by rt_enqueue.
 *
 * A worker-side callback or a trace with live code afterwards requires the
 * queue/CPS resume ABI.  Do not lower either case here until that ABI exists.
 */

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

static bool
is_terminal_primary_trace(nir_intrinsic_instr *intrin)
{
   nir_instr *next = nir_instr_next(&intrin->instr);
   return next && next->type == nir_instr_type_jump &&
          (nir_instr_as_jump(next)->type == nir_jump_return ||
           nir_instr_as_jump(next)->type == nir_jump_halt);
}

static unsigned
count_trace_ray_intrinsics(nir_shader *shader)
{
   unsigned count = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_trace_ray)
               count++;
         }
      }
   }

   return count;
}

static bool
lower_terminal_primary_trace(nir_builder *b, nir_intrinsic_instr *intrin,
                             void *data)
{
   (void)data;
   if (intrin->intrinsic != nir_intrinsic_trace_ray ||
       !is_terminal_primary_trace(intrin))
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   /* The dummy if gives the inserted halt a structured CFG home.  Dead-CF
    * cleanup removes the unreachable original return afterwards. */
   nir_push_if(b, nir_imm_true(b));
   struct vt_nir_rt_trace_params params = {
      .accel_struct = intrin->src[0].ssa,
      .flags = intrin->src[1].ssa,
      .cull_mask = intrin->src[2].ssa,
      .sbt_base = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HIT_SBT_ADDR, 64),
      .sbt_stride_bytes =
         load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_HIT_SBT_STRIDE, 64),
      .miss_sbt_base = load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_MISS_SBT_ADDR, 64),
      .miss_sbt_stride_bytes =
         load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_MISS_SBT_STRIDE, 64),
      .sbt_offset = intrin->src[3].ssa,
      .sbt_stride = intrin->src[4].ssa,
      .miss_index = intrin->src[5].ssa,
      .origin = intrin->src[6].ssa,
      .tmin = intrin->src[7].ssa,
      .direction = intrin->src[8].ssa,
      .tmax = intrin->src[9].ssa,
      .payload = intrin->src[10].ssa,
   };

   /* The queue owns logical ray references.  A mailbox, in contrast, belongs
    * to the physical worker/lane that is currently executing this invocation. */
   nir_def *id = nir_load_ray_launch_id(b);
   nir_def *mailbox_pool = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_BASE, 64);
   nir_def *mailbox_stride = load_dispatch_arg(
      b, VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_STRIDE, 32);
   /* The bridge launches raygen with local-x == 32.  Masking global x gives
    * the physical lane even for a partial final workgroup.  worker_id is the
    * globally unique resident (SM, hardware-warp) slot supplied by the
    * execution backend. */
   nir_intrinsic_instr *worker_intrin = nir_intrinsic_instr_create(
      b->shader, nir_intrinsic_ventus_rt_worker_id);
   worker_intrin->def.num_components = 1;
   worker_intrin->def.bit_size = 32;
   nir_builder_instr_insert(b, &worker_intrin->instr);
   nir_def *lane_id = nir_iand_imm(b, nir_channel(b, id, 0),
                                   VT_RT_GLOBAL_WARP_SIZE - 1u);
   nir_def *mailbox_addr = vt_nir_rt_submission_mailbox_addr(
      b, mailbox_pool, mailbox_stride, &worker_intrin->def, lane_id,
      VT_RT_SUBMISSION_MAILBOX_PRIMARY);
   struct vt_nir_rt_submission_mailbox_args mailbox = {
      .base_addr = mailbox_addr,
      .generation = load_dispatch_arg(
         b, VT_NIR_RT_DISPATCH_ARG_SUBMISSION_MAILBOX_GENERATION, 32),
      .queue_base_addr = load_dispatch_arg(
         b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_QUEUE_BASE, 64),
   };
   vt_nir_rt_store_submission_mailbox(
      b, &mailbox, &params, nir_imm_int(b, 0),
      load_dispatch_arg(b, VT_NIR_RT_DISPATCH_ARG_GLOBAL_QUEUE_GENERATION, 32),
      nir_imm_int(b, 0));

   /* RV32 instruction selection consumes the checked low half.  The mailbox
    * itself retains the full u64 ABI address for the future RV64 path. */
   nir_ventus_rt_enqueue(b, nir_u2u32(b, mailbox_addr));
   nir_jump(b, nir_jump_halt);
   nir_pop_if(b, NULL);
   return true;
}

bool
vt_nir_lower_rt_global_raygen(nir_shader *shader,
                              const struct vt_nir_rt_config *config)
{
   if (!shader || shader->info.stage != MESA_SHADER_RAYGEN || !config ||
       config->execution_model != VT_RT_EXECUTION_WAVEFRONT_GLOBAL_WORKER)
      return false;

   /* The global worker contract has no callback/CPS execution path yet. */
   if (config->enable_monolithic_shader_callbacks || config->emit_closest_hit ||
       config->emit_miss || config->emit_any_hit || config->emit_intersection)
      return false;

   /* A second traceRayEXT needs a parent continuation and/or a child record;
    * refusing it is safer than silently dropping work after the first halt. */
   if (count_trace_ray_intrinsics(shader) != 1)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (!impl)
      return false;

   bool progress = nir_shader_intrinsics_pass(
      shader, lower_terminal_primary_trace, nir_metadata_none, (void *)config);
   if (!progress)
      return false;

   nir_opt_dead_cf(shader);
   nir_index_blocks(impl);
   nir_index_ssa_defs(impl);
   return true;
}
