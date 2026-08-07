/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VENTUS_NIR_TO_LLVM_H
#define VENTUS_NIR_TO_LLVM_H

#include <stdbool.h>
#include <stdint.h>

#include "nir/nir.h"

#ifdef __cplusplus
namespace llvm {
class Module;
}
extern "C" {
#endif

struct vt_nir_to_llvm_options {
   const char *module_name;
   const char *target_triple;
   const char *data_layout;
   /* Optional entry symbol for independently dispatched RT callback kernels. */
   const char *entry_point_name;
   bool entry_point_is_kernel;
   bool require_ssa_function_temps;
   /* Global RT has no PDS-visible launch-id slots. */
   bool global_rt_wavefront;
   /* A global CPS resume uses a compact physical dispatch and restores its
    * logical launch ID from the completion plane.  It needs a ray-ref-count
    * guard, not the root raygen image-extent guard. */
   bool global_rt_compact_resume;
   /* MegaKernel private allocas start after the linked RT frame.  Zero keeps
    * the independent-allocation layout used by staged callback kernels. */
   uint32_t private_base_bytes;
};

llvm::Module *vt_nir_to_llvm_translate(nir_shader **shaders,
                                        unsigned shader_count,
                                        const struct vt_nir_to_llvm_options *options);

#ifdef __cplusplus
}
#endif

#endif
