/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VENTUS_NIR_TO_LLVM_H
#define VENTUS_NIR_TO_LLVM_H

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
};

llvm::Module *vt_nir_to_llvm_translate(nir_shader **shaders,
                                        unsigned shader_count,
                                        const struct vt_nir_to_llvm_options *options);

#ifdef __cplusplus
}
#endif

#endif
