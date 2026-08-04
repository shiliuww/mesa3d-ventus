/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 *
 * Kernel metadata is consumed by crt0 and the NIR-to-LLVM metadata
 * intrinsic.  Keep its byte offsets here so host, compiler and startup code
 * cannot silently disagree when the device address width changes.
 */
#ifndef VENTUS_KERNEL_METADATA_H
#define VENTUS_KERNEL_METADATA_H

enum ventus_kernel_metadata_rv32_offset {
   VENTUS_KNL32_ENTRY = 0,
   VENTUS_KNL32_ARG_BASE = 4,
   VENTUS_KNL32_WORK_DIM = 8,
   VENTUS_KNL32_GL_SIZE_X = 12,
   VENTUS_KNL32_GL_SIZE_Y = 16,
   VENTUS_KNL32_GL_SIZE_Z = 20,
   VENTUS_KNL32_LC_SIZE_X = 24,
   VENTUS_KNL32_LC_SIZE_Y = 28,
   VENTUS_KNL32_LC_SIZE_Z = 32,
   VENTUS_KNL32_GL_OFFSET_X = 36,
   VENTUS_KNL32_GL_OFFSET_Y = 40,
   VENTUS_KNL32_GL_OFFSET_Z = 44,
   VENTUS_KNL32_PDS_SIZE_PER_THREAD = 64,
   VENTUS_KNL32_MAX_SIZE = 68,
};

/* RV64 keeps launch dimensions as u32, but aligns each device address to
 * eight bytes.  ARG_BASE must therefore not share an RV32-sized metadata word
 * with ENTRY. */
enum ventus_kernel_metadata_rv64_offset {
   VENTUS_KNL64_ENTRY = 0,
   VENTUS_KNL64_ARG_BASE = 8,
   VENTUS_KNL64_WORK_DIM = 16,
   VENTUS_KNL64_GL_SIZE_X = 20,
   VENTUS_KNL64_GL_SIZE_Y = 24,
   VENTUS_KNL64_GL_SIZE_Z = 28,
   VENTUS_KNL64_LC_SIZE_X = 32,
   VENTUS_KNL64_LC_SIZE_Y = 36,
   VENTUS_KNL64_LC_SIZE_Z = 40,
   VENTUS_KNL64_GL_OFFSET_X = 44,
   VENTUS_KNL64_GL_OFFSET_Y = 48,
   VENTUS_KNL64_GL_OFFSET_Z = 52,
   VENTUS_KNL64_PRINT_ADDR = 56,
   VENTUS_KNL64_PRINT_SIZE = 64,
   VENTUS_KNL64_LDS_STACK_SIZE_PER_WF = 68,
   VENTUS_KNL64_LDS_NON_STACK_SIZE = 72,
   VENTUS_KNL64_PDS_SIZE_PER_THREAD = 80,
   VENTUS_KNL64_MAX_SIZE = 88,
};

#endif
