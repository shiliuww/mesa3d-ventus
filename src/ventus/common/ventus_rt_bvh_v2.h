/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VENTUS_RT_BVH_V2_H
#define VENTUS_RT_BVH_V2_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ventus_vtas_v2_validation_error {
   VENTUS_VTAS_V2_VALID = 0,
   VENTUS_VTAS_V2_TRUNCATED,
   VENTUS_VTAS_V2_BAD_MAGIC,
   VENTUS_VTAS_V2_BAD_VERSION,
   VENTUS_VTAS_V2_BAD_HEADER,
   VENTUS_VTAS_V2_BAD_ROOT,
   VENTUS_VTAS_V2_BAD_NODE_REF,
   VENTUS_VTAS_V2_BAD_NODE_TYPE,
   VENTUS_VTAS_V2_BAD_PARENT,
   VENTUS_VTAS_V2_BAD_BACKLINK,
   VENTUS_VTAS_V2_DUPLICATE_OR_CYCLE,
   VENTUS_VTAS_V2_NONZERO_RESERVED,
   VENTUS_VTAS_V2_NODE_COUNT_MISMATCH,
};

struct ventus_vtas_v2_validation_result {
   enum ventus_vtas_v2_validation_error error;
   uint32_t node_ref;
   uint32_t detail;
};

bool ventus_vtas_v2_validate(const void *image, size_t image_size,
                             struct ventus_vtas_v2_validation_result *result);

const char *ventus_vtas_v2_validation_error_string(
   enum ventus_vtas_v2_validation_error error);

#ifdef __cplusplus
}
#endif

#endif
