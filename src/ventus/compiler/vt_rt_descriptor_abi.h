/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VT_RT_DESCRIPTOR_ABI_H
#define VT_RT_DESCRIPTOR_ABI_H

#include <stdbool.h>
#include <stdint.h>

enum vt_rt_descriptor_table_layout {
   VT_RT_DESCRIPTOR_BINDINGS_PER_SET = 32,
   VT_RT_DESCRIPTOR_SET_COUNT = 4,
   VT_RT_DESCRIPTOR_TABLE_SLOT_COUNT =
      VT_RT_DESCRIPTOR_BINDINGS_PER_SET * VT_RT_DESCRIPTOR_SET_COUNT,
};

static inline bool
vt_rt_descriptor_location_valid(uint32_t set, uint32_t binding)
{
   return set < VT_RT_DESCRIPTOR_SET_COUNT &&
          binding < VT_RT_DESCRIPTOR_BINDINGS_PER_SET;
}

static inline uint32_t
vt_rt_descriptor_table_slot(uint32_t set, uint32_t binding)
{
   return set * VT_RT_DESCRIPTOR_BINDINGS_PER_SET + binding;
}

#endif
