/* SPDX-License-Identifier: MIT */
#ifndef VENTUS_RT_ADDRESS_H
#define VENTUS_RT_ADDRESS_H

#include <stdbool.h>
#include <stdint.h>

/* RV32 execution may consume a u64 host/device ABI address only after this
 * check.  A zero-size allocation still validates its base as one byte so a
 * non-null high address can never be silently narrowed. */
static inline bool
ventus_rt_device_range_fits_u32(uint64_t addr, uint64_t size)
{
   if (!size)
      size = 1;
   return addr <= UINT32_MAX && size - 1 <= UINT32_MAX - addr;
}

#endif
