/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VENTUS_RT_V5_H
#define VENTUS_RT_V5_H

#include <stdbool.h>
#include <stdint.h>

#define VENTUS_RT_V5_LANES_PER_WARP 32u
#define VENTUS_RT_V5_PDS_SIZE_BYTES 384u
#define VENTUS_RT_V5_PRIVATE_LANE_STRIDE_BYTES 128u
#define VENTUS_RT_V5_PRIVATE_WARP_STRIDE_BYTES 4096u
#define VENTUS_RT_V5_PRIVATE_ALIGNMENT_BYTES 128u
#define VENTUS_RT_V5_CONTINUATION_ALIGNMENT_BYTES 16u
#define VENTUS_RT_V5_CALL_DATA_OFFSET 0x120u
#define VENTUS_RT_V5_CALL_DATA_CAPACITY_BYTES 96u

enum ventus_rt_v5_pds_offset {
   VENTUS_RT_V5_PDS_TRAVERSABLE_REFERENCE = 0x000,
   VENTUS_RT_V5_PDS_RAY_FLAGS = 0x008,
   VENTUS_RT_V5_PDS_TRACE_SELECTOR_CONTROL = 0x00c,
   VENTUS_RT_V5_PDS_WORLD_ORIGIN = 0x020,
   VENTUS_RT_V5_PDS_RAY_TMIN = 0x02c,
   VENTUS_RT_V5_PDS_WORLD_DIRECTION = 0x030,
   VENTUS_RT_V5_PDS_RAY_TMAX = 0x03c,
   VENTUS_RT_V5_PDS_CONTINUATION_FRAME_ADDRESS = 0x040,
   VENTUS_RT_V5_PDS_RECURSION_DEPTH = 0x050,
   VENTUS_RT_V5_PDS_TRACE_RETURN_CONTROL = 0x054,
   VENTUS_RT_V5_PDS_CALLBACK_BASE = 0x060,
   VENTUS_RT_V5_PDS_COMMITTED_BASE = 0x0c0,
   VENTUS_RT_V5_PDS_CALL_DATA_BASE = VENTUS_RT_V5_CALL_DATA_OFFSET,
};

enum ventus_rt_v5_private_offset {
   VENTUS_RT_V5_PRIVATE_FRONTIER_0 = 0x00,
   VENTUS_RT_V5_PRIVATE_FRONTIER_4 = 0x20,
   VENTUS_RT_V5_PRIVATE_RAY_TMIN = 0x30,
   VENTUS_RT_V5_PRIVATE_RAY_TMAX = 0x34,
   VENTUS_RT_V5_PRIVATE_RAY_FLAGS = 0x38,
   VENTUS_RT_V5_PRIVATE_HOT_CONTROL = 0x3c,
   VENTUS_RT_V5_PRIVATE_ORIGIN = 0x40,
   VENTUS_RT_V5_PRIVATE_INV_DIRECTION = 0x4c,
   VENTUS_RT_V5_PRIVATE_ACTIVE_AS_BASE = 0x58,
   VENTUS_RT_V5_PRIVATE_DIRECTION = 0x60,
};

enum ventus_rt_v5_return_reason {
   VENTUS_RT_V5_RETURN_NONE_OR_INVALID = 0x00,
   VENTUS_RT_V5_RETURN_MISS = 0x01,
   VENTUS_RT_V5_RETURN_CLOSEST_HIT_READY = 0x02,
   VENTUS_RT_V5_RETURN_ANY_HIT_REQUIRED = 0x03,
   VENTUS_RT_V5_RETURN_INTERSECTION_REQUIRED = 0x04,
   VENTUS_RT_V5_RETURN_HIT_DONE_NO_SHADER = 0x05,
   VENTUS_RT_V5_RETURN_FAULT = 0x06,
   VENTUS_RT_V5_RETURN_UNSUPPORTED = 0x07,
};

#define VENTUS_RT_V5_COMPLETION_VALID (1u << 31)

struct ventus_rt_v5_private_placement {
   uint64_t region_offset;
   uint64_t total_bytes;
   uint64_t num_lds;
};

static inline uint32_t
ventus_rt_v5_compact_result(enum ventus_rt_v5_return_reason reason)
{
   return VENTUS_RT_V5_COMPLETION_VALID | (uint32_t)reason;
}

static inline bool
ventus_rt_v5_align_u64(uint64_t value, uint64_t alignment,
                       uint64_t *result)
{
   if (!result || alignment == 0 || (alignment & (alignment - 1)) != 0)
      return false;
   const uint64_t mask = alignment - 1;
   if (value > UINT64_MAX - mask)
      return false;
   *result = (value + mask) & ~mask;
   return true;
}

static inline bool
ventus_rt_v5_private_address(uint64_t cta_shared_base,
                             uint64_t region_offset,
                             uint32_t warp_index_in_cta, uint32_t lane_id,
                             uint64_t *address)
{
   if (!address || lane_id >= VENTUS_RT_V5_LANES_PER_WARP)
      return false;

   const uint64_t warp_offset =
      (uint64_t)warp_index_in_cta * VENTUS_RT_V5_PRIVATE_WARP_STRIDE_BYTES;
   const uint64_t lane_offset =
      (uint64_t)lane_id * VENTUS_RT_V5_PRIVATE_LANE_STRIDE_BYTES;
   if (warp_index_in_cta != 0 &&
       warp_offset / VENTUS_RT_V5_PRIVATE_WARP_STRIDE_BYTES !=
          warp_index_in_cta)
      return false;
   if (cta_shared_base > UINT64_MAX - region_offset)
      return false;
   uint64_t value = cta_shared_base + region_offset;
   if (value > UINT64_MAX - warp_offset)
      return false;
   value += warp_offset;
   if (value > UINT64_MAX - lane_offset)
      return false;
   *address = value + lane_offset;
   return (*address & (VENTUS_RT_V5_PRIVATE_ALIGNMENT_BYTES - 1)) == 0;
}

static inline bool
ventus_rt_v5_private_placement(uint64_t non_rt_cta_lds_bytes,
                               uint32_t num_warps,
                               uint64_t per_cta_lds_capability,
                               struct ventus_rt_v5_private_placement *placement)
{
   if (!placement || num_warps == 0)
      return false;

   uint64_t region_offset = 0;
   if (!ventus_rt_v5_align_u64(non_rt_cta_lds_bytes,
                               VENTUS_RT_V5_PRIVATE_ALIGNMENT_BYTES,
                               &region_offset))
      return false;
   const uint64_t total_bytes =
      (uint64_t)num_warps * VENTUS_RT_V5_PRIVATE_WARP_STRIDE_BYTES;
   if (num_warps != 0 &&
       total_bytes / VENTUS_RT_V5_PRIVATE_WARP_STRIDE_BYTES != num_warps)
      return false;
   if (region_offset > UINT64_MAX - total_bytes)
      return false;

   const uint64_t num_lds = region_offset + total_bytes;
   if (num_lds > per_cta_lds_capability)
      return false;
   placement->region_offset = region_offset;
   placement->total_bytes = total_bytes;
   placement->num_lds = num_lds;
   return true;
}

#if defined(__cplusplus)
static_assert(VENTUS_RT_V5_PRIVATE_WARP_STRIDE_BYTES ==
                 VENTUS_RT_V5_LANES_PER_WARP *
                    VENTUS_RT_V5_PRIVATE_LANE_STRIDE_BYTES,
              "V5 private warp stride mismatch");
static_assert(VENTUS_RT_V5_CALL_DATA_OFFSET +
                    VENTUS_RT_V5_CALL_DATA_CAPACITY_BYTES ==
                 VENTUS_RT_V5_PDS_SIZE_BYTES,
              "V5 PDS call-data extent mismatch");
#else
_Static_assert(VENTUS_RT_V5_PRIVATE_WARP_STRIDE_BYTES ==
                  VENTUS_RT_V5_LANES_PER_WARP *
                     VENTUS_RT_V5_PRIVATE_LANE_STRIDE_BYTES,
               "V5 private warp stride mismatch");
_Static_assert(VENTUS_RT_V5_CALL_DATA_OFFSET +
                     VENTUS_RT_V5_CALL_DATA_CAPACITY_BYTES ==
                  VENTUS_RT_V5_PDS_SIZE_BYTES,
               "V5 PDS call-data extent mismatch");
#endif

#endif
