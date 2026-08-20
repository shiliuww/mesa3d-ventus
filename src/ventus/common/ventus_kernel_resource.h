/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#ifndef VENTUS_KERNEL_RESOURCE_H
#define VENTUS_KERNEL_RESOURCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define VENTUS_KERNEL_RESOURCE_V3_VERSION 3u
#define VENTUS_KERNEL_RESOURCE_V4_VERSION 4u
#define VENTUS_KERNEL_RESOURCE_V3_SIZE 56u
#define VENTUS_KERNEL_RESOURCE_V4_SIZE 96u

#define VENTUS_RT_PDS_LAYOUT_V5 5u
#define VENTUS_RT_PRIVATE_LAYOUT_V5 5u
#define VENTUS_RT_HOT_PDS_STRIDE_V5 384u
#define VENTUS_RT_MAX_PAYLOAD_BYTES_V5 96u
#define VENTUS_RT_MAX_CALLABLE_DATA_BYTES_V5 96u

enum ventus_kernel_resource_flags {
   VENTUS_KERNEL_RESOURCE_HAS_DYNAMIC_ALLOCA = 1u << 0,
   VENTUS_KERNEL_RESOURCE_HAS_RECURSION = 1u << 1,
   VENTUS_KERNEL_RESOURCE_HAS_INDIRECT_CALL = 1u << 2,
   VENTUS_KERNEL_RESOURCE_HAS_UNKNOWN_EXTERNAL_CALLEE = 1u << 3,
   VENTUS_KERNEL_RESOURCE_STACK_PEAK_UNAVAILABLE = 1u << 4,
   VENTUS_KERNEL_RESOURCE_REGISTER_USAGE_INCOMPLETE = 1u << 5,
};

enum ventus_rt_resource_feature {
   VENTUS_RT_RESOURCE_USES_TRAVERSAL = 1u << 0,
   VENTUS_RT_RESOURCE_USES_TRACE_CONTINUATION = 1u << 1,
   VENTUS_RT_RESOURCE_USES_CALLABLE_CONTINUATION = 1u << 2,
   VENTUS_RT_RESOURCE_KNOWN_FEATURES =
      VENTUS_RT_RESOURCE_USES_TRAVERSAL |
      VENTUS_RT_RESOURCE_USES_TRACE_CONTINUATION |
      VENTUS_RT_RESOURCE_USES_CALLABLE_CONTINUATION,
};

struct ventus_kernel_resource_v3 {
   uint32_t version;
   uint32_t flags;
   uint64_t vgpr_used;
   uint64_t sgpr_used;
   uint64_t lds_static_bytes;
   uint64_t pds_static_bytes;
   uint64_t lds_stack_peak_bytes;
   uint64_t pds_stack_peak_bytes;
};

struct ventus_kernel_resource_v4 {
   uint32_t version;
   uint32_t flags;
   uint64_t vgpr_used;
   uint64_t sgpr_used;
   uint64_t lds_static_bytes;
   uint64_t pds_static_bytes;
   uint64_t lds_stack_peak_bytes;
   uint64_t pds_stack_peak_bytes;
   uint32_t rt_feature_flags;
   uint32_t rt_pds_layout_id;
   uint32_t rt_private_layout_id;
   uint32_t rt_hot_pds_offset_per_thread;
   uint32_t rt_hot_pds_stride_bytes;
   uint32_t rt_continuation_offset_per_thread;
   uint32_t rt_continuation_bytes_per_thread;
   uint32_t rt_payload_size_bytes;
   uint32_t rt_callable_data_size_bytes;
   uint32_t reserved_zero;
};

/* Inputs whose values are established by whole-pipeline RT analysis.  Layout
 * identities, the hot-PDS stride, the V4 version, and reserved_zero are fixed
 * by the V5 contract and are therefore not caller-controlled. */
struct ventus_kernel_resource_v5_build_info {
   uint32_t feature_flags;
   uint32_t hot_pds_offset_per_thread;
   uint32_t continuation_offset_per_thread;
   uint32_t continuation_bytes_per_thread;
   uint32_t payload_size_bytes;
   uint32_t callable_data_size_bytes;
};

enum ventus_kernel_resource_v4_error {
   VENTUS_KERNEL_RESOURCE_V4_VALID = 0,
   VENTUS_KERNEL_RESOURCE_V4_BAD_VERSION,
   VENTUS_KERNEL_RESOURCE_V4_BAD_RESERVED,
   VENTUS_KERNEL_RESOURCE_V4_BAD_FEATURE_FLAGS,
   VENTUS_KERNEL_RESOURCE_V4_RT_REQUIRED,
   VENTUS_KERNEL_RESOURCE_V4_BAD_LAYOUT,
   VENTUS_KERNEL_RESOURCE_V4_BAD_HOT_PDS,
   VENTUS_KERNEL_RESOURCE_V4_BAD_CONTINUATION,
   VENTUS_KERNEL_RESOURCE_V4_BAD_PAYLOAD,
   VENTUS_KERNEL_RESOURCE_V4_BAD_CALLABLE_DATA,
   VENTUS_KERNEL_RESOURCE_V4_BAD_EXTENT,
   VENTUS_KERNEL_RESOURCE_V4_BAD_ARGUMENT,
   VENTUS_KERNEL_RESOURCE_V4_RT_MISMATCH,
};

static inline bool
ventus_u64_range_end(uint64_t offset, uint64_t size, uint64_t *end)
{
   if (offset > UINT64_MAX - size)
      return false;
   *end = offset + size;
   return true;
}

static inline bool
ventus_kernel_resource_v4_has_rt(
   const struct ventus_kernel_resource_v4 *resource)
{
   return resource->rt_feature_flags != 0 ||
          resource->rt_pds_layout_id != 0 ||
          resource->rt_private_layout_id != 0 ||
          resource->rt_hot_pds_offset_per_thread != 0 ||
          resource->rt_hot_pds_stride_bytes != 0 ||
          resource->rt_continuation_offset_per_thread != 0 ||
          resource->rt_continuation_bytes_per_thread != 0 ||
          resource->rt_payload_size_bytes != 0 ||
          resource->rt_callable_data_size_bytes != 0;
}

static inline enum ventus_kernel_resource_v4_error
ventus_kernel_resource_v4_validate(
   const struct ventus_kernel_resource_v4 *resource, bool require_rt)
{
   uint64_t hot_end = 0;
   uint64_t continuation_end = 0;

   if (!resource || resource->version != VENTUS_KERNEL_RESOURCE_V4_VERSION)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_VERSION;
   if (resource->reserved_zero != 0)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_RESERVED;
   if (resource->rt_feature_flags & ~VENTUS_RT_RESOURCE_KNOWN_FEATURES)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_FEATURE_FLAGS;

   const bool has_rt = ventus_kernel_resource_v4_has_rt(resource);
   if (!has_rt)
      return require_rt ? VENTUS_KERNEL_RESOURCE_V4_RT_REQUIRED :
                          VENTUS_KERNEL_RESOURCE_V4_VALID;
   if (!(resource->rt_feature_flags & VENTUS_RT_RESOURCE_USES_TRAVERSAL))
      return VENTUS_KERNEL_RESOURCE_V4_BAD_FEATURE_FLAGS;
   if (resource->rt_pds_layout_id != VENTUS_RT_PDS_LAYOUT_V5 ||
       resource->rt_private_layout_id != VENTUS_RT_PRIVATE_LAYOUT_V5)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_LAYOUT;
   if (resource->rt_hot_pds_stride_bytes !=
       VENTUS_RT_HOT_PDS_STRIDE_V5)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_HOT_PDS;
   if (resource->rt_payload_size_bytes >
       VENTUS_RT_MAX_PAYLOAD_BYTES_V5)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_PAYLOAD;
   if (resource->rt_callable_data_size_bytes >
       VENTUS_RT_MAX_CALLABLE_DATA_BYTES_V5)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_CALLABLE_DATA;
   if ((resource->rt_feature_flags &
        VENTUS_RT_RESOURCE_USES_TRACE_CONTINUATION) &&
       resource->rt_continuation_bytes_per_thread == 0)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_CONTINUATION;
   if ((resource->rt_feature_flags &
        VENTUS_RT_RESOURCE_USES_CALLABLE_CONTINUATION) &&
       resource->rt_callable_data_size_bytes == 0)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_CALLABLE_DATA;

   if (!ventus_u64_range_end(resource->rt_hot_pds_offset_per_thread,
                             resource->rt_hot_pds_stride_bytes, &hot_end) ||
       !ventus_u64_range_end(resource->rt_continuation_offset_per_thread,
                             resource->rt_continuation_bytes_per_thread,
                             &continuation_end))
      return VENTUS_KERNEL_RESOURCE_V4_BAD_EXTENT;
   if (hot_end > resource->pds_static_bytes ||
       continuation_end > resource->pds_static_bytes)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_EXTENT;
   if (resource->rt_continuation_bytes_per_thread != 0 &&
       resource->rt_continuation_offset_per_thread < hot_end &&
       continuation_end > resource->rt_hot_pds_offset_per_thread)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_CONTINUATION;

   return VENTUS_KERNEL_RESOURCE_V4_VALID;
}

static inline bool
ventus_kernel_resource_v4_rt_equal(
   const struct ventus_kernel_resource_v4 *lhs,
   const struct ventus_kernel_resource_v4 *rhs)
{
   return lhs && rhs &&
          lhs->rt_feature_flags == rhs->rt_feature_flags &&
          lhs->rt_pds_layout_id == rhs->rt_pds_layout_id &&
          lhs->rt_private_layout_id == rhs->rt_private_layout_id &&
          lhs->rt_hot_pds_offset_per_thread ==
             rhs->rt_hot_pds_offset_per_thread &&
          lhs->rt_hot_pds_stride_bytes == rhs->rt_hot_pds_stride_bytes &&
          lhs->rt_continuation_offset_per_thread ==
             rhs->rt_continuation_offset_per_thread &&
          lhs->rt_continuation_bytes_per_thread ==
             rhs->rt_continuation_bytes_per_thread &&
          lhs->rt_payload_size_bytes == rhs->rt_payload_size_bytes &&
          lhs->rt_callable_data_size_bytes ==
             rhs->rt_callable_data_size_bytes &&
          lhs->reserved_zero == rhs->reserved_zero;
}

/* Promote the unchanged V3 prefix to a V4 non-RT record.  V3 remains a valid
 * wire format for non-RT kernels; this helper is for producers that need one
 * in-memory V4 representation without inferring any RT capability. */
static inline enum ventus_kernel_resource_v4_error
ventus_kernel_resource_v4_from_v3(
   const struct ventus_kernel_resource_v3 *base,
   struct ventus_kernel_resource_v4 *resource)
{
   if (!base || !resource)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_ARGUMENT;
   if (base->version != VENTUS_KERNEL_RESOURCE_V3_VERSION)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_VERSION;

   struct ventus_kernel_resource_v4 candidate = {0};
   candidate.version = VENTUS_KERNEL_RESOURCE_V4_VERSION;
   candidate.flags = base->flags;
   candidate.vgpr_used = base->vgpr_used;
   candidate.sgpr_used = base->sgpr_used;
   candidate.lds_static_bytes = base->lds_static_bytes;
   candidate.pds_static_bytes = base->pds_static_bytes;
   candidate.lds_stack_peak_bytes = base->lds_stack_peak_bytes;
   candidate.pds_stack_peak_bytes = base->pds_stack_peak_bytes;

   const enum ventus_kernel_resource_v4_error error =
      ventus_kernel_resource_v4_validate(&candidate, false);
   if (error == VENTUS_KERNEL_RESOURCE_V4_VALID)
      *resource = candidate;
   return error;
}

/* Build a complete V4 record for a V5 RT kernel.  The output is transactional:
 * validation failure leaves *resource unchanged, so callers cannot publish a
 * partially initialized resource section. */
static inline enum ventus_kernel_resource_v4_error
ventus_kernel_resource_v4_build_v5(
   const struct ventus_kernel_resource_v3 *base,
   const struct ventus_kernel_resource_v5_build_info *info,
   struct ventus_kernel_resource_v4 *resource)
{
   if (!base || !info || !resource)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_ARGUMENT;

   struct ventus_kernel_resource_v4 candidate;
   enum ventus_kernel_resource_v4_error error =
      ventus_kernel_resource_v4_from_v3(base, &candidate);
   if (error != VENTUS_KERNEL_RESOURCE_V4_VALID)
      return error;

   candidate.rt_feature_flags = info->feature_flags;
   candidate.rt_pds_layout_id = VENTUS_RT_PDS_LAYOUT_V5;
   candidate.rt_private_layout_id = VENTUS_RT_PRIVATE_LAYOUT_V5;
   candidate.rt_hot_pds_offset_per_thread =
      info->hot_pds_offset_per_thread;
   candidate.rt_hot_pds_stride_bytes = VENTUS_RT_HOT_PDS_STRIDE_V5;
   candidate.rt_continuation_offset_per_thread =
      info->continuation_offset_per_thread;
   candidate.rt_continuation_bytes_per_thread =
      info->continuation_bytes_per_thread;
   candidate.rt_payload_size_bytes = info->payload_size_bytes;
   candidate.rt_callable_data_size_bytes = info->callable_data_size_bytes;
   candidate.reserved_zero = 0;

   error = ventus_kernel_resource_v4_validate(&candidate, true);
   if (error == VENTUS_KERNEL_RESOURCE_V4_VALID)
      *resource = candidate;
   return error;
}

/* Validate both final records before comparing their RT trailers.  Generic
 * register and stack requirements may still be enveloped independently, but
 * all reachable entries in one RT pipeline must publish one identical RT
 * contract. */
static inline enum ventus_kernel_resource_v4_error
ventus_kernel_resource_v4_validate_rt_compatible(
   const struct ventus_kernel_resource_v4 *lhs,
   const struct ventus_kernel_resource_v4 *rhs, bool require_rt)
{
   if (!lhs || !rhs)
      return VENTUS_KERNEL_RESOURCE_V4_BAD_ARGUMENT;

   enum ventus_kernel_resource_v4_error error =
      ventus_kernel_resource_v4_validate(lhs, require_rt);
   if (error != VENTUS_KERNEL_RESOURCE_V4_VALID)
      return error;
   error = ventus_kernel_resource_v4_validate(rhs, require_rt);
   if (error != VENTUS_KERNEL_RESOURCE_V4_VALID)
      return error;

   return ventus_kernel_resource_v4_rt_equal(lhs, rhs) ?
      VENTUS_KERNEL_RESOURCE_V4_VALID :
      VENTUS_KERNEL_RESOURCE_V4_RT_MISMATCH;
}

#if defined(__cplusplus)
static_assert(sizeof(struct ventus_kernel_resource_v3) ==
                 VENTUS_KERNEL_RESOURCE_V3_SIZE,
              "Ventus kernel resource V3 size mismatch");
static_assert(sizeof(struct ventus_kernel_resource_v4) ==
                 VENTUS_KERNEL_RESOURCE_V4_SIZE,
              "Ventus kernel resource V4 size mismatch");
#else
_Static_assert(sizeof(struct ventus_kernel_resource_v3) ==
                  VENTUS_KERNEL_RESOURCE_V3_SIZE,
               "Ventus kernel resource V3 size mismatch");
_Static_assert(sizeof(struct ventus_kernel_resource_v4) ==
                  VENTUS_KERNEL_RESOURCE_V4_SIZE,
               "Ventus kernel resource V4 size mismatch");
#endif

#endif
