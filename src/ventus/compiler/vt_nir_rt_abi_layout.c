/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

#include <limits.h>

struct vt_nir_rt_abi_layout
vt_nir_rt_default_abi_layout(void)
{
   struct vt_nir_rt_abi_layout layout = {0};

   layout.ray_state_base_bytes = VT_RT_ABI_RAY_STATE_BASE_BYTES;
   layout.ray_state_size_bytes = VT_RT_ABI_RAY_STATE_SIZE_BYTES;

   layout.control_base_bytes = VT_RT_ABI_CONTROL_BASE_BYTES;
   layout.control_size_bytes = VT_RT_ABI_CONTROL_SIZE_BYTES;

   layout.candidate_hit_record_base_bytes =
      VT_RT_ABI_CANDIDATE_HIT_RECORD_BASE_BYTES;
   layout.candidate_hit_record_size_bytes =
      VT_RT_ABI_CANDIDATE_HIT_RECORD_SIZE_BYTES;

   layout.hit_record_base_bytes = VT_RT_ABI_COMMITTED_HIT_RECORD_BASE_BYTES;
   layout.hit_record_size_bytes = VT_RT_ABI_COMMITTED_HIT_RECORD_SIZE_BYTES;

   layout.hit_attrib_base_bytes = VT_RT_ABI_HIT_ATTRIB_BASE_BYTES;
   layout.hit_attrib_size_bytes = VT_RT_ABI_HIT_ATTRIB_SIZE_BYTES;
   layout.hit_attrib_barycentrics = false;

   layout.payload_base_bytes = VT_RT_ABI_PAYLOAD_BASE_BYTES;
   layout.payload_size_bytes = VT_RT_ABI_DEFAULT_PAYLOAD_SIZE_BYTES;
   layout.region_size_bytes = VT_RT_ABI_FIXED_HEADER_SIZE_BYTES;
   layout.private_base_bytes = layout.region_size_bytes;
   layout.payload_spill_base_bytes = layout.private_base_bytes;
   layout.payload_spill_size_bytes = 0;
   layout.max_ray_recursion_depth = 0;
   layout.total_size_bytes = VT_RT_ABI_FIXED_HEADER_SIZE_BYTES;

   return layout;
}

bool
vt_nir_rt_abi_layout_set_hit_attrib_size(
   struct vt_nir_rt_abi_layout *layout, uint32_t hit_attrib_size_bytes,
   bool hit_attrib_barycentrics)
{
   if (!layout || hit_attrib_size_bytes > UINT32_MAX -
                         layout->hit_attrib_base_bytes)
      return false;

   layout->hit_attrib_size_bytes = hit_attrib_size_bytes;
   layout->hit_attrib_barycentrics = hit_attrib_barycentrics;
   layout->payload_base_bytes = layout->hit_attrib_base_bytes +
                                hit_attrib_size_bytes;
   layout->region_size_bytes = layout->payload_base_bytes;
   layout->private_base_bytes = layout->region_size_bytes;
   layout->payload_spill_base_bytes = layout->private_base_bytes;
   layout->payload_spill_size_bytes = 0;
   layout->total_size_bytes = layout->region_size_bytes;
   return true;
}

bool
vt_nir_rt_abi_layout_set_payload_size(struct vt_nir_rt_abi_layout *layout,
                                      uint32_t payload_size_bytes)
{
   if (!layout || payload_size_bytes > UINT32_MAX - layout->payload_base_bytes)
      return false;

   layout->payload_size_bytes = payload_size_bytes;
   layout->region_size_bytes = layout->payload_base_bytes + payload_size_bytes;
   layout->private_base_bytes = layout->region_size_bytes;
   layout->payload_spill_base_bytes = layout->private_base_bytes;
   layout->payload_spill_size_bytes = 0;
   layout->total_size_bytes = layout->region_size_bytes;
   return true;
}

bool
vt_nir_rt_abi_layout_set_recursion_depth(
   struct vt_nir_rt_abi_layout *layout, uint32_t max_ray_recursion_depth)
{
   if (!layout || max_ray_recursion_depth > UINT32_MAX)
      return false;

   const uint64_t spill_size =
      (uint64_t)layout->payload_size_bytes * max_ray_recursion_depth;
   const uint64_t total = (uint64_t)layout->region_size_bytes + spill_size;
   if (spill_size > UINT32_MAX || total > UINT32_MAX)
      return false;

   layout->max_ray_recursion_depth = max_ray_recursion_depth;
   layout->payload_spill_base_bytes = layout->region_size_bytes;
   layout->payload_spill_size_bytes = (uint32_t)spill_size;
   layout->private_base_bytes = (uint32_t)total;
   layout->total_size_bytes = (uint32_t)total;
   return true;
}

void
vt_nir_rt_config_apply_abi_layout(struct vt_nir_rt_config *config,
                                  const struct vt_nir_rt_abi_layout *layout)
{
   if (!config)
      return;

   struct vt_nir_rt_abi_layout default_layout;
   if (!layout) {
      default_layout = vt_nir_rt_default_abi_layout();
      layout = &default_layout;
   }

   for (unsigned i = 0; i < VT_RT_LAYOUT_COUNT; ++i) {
      /* Megakernel v4: fixed RT state is at PDS logical offset zero.  crt0
       * initializes TP above total_size_bytes, so LLVM AS5 stack objects do
       * not overlap this prefix. */
      config->scratchpad[i].base_bytes = 0;
      config->scratchpad[i].ray_state_base_bytes =
         layout->ray_state_base_bytes;
      config->scratchpad[i].payload_base_bytes =
         layout->payload_base_bytes;
      config->scratchpad[i].control_base_bytes =
         layout->control_base_bytes;
      config->scratchpad[i].candidate_hit_record_base_bytes =
         layout->candidate_hit_record_base_bytes;
      config->scratchpad[i].hit_record_base_bytes =
         layout->hit_record_base_bytes;
      config->scratchpad[i].hit_attrib_base_bytes =
         layout->hit_attrib_base_bytes;
      config->scratchpad[i].hit_attrib_size_bytes =
         layout->hit_attrib_size_bytes;
      config->scratchpad[i].hit_attrib_barycentrics =
         layout->hit_attrib_barycentrics;
      config->scratchpad[i].payload_spill_base_bytes =
         layout->payload_spill_base_bytes;
      config->scratchpad[i].payload_spill_size_bytes =
         layout->payload_spill_size_bytes;
      config->scratchpad[i].max_ray_recursion_depth =
         layout->max_ray_recursion_depth;
      config->scratchpad[i].private_base_bytes = layout->private_base_bytes;
      config->scratchpad[i].total_size_bytes = layout->total_size_bytes;
      config->scratchpad[i].stride_bytes =
         MAX2(config->scratchpad[i].stride_bytes, layout->total_size_bytes);
   }
}

void
vt_nir_rt_config_apply_global_cps_layout(
   struct vt_nir_rt_config *config,
   const struct vt_nir_rt_abi_layout *layout)
{
   if (!config)
      return;

   struct vt_nir_rt_abi_layout default_layout;
   if (!layout) {
      default_layout = vt_nir_rt_default_abi_layout();
      layout = &default_layout;
   }

   const uint32_t frame_size = layout->region_size_bytes;
   for (unsigned i = 0; i < VT_RT_LAYOUT_COUNT; ++i) {
      config->scratchpad[i].base_bytes = 0;
      config->scratchpad[i].ray_state_base_bytes =
         layout->ray_state_base_bytes;
      config->scratchpad[i].payload_base_bytes =
         layout->payload_base_bytes;
      config->scratchpad[i].control_base_bytes =
         layout->control_base_bytes;
      config->scratchpad[i].candidate_hit_record_base_bytes =
         layout->candidate_hit_record_base_bytes;
      config->scratchpad[i].hit_record_base_bytes =
         layout->hit_record_base_bytes;
      config->scratchpad[i].hit_attrib_base_bytes =
         layout->hit_attrib_base_bytes;
      config->scratchpad[i].hit_attrib_size_bytes =
         layout->hit_attrib_size_bytes;
      config->scratchpad[i].hit_attrib_barycentrics =
         layout->hit_attrib_barycentrics;
      config->scratchpad[i].payload_spill_base_bytes = 0;
      config->scratchpad[i].payload_spill_size_bytes = 0;
      config->scratchpad[i].max_ray_recursion_depth = 0;
      /* Global CPS owns its continuation storage; it does not reserve the
       * Megakernel private-frame prefix. */
      config->scratchpad[i].private_base_bytes = 0;
      config->scratchpad[i].total_size_bytes = frame_size;
      /* Global CPS is independent from legacy PDS sizing.  CPS lowering
       * grows scratch beyond this static RT region for call spills. */
      config->scratchpad[i].stride_bytes = frame_size;
   }
}
