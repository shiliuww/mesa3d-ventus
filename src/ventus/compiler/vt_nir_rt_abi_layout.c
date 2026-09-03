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

   layout.ray_state_base_bytes = VT_RT_LOCAL_RAY_STATE_BASE_BYTES;
   layout.ray_state_size_bytes = VT_RT_LOCAL_RAY_STATE_SIZE_BYTES;

   layout.control_base_bytes = VT_RT_LOCAL_CONTROL_BASE_BYTES;
   layout.control_size_bytes = VT_RT_LOCAL_CONTROL_SIZE_BYTES;

   layout.candidate_hit_record_base_bytes =
      VT_RT_LOCAL_CANDIDATE_HIT_RECORD_BASE_BYTES;
   layout.candidate_hit_record_size_bytes =
      VT_RT_LOCAL_CANDIDATE_HIT_RECORD_SIZE_BYTES;

   layout.hit_record_base_bytes = VT_RT_LOCAL_COMMITTED_HIT_RECORD_BASE_BYTES;
   layout.hit_record_size_bytes = VT_RT_LOCAL_COMMITTED_HIT_RECORD_SIZE_BYTES;

   layout.hit_attrib_base_bytes = VT_RT_PDS_CANDIDATE_HIT_ATTRIB_BASE_BYTES;
   layout.committed_hit_attrib_base_bytes =
      VT_RT_PDS_COMMITTED_HIT_ATTRIB_BASE_BYTES(0);
   layout.hit_attrib_size_bytes = VT_RT_PDS_HIT_ATTRIB_SIZE_BYTES;
   layout.hit_attrib_barycentrics = false;

   layout.payload_base_bytes = VT_RT_PDS_PAYLOAD_BASE_BYTES(0);
   layout.payload_size_bytes = VT_RT_PDS_DEFAULT_PAYLOAD_SIZE_BYTES;
   layout.region_size_bytes = 0;
   layout.private_base_bytes = 0;
   layout.payload_spill_base_bytes = layout.region_size_bytes;
   layout.payload_spill_size_bytes = 0;
   layout.max_ray_recursion_depth = 0;
   layout.total_size_bytes = 0;

   return layout;
}

bool
vt_nir_rt_abi_layout_set_hit_attrib_size(
   struct vt_nir_rt_abi_layout *layout, uint32_t hit_attrib_size_bytes,
   bool hit_attrib_barycentrics)
{
   if (!layout || hit_attrib_size_bytes >
                      (UINT32_MAX - layout->hit_attrib_base_bytes) / 2)
      return false;

   layout->hit_attrib_size_bytes = hit_attrib_size_bytes;
   layout->hit_attrib_barycentrics = hit_attrib_barycentrics;
   layout->committed_hit_attrib_base_bytes =
      layout->hit_attrib_base_bytes + hit_attrib_size_bytes;
   layout->payload_base_bytes =
      VT_RT_PDS_PAYLOAD_BASE_BYTES(hit_attrib_size_bytes);
   layout->region_size_bytes = layout->payload_base_bytes;
   layout->private_base_bytes = layout->region_size_bytes;
   layout->payload_spill_base_bytes = layout->region_size_bytes;
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
   layout->payload_spill_base_bytes = layout->region_size_bytes;
   layout->payload_spill_size_bytes = 0;
   layout->total_size_bytes = layout->region_size_bytes;
   return true;
}

bool
vt_nir_rt_abi_layout_set_recursion_depth(
   struct vt_nir_rt_abi_layout *layout, uint32_t max_ray_recursion_depth)
{
   if (!layout ||
       max_ray_recursion_depth > VT_RT_TRACE_META1_MAX_RECURSION_DEPTH)
      return false;

   /* D=0 still owns the initial payload frame.  D>0 has exactly D frames,
    * so PDS is 2A + P * max(1, D), with no RT Local prefix. */
   const uint32_t frame_count = MAX2(max_ray_recursion_depth, 1u);
   const uint64_t spill_size = (uint64_t)layout->payload_size_bytes *
      (frame_count - 1);
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
      /* Fixed RT state is exclusively RT Local SRAM.  PDS starts with the
       * variable-size attributes/payload region, and ordinary LLVM private
       * objects follow only that PDS-backed region. */
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
      config->scratchpad[i].committed_hit_attrib_base_bytes =
         layout->committed_hit_attrib_base_bytes;
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
         MAX2(config->scratchpad[i].stride_bytes, layout->private_base_bytes);
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

   const uint32_t frame_size =
      VT_RT_LOCAL_HEADER_SIZE_BYTES + layout->region_size_bytes;
   for (unsigned i = 0; i < VT_RT_LAYOUT_COUNT; ++i) {
      config->scratchpad[i].base_bytes = 0;
      config->scratchpad[i].ray_state_base_bytes =
         layout->ray_state_base_bytes;
      config->scratchpad[i].payload_base_bytes =
         VT_RT_LOCAL_HEADER_SIZE_BYTES + layout->payload_base_bytes;
      config->scratchpad[i].control_base_bytes =
         layout->control_base_bytes;
      config->scratchpad[i].candidate_hit_record_base_bytes =
         layout->candidate_hit_record_base_bytes;
      config->scratchpad[i].hit_record_base_bytes =
         layout->hit_record_base_bytes;
      config->scratchpad[i].hit_attrib_base_bytes =
         VT_RT_LOCAL_HEADER_SIZE_BYTES + layout->hit_attrib_base_bytes;
      config->scratchpad[i].committed_hit_attrib_base_bytes =
         VT_RT_LOCAL_HEADER_SIZE_BYTES + layout->committed_hit_attrib_base_bytes;
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
      config->scratchpad[i].total_size_bytes =
         VT_RT_LOCAL_HEADER_SIZE_BYTES + layout->total_size_bytes;
      /* Global CPS is independent from legacy PDS sizing.  CPS lowering
       * grows scratch beyond this static RT region for call spills. */
      config->scratchpad[i].stride_bytes = frame_size;
   }
}
