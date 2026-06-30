/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

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

   layout.payload_base_bytes = VT_RT_ABI_PAYLOAD_BASE_BYTES;
   layout.payload_size_bytes = VT_RT_ABI_DEFAULT_PAYLOAD_SIZE_BYTES;

   layout.continuation_base_bytes = VT_RT_ABI_CONTINUATION_BASE_BYTES;
   layout.continuation_size_bytes = VT_RT_ABI_CONTINUATION_SIZE_BYTES;
   layout.total_size_bytes = VT_RT_ABI_TOTAL_SIZE_BYTES;

   return layout;
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
      config->scratchpad[i].continuation_base_bytes =
         layout->continuation_base_bytes;
      config->scratchpad[i].continuation_size_bytes =
         layout->continuation_size_bytes;
      config->scratchpad[i].total_size_bytes = layout->total_size_bytes;
      config->scratchpad[i].stride_bytes =
         MAX2(config->scratchpad[i].stride_bytes, layout->total_size_bytes);
   }
}
