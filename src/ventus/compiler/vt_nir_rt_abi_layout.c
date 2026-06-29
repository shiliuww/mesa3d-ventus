/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

#include "util/u_math.h"

struct vt_abi_region_allocator {
   uint32_t offset;
};

static uint32_t
alloc_region(struct vt_abi_region_allocator *alloc, uint32_t size,
             uint32_t alignment)
{
   alloc->offset = align(alloc->offset, alignment);
   uint32_t base = alloc->offset;
   alloc->offset += size;
   return base;
}

struct vt_nir_rt_abi_layout
vt_nir_rt_default_abi_layout(void)
{
   struct vt_abi_region_allocator alloc = {0};
   struct vt_nir_rt_abi_layout layout = {0};

   layout.ray_state_base_bytes = alloc_region(&alloc, VT_RT_SLOT_SBT_INDEX * 4 + 4, 16);
   layout.ray_state_size_bytes = VT_RT_SLOT_SBT_INDEX * 4 + 4;

   layout.payload_base_bytes =
      alloc_region(&alloc, (VT_RT_PAYLOAD_RAY_DIRECTION_Z + 1) * 4, 16);
   layout.payload_size_bytes = (VT_RT_PAYLOAD_RAY_DIRECTION_Z + 1) * 4;

   layout.control_base_bytes =
      alloc_region(&alloc, VT_RT_CONTROL_WORD_COUNT * 4, 16);
   layout.control_size_bytes = VT_RT_CONTROL_WORD_COUNT * 4;

   layout.candidate_hit_record_base_bytes =
      alloc_region(&alloc, VT_RT_HIT_RECORD_WORD_COUNT * 4, 16);
   layout.candidate_hit_record_size_bytes = VT_RT_HIT_RECORD_WORD_COUNT * 4;

   layout.hit_record_base_bytes =
      alloc_region(&alloc, VT_RT_HIT_RECORD_WORD_COUNT * 4, 16);
   layout.hit_record_size_bytes = VT_RT_HIT_RECORD_WORD_COUNT * 4;

   layout.hit_attrib_base_bytes = alloc_region(&alloc, 32, 16);
   layout.hit_attrib_size_bytes = 32;

   layout.continuation_base_bytes = alloc_region(&alloc, 0, 16);
   layout.continuation_size_bytes = 0;
   layout.total_size_bytes = align(alloc.offset, 16);

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
