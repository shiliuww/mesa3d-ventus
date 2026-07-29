/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vt_nir.h"

#include <stdio.h>

static bool
validate_descriptor_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin,
                              void *data)
{
   bool *unsupported = data;
   (void)b;

   switch (intrin->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      if (!vt_rt_descriptor_location_valid(
             nir_intrinsic_desc_set(intrin),
             nir_intrinsic_binding(intrin))) {
         fprintf(stderr,
                 "ventus-nir: RT descriptor set=%u binding=%u exceeds the "
                 "%u-set x %u-binding table\n",
                 nir_intrinsic_desc_set(intrin),
                 nir_intrinsic_binding(intrin),
                 VT_RT_DESCRIPTOR_SET_COUNT,
                 VT_RT_DESCRIPTOR_BINDINGS_PER_SET);
         *unsupported = true;
      }
      break;
   case nir_intrinsic_vulkan_resource_reindex:
      fprintf(stderr,
              "ventus-nir: descriptor arrays/dynamic indexing are not supported "
              "by the RT descriptor ABI\n");
      *unsupported = true;
      break;
   case nir_intrinsic_load_vulkan_descriptor:
      switch (nir_intrinsic_desc_type(intrin)) {
      case nir_descriptor_type_uniform_buffer:
      case nir_descriptor_type_storage_buffer:
      case nir_descriptor_type_acceleration_structure:
         break;
      default:
         fprintf(stderr,
                 "ventus-nir: unsupported RT Vulkan descriptor type %u\n",
                 nir_intrinsic_desc_type(intrin));
         *unsupported = true;
         break;
      }
      break;
   default:
      break;
   }

   return false;
}

bool
vt_nir_lower_descriptors(nir_shader *shader)
{
   bool unsupported = false;
   nir_shader_intrinsics_pass(shader, validate_descriptor_intrinsic,
                              nir_metadata_all, &unsupported);
   return false;
}
