#include <assert.h>
#include <stddef.h>
#include <string.h>

#include "ventus/common/ventus_kernel_resource.h"

static struct ventus_kernel_resource_v4
valid_rt_resource(void)
{
   struct ventus_kernel_resource_v4 resource;
   memset(&resource, 0, sizeof(resource));
   resource.version = VENTUS_KERNEL_RESOURCE_V4_VERSION;
   resource.pds_static_bytes = 1024;
   resource.rt_feature_flags =
      VENTUS_RT_RESOURCE_USES_TRAVERSAL |
      VENTUS_RT_RESOURCE_USES_TRACE_CONTINUATION |
      VENTUS_RT_RESOURCE_USES_CALLABLE_CONTINUATION;
   resource.rt_pds_layout_id = VENTUS_RT_PDS_LAYOUT_V5;
   resource.rt_private_layout_id = VENTUS_RT_PRIVATE_LAYOUT_V5;
   resource.rt_hot_pds_offset_per_thread = 128;
   resource.rt_hot_pds_stride_bytes = VENTUS_RT_HOT_PDS_STRIDE_V5;
   resource.rt_continuation_offset_per_thread = 512;
   resource.rt_continuation_bytes_per_thread = 256;
   resource.rt_payload_size_bytes = VENTUS_RT_MAX_PAYLOAD_BYTES_V5;
   resource.rt_callable_data_size_bytes =
      VENTUS_RT_MAX_CALLABLE_DATA_BYTES_V5;
   return resource;
}

int
main(void)
{
   assert(sizeof(struct ventus_kernel_resource_v3) == 56);
   assert(sizeof(struct ventus_kernel_resource_v4) == 96);
   assert(offsetof(struct ventus_kernel_resource_v4, rt_feature_flags) == 0x38);
   assert(offsetof(struct ventus_kernel_resource_v4, reserved_zero) == 0x5c);

   struct ventus_kernel_resource_v4 resource = valid_rt_resource();
   struct ventus_kernel_resource_v4 matching = resource;
   assert(ventus_kernel_resource_v4_rt_equal(&resource, &matching));
   matching.rt_payload_size_bytes--;
   assert(!ventus_kernel_resource_v4_rt_equal(&resource, &matching));
   assert(ventus_kernel_resource_v4_validate(&resource, true) ==
          VENTUS_KERNEL_RESOURCE_V4_VALID);

   resource.reserved_zero = 1;
   assert(ventus_kernel_resource_v4_validate(&resource, true) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_RESERVED);
   resource = valid_rt_resource();
   resource.rt_feature_flags |= 1u << 31;
   assert(ventus_kernel_resource_v4_validate(&resource, true) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_FEATURE_FLAGS);
   resource = valid_rt_resource();
   resource.rt_private_layout_id = 4;
   assert(ventus_kernel_resource_v4_validate(&resource, true) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_LAYOUT);
   resource = valid_rt_resource();
   resource.rt_payload_size_bytes = 97;
   assert(ventus_kernel_resource_v4_validate(&resource, true) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_PAYLOAD);
   resource = valid_rt_resource();
   resource.rt_continuation_offset_per_thread = 256;
   assert(ventus_kernel_resource_v4_validate(&resource, true) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_CONTINUATION);
   resource = valid_rt_resource();
   resource.pds_static_bytes = 700;
   assert(ventus_kernel_resource_v4_validate(&resource, true) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_EXTENT);

   memset(&resource, 0, sizeof(resource));
   resource.version = VENTUS_KERNEL_RESOURCE_V4_VERSION;
   assert(ventus_kernel_resource_v4_validate(&resource, false) ==
          VENTUS_KERNEL_RESOURCE_V4_VALID);
   assert(ventus_kernel_resource_v4_validate(&resource, true) ==
          VENTUS_KERNEL_RESOURCE_V4_RT_REQUIRED);
   return 0;
}
