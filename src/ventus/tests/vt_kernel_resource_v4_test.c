#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ventus/common/ventus_kernel_resource.h"

static struct ventus_kernel_resource_v3
valid_v3_resource(void)
{
   struct ventus_kernel_resource_v3 resource;
   memset(&resource, 0, sizeof(resource));
   resource.version = VENTUS_KERNEL_RESOURCE_V3_VERSION;
   resource.flags = VENTUS_KERNEL_RESOURCE_HAS_RECURSION;
   resource.vgpr_used = 48;
   resource.sgpr_used = 36;
   resource.lds_static_bytes = 256;
   resource.pds_static_bytes = 1024;
   resource.lds_stack_peak_bytes = 128;
   resource.pds_stack_peak_bytes = 64;
   return resource;
}

static struct ventus_kernel_resource_v5_build_info
valid_v5_build_info(void)
{
   struct ventus_kernel_resource_v5_build_info info;
   memset(&info, 0, sizeof(info));
   info.feature_flags =
      VENTUS_RT_RESOURCE_USES_TRAVERSAL |
      VENTUS_RT_RESOURCE_USES_TRACE_CONTINUATION |
      VENTUS_RT_RESOURCE_USES_CALLABLE_CONTINUATION;
   info.hot_pds_offset_per_thread = 128;
   info.continuation_offset_per_thread = 512;
   info.continuation_bytes_per_thread = 256;
   info.payload_size_bytes = VENTUS_RT_MAX_PAYLOAD_BYTES_V5;
   info.callable_data_size_bytes =
      VENTUS_RT_MAX_CALLABLE_DATA_BYTES_V5;
   return info;
}

static struct ventus_kernel_resource_v4
valid_rt_resource(void)
{
   const struct ventus_kernel_resource_v3 base = valid_v3_resource();
   const struct ventus_kernel_resource_v5_build_info info =
      valid_v5_build_info();
   struct ventus_kernel_resource_v4 resource;
   memset(&resource, 0xa5, sizeof(resource));
   assert(ventus_kernel_resource_v4_build_v5(&base, &info, &resource) ==
          VENTUS_KERNEL_RESOURCE_V4_VALID);
   return resource;
}

int
main(void)
{
   assert(sizeof(struct ventus_kernel_resource_v3) == 56);
   assert(sizeof(struct ventus_kernel_resource_v4) == 96);
   assert(offsetof(struct ventus_kernel_resource_v4, rt_feature_flags) == 0x38);
   assert(offsetof(struct ventus_kernel_resource_v4, reserved_zero) == 0x5c);

   const struct ventus_kernel_resource_v3 base = valid_v3_resource();
   const struct ventus_kernel_resource_v5_build_info build_info =
      valid_v5_build_info();
   struct ventus_kernel_resource_v4 resource = valid_rt_resource();
   assert(resource.version == VENTUS_KERNEL_RESOURCE_V4_VERSION);
   assert(resource.flags == base.flags);
   assert(resource.vgpr_used == base.vgpr_used);
   assert(resource.sgpr_used == base.sgpr_used);
   assert(resource.lds_static_bytes == base.lds_static_bytes);
   assert(resource.pds_static_bytes == base.pds_static_bytes);
   assert(resource.lds_stack_peak_bytes == base.lds_stack_peak_bytes);
   assert(resource.pds_stack_peak_bytes == base.pds_stack_peak_bytes);
   assert(resource.rt_pds_layout_id == VENTUS_RT_PDS_LAYOUT_V5);
   assert(resource.rt_private_layout_id == VENTUS_RT_PRIVATE_LAYOUT_V5);
   assert(resource.rt_hot_pds_stride_bytes == VENTUS_RT_HOT_PDS_STRIDE_V5);
   assert(resource.reserved_zero == 0);

   struct ventus_kernel_resource_v4 matching = resource;
   assert(ventus_kernel_resource_v4_rt_equal(&resource, &matching));
   assert(ventus_kernel_resource_v4_validate_rt_compatible(
             &resource, &matching, true) == VENTUS_KERNEL_RESOURCE_V4_VALID);
   matching.rt_payload_size_bytes--;
   assert(!ventus_kernel_resource_v4_rt_equal(&resource, &matching));
   assert(ventus_kernel_resource_v4_validate_rt_compatible(
             &resource, &matching, true) ==
          VENTUS_KERNEL_RESOURCE_V4_RT_MISMATCH);
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

   struct ventus_kernel_resource_v5_build_info invalid_info = build_info;
   struct ventus_kernel_resource_v4 unchanged;
   memset(&unchanged, 0x5a, sizeof(unchanged));
   resource = unchanged;
   invalid_info.payload_size_bytes = VENTUS_RT_MAX_PAYLOAD_BYTES_V5 + 1;
   assert(ventus_kernel_resource_v4_build_v5(&base, &invalid_info, &resource) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_PAYLOAD);
   assert(memcmp(&resource, &unchanged, sizeof(resource)) == 0);

   invalid_info = build_info;
   invalid_info.callable_data_size_bytes =
      VENTUS_RT_MAX_CALLABLE_DATA_BYTES_V5 + 1;
   assert(ventus_kernel_resource_v4_build_v5(&base, &invalid_info, &resource) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_CALLABLE_DATA);
   assert(memcmp(&resource, &unchanged, sizeof(resource)) == 0);

   invalid_info = build_info;
   invalid_info.continuation_offset_per_thread = 256;
   assert(ventus_kernel_resource_v4_build_v5(&base, &invalid_info, &resource) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_CONTINUATION);
   assert(memcmp(&resource, &unchanged, sizeof(resource)) == 0);

   invalid_info = build_info;
   invalid_info.continuation_offset_per_thread = 900;
   assert(ventus_kernel_resource_v4_build_v5(&base, &invalid_info, &resource) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_EXTENT);
   assert(memcmp(&resource, &unchanged, sizeof(resource)) == 0);

   invalid_info = build_info;
   invalid_info.continuation_bytes_per_thread = UINT32_MAX;
   assert(ventus_kernel_resource_v4_build_v5(&base, &invalid_info, &resource) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_EXTENT);
   assert(memcmp(&resource, &unchanged, sizeof(resource)) == 0);

   struct ventus_kernel_resource_v4 non_rt;
   memset(&non_rt, 0xa5, sizeof(non_rt));
   assert(ventus_kernel_resource_v4_from_v3(&base, &non_rt) ==
          VENTUS_KERNEL_RESOURCE_V4_VALID);
   assert(non_rt.version == VENTUS_KERNEL_RESOURCE_V4_VERSION);
   assert(non_rt.flags == base.flags);
   assert(!ventus_kernel_resource_v4_has_rt(&non_rt));
   assert(non_rt.reserved_zero == 0);
   assert(ventus_kernel_resource_v4_validate(&non_rt, false) ==
          VENTUS_KERNEL_RESOURCE_V4_VALID);
   assert(ventus_kernel_resource_v4_validate(&non_rt, true) ==
          VENTUS_KERNEL_RESOURCE_V4_RT_REQUIRED);

   struct ventus_kernel_resource_v3 bad_base = base;
   bad_base.version = VENTUS_KERNEL_RESOURCE_V4_VERSION;
   assert(ventus_kernel_resource_v4_from_v3(&bad_base, &resource) ==
          VENTUS_KERNEL_RESOURCE_V4_BAD_VERSION);
   assert(memcmp(&resource, &unchanged, sizeof(resource)) == 0);

   memset(&resource, 0, sizeof(resource));
   resource.version = VENTUS_KERNEL_RESOURCE_V4_VERSION;
   assert(ventus_kernel_resource_v4_validate(&resource, false) ==
          VENTUS_KERNEL_RESOURCE_V4_VALID);
   assert(ventus_kernel_resource_v4_validate(&resource, true) ==
          VENTUS_KERNEL_RESOURCE_V4_RT_REQUIRED);
   return 0;
}
