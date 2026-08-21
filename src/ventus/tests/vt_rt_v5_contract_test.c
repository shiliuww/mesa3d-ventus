#include <assert.h>
#include <stdint.h>

#include "ventus/common/ventus_rt_v5.h"

int
main(void)
{
   assert(VENTUS_RT_V5_PDS_SIZE_BYTES == 384);
   assert(VENTUS_RT_V5_PRIVATE_LANE_STRIDE_BYTES == 128);
   assert(VENTUS_RT_V5_PRIVATE_WARP_STRIDE_BYTES == 4096);
   assert(VENTUS_RT_V5_PRIVATE_FRONTIER_0 == 0x00);
   assert(VENTUS_RT_V5_PRIVATE_RAY_TMIN == 0x30);
   assert(VENTUS_RT_V5_PRIVATE_INV_DIRECTION == 0x4c);
   assert(VENTUS_RT_V5_PRIVATE_ACTIVE_AS_BASE == 0x58);
   assert(VENTUS_RT_V5_PRIVATE_DIRECTION == 0x60);
   assert(VENTUS_RT_V5_PRIVATE_INSTANCE_SBT_CONTRIBUTION == 0x6c);
   assert(VENTUS_RT_V5_PRIVATE_INSTANCE_METADATA_REFERENCE == 0x70);
   assert(VENTUS_RT_V5_PRIVATE_INSTANCE_INDEX == 0x78);
   assert(VENTUS_RT_V5_PRIVATE_INSTANCE_CUSTOM_INDEX == 0x7c);
   assert(VENTUS_RT_V5_PDS_CALLBACK_BASE == 0x060);
   assert(VENTUS_RT_V5_PDS_COMMITTED_BASE == 0x0c0);
   assert(VENTUS_RT_V5_PDS_CALL_DATA_BASE == 0x120);
   assert(ventus_rt_v5_compact_result(VENTUS_RT_V5_RETURN_MISS) ==
          0x80000001u);

   uint64_t address = 0;
   assert(ventus_rt_v5_private_address(0, 0x180, 0, 0, &address));
   assert(address == 0x180);
   assert(ventus_rt_v5_private_address(0, 0x180, 1, 31, &address));
   assert(address == 0x2100);
   assert(!ventus_rt_v5_private_address(0, 0x180, 0, 32, &address));
   assert(!ventus_rt_v5_private_address(0, 1, 0, 0, &address));

   struct ventus_rt_v5_private_placement placement;
   assert(ventus_rt_v5_private_placement(384, 1, 8192, &placement));
   assert(placement.region_offset == 384);
   assert(placement.total_bytes == 4096);
   assert(placement.num_lds == 4480);
   assert(ventus_rt_v5_private_placement(385, 2, 16384, &placement));
   assert(placement.region_offset == 512);
   assert(placement.total_bytes == 8192);
   assert(placement.num_lds == 8704);
   assert(!ventus_rt_v5_private_placement(0, 0, 4096, &placement));
   assert(!ventus_rt_v5_private_placement(4096, 1, 4096, &placement));

   struct ventus_rt_v5_pds_backing_layout pds_layout;
   assert(ventus_rt_v5_pds_backing_layout(0x90010000, 384, 256, 32,
                                           &pds_layout));
   assert(pds_layout.compiler_stack_bytes_per_wg == 8192);
   assert(pds_layout.static_bytes_per_wg == 12288);
   assert(pds_layout.total_bytes_per_wg == 20480);
   assert(pds_layout.csr_base == 0x90012000);
   assert(ventus_rt_v5_pds_backing_layout(0x1000, 384, 0, 32,
                                           &pds_layout));
   assert(pds_layout.csr_base == 0x1000);
   assert(!ventus_rt_v5_pds_backing_layout(0, 384, 256, 0,
                                            &pds_layout));
   assert(!ventus_rt_v5_pds_backing_layout(
      UINT64_MAX - 31, 384, 1, 32, &pds_layout));
   return 0;
}
