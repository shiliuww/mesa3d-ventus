#include <assert.h>
#include <stdint.h>

#include "ventus/common/ventus_rt_v5.h"

int
main(void)
{
   assert(VENTUS_RT_V5_PDS_SIZE_BYTES == 384);
   assert(VENTUS_RT_V5_PRIVATE_LANE_STRIDE_BYTES == 128);
   assert(VENTUS_RT_V5_PRIVATE_WARP_STRIDE_BYTES == 4096);
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
   return 0;
}
