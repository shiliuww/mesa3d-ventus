#include <assert.h>
#include <stdint.h>

#include "ventus/common/ventus_kernel_metadata.h"

int
main(void)
{
   assert(VENTUS_KNL32_PDS_SIZE_PER_THREAD == 64);
   assert(VENTUS_KNL32_PDS_SIZE_PER_THREAD + sizeof(uint32_t) ==
          VENTUS_KNL32_PDS_STACK_BASE_PER_THREAD);
   assert(VENTUS_KNL32_PDS_STACK_BASE_PER_THREAD == 68);
   assert(VENTUS_KNL32_PDS_STACK_BASE_PER_THREAD + sizeof(uint32_t) ==
          VENTUS_KNL32_MAX_SIZE);
   assert(VENTUS_KNL32_MAX_SIZE == 72);

   assert(VENTUS_KNL64_PDS_SIZE_PER_THREAD == 80);
   assert(VENTUS_KNL64_PDS_SIZE_PER_THREAD + sizeof(uint64_t) ==
          VENTUS_KNL64_PDS_STACK_BASE_PER_THREAD);
   assert(VENTUS_KNL64_PDS_STACK_BASE_PER_THREAD == 88);
   assert(VENTUS_KNL64_PDS_STACK_BASE_PER_THREAD + sizeof(uint64_t) ==
          VENTUS_KNL64_MAX_SIZE);
   assert(VENTUS_KNL64_MAX_SIZE == 96);
   return 0;
}
