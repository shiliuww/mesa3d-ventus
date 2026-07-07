#include "vtvk_private.h"

#include "vk_common_entrypoints.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"

struct vtvk_pipeline_cache {
   uint32_t unused;
};
static void *
vtvk_handle_to_ptr(uint64_t handle)
{
   return (void *)(uintptr_t)handle;
}

static uint64_t
vtvk_ptr_to_handle(void *ptr)
{
   return (uint64_t)(uintptr_t)ptr;
}
VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateShaderModule(VkDevice device,
                        const VkShaderModuleCreateInfo *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator,
                        VkShaderModule *pShaderModule)
{
   return vk_common_CreateShaderModule(device, pCreateInfo, pAllocator,
                                       pShaderModule);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyShaderModule(VkDevice device,
                         VkShaderModule shaderModule,
                         const VkAllocationCallbacks *pAllocator)
{
   vk_common_DestroyShaderModule(device, shaderModule, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreatePipelineCache(VkDevice _device,
                         const VkPipelineCacheCreateInfo *pCreateInfo,
                         const VkAllocationCallbacks *pAllocator,
                         VkPipelineCache *pPipelineCache)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   (void)pCreateInfo;

   struct vtvk_pipeline_cache *cache =
      vk_zalloc(alloc, sizeof(*cache), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cache)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pPipelineCache = (VkPipelineCache)vtvk_ptr_to_handle(cache);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyPipelineCache(VkDevice _device,
                          VkPipelineCache pipelineCache,
                          const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)pipelineCache));
}

