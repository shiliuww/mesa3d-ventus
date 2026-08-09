#include "vtvk_private.h"

#include "vk_common_entrypoints.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"

struct vtvk_fence {
   bool signaled;
};

struct vtvk_semaphore {
   uint64_t value;
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

static void
vtvk_signal_fence(VkFence _fence)
{
   struct vtvk_fence *fence = vtvk_handle_to_ptr((uint64_t)_fence);
   if (fence)
      fence->signaled = true;
}

static VkResult
vtvk_submit_command_buffer(VkCommandBuffer command_buffer)
{
   struct vtvk_cmd_buffer *cmd =
      vtvk_handle_to_ptr((uint64_t)command_buffer);
   return vtvk_cmd_buffer_execute_rt_commands(cmd);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateFence(VkDevice _device,
                 const VkFenceCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator,
                 VkFence *pFence)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_fence *fence =
      vk_zalloc(alloc, sizeof(*fence), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!fence)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   fence->signaled = pCreateInfo->flags & VK_FENCE_CREATE_SIGNALED_BIT;
   *pFence = (VkFence)vtvk_ptr_to_handle(fence);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyFence(VkDevice _device, VkFence fence,
                  const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)fence));
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_ResetFences(VkDevice device, uint32_t fenceCount, const VkFence *pFences)
{
   for (uint32_t i = 0; i < fenceCount; i++) {
      struct vtvk_fence *fence = vtvk_handle_to_ptr((uint64_t)pFences[i]);
      if (fence)
         fence->signaled = false;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetFenceStatus(VkDevice device, VkFence _fence)
{
   struct vtvk_fence *fence = vtvk_handle_to_ptr((uint64_t)_fence);
   return !fence || fence->signaled ? VK_SUCCESS : VK_NOT_READY;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_WaitForFences(VkDevice device,
                   uint32_t fenceCount,
                   const VkFence *pFences,
                   VkBool32 waitAll,
                   uint64_t timeout)
{
   (void)device;
   (void)waitAll;
   (void)timeout;
   for (uint32_t i = 0; i < fenceCount; i++) {
      struct vtvk_fence *fence = vtvk_handle_to_ptr((uint64_t)pFences[i]);
      if (fence)
         fence->signaled = true;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_QueueSubmit(VkQueue queue,
                 uint32_t submitCount,
                 const VkSubmitInfo *pSubmits,
                 VkFence fence)
{
   (void)queue;
   for (uint32_t submit = 0; submit < submitCount; submit++) {
      for (uint32_t i = 0; i < pSubmits[submit].commandBufferCount; i++) {
         VkResult result = vtvk_submit_command_buffer(
            pSubmits[submit].pCommandBuffers[i]);
         if (result != VK_SUCCESS)
            return result;
      }
   }
   vtvk_signal_fence(fence);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_QueueSubmit2(VkQueue queue,
                  uint32_t submitCount,
                  const VkSubmitInfo2 *pSubmits,
                  VkFence fence)
{
   (void)queue;
   for (uint32_t submit = 0; submit < submitCount; submit++) {
      for (uint32_t i = 0; i < pSubmits[submit].commandBufferInfoCount; i++) {
         VkResult result = vtvk_submit_command_buffer(
            pSubmits[submit].pCommandBufferInfos[i].commandBuffer);
         if (result != VK_SUCCESS)
            return result;
      }
   }
   vtvk_signal_fence(fence);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_QueueBindSparse(VkQueue queue,
                     uint32_t bindInfoCount,
                     const VkBindSparseInfo *pBindInfo,
                     VkFence fence)
{
   (void)queue;
   (void)bindInfoCount;
   (void)pBindInfo;
   vtvk_signal_fence(fence);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateSemaphore(VkDevice _device,
                     const VkSemaphoreCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkSemaphore *pSemaphore)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_semaphore *semaphore =
      vk_zalloc(alloc, sizeof(*semaphore), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!semaphore)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pSemaphore = (VkSemaphore)vtvk_ptr_to_handle(semaphore);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroySemaphore(VkDevice _device,
                      VkSemaphore semaphore,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)semaphore));
}
