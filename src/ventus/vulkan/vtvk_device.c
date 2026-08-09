#include "vtvk_private.h"

#include "wsi_common.h"
#include "vk_util.h"
#include "vtvk_driver_bridge.h"
#include "vtvk_entrypoints.h"

static VkResult
vtvk_queue_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   (void)queue;
   for (uint32_t i = 0; i < submit->command_buffer_count; i++) {
      struct vtvk_cmd_buffer *cmd =
         container_of(submit->command_buffers[i], struct vtvk_cmd_buffer, vk);
      VkResult result = vtvk_cmd_buffer_execute_rt_commands(cmd);
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateDevice(VkPhysicalDevice physicalDevice,
                  const VkDeviceCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *pAllocator,
                  VkDevice *pDevice)
{
   VK_FROM_HANDLE(vtvk_physical_device, physical_device, physicalDevice);

   for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      const VkDeviceQueueCreateInfo *qci = &pCreateInfo->pQueueCreateInfos[i];
      if (qci->queueFamilyIndex != VTVK_QUEUE_FAMILY || qci->queueCount > 1)
         return vk_error(physical_device, VK_ERROR_INITIALIZATION_FAILED);
   }

   struct vtvk_device *device =
      vk_zalloc(pAllocator ? pAllocator : &physical_device->vk.instance->alloc,
                sizeof(*device), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device)
      return vk_error(physical_device, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_device_dispatch_table dispatch_table;
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &vtvk_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &wsi_device_entrypoints, false);

   VkResult result = vk_device_init(&device->vk, &physical_device->vk,
                                    &dispatch_table, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS)
      goto fail_device;

   device->rt_uses_rv64 = physical_device->rt_uses_rv64;
   device->vk.command_buffer_ops = &vtvk_command_buffer_ops;

   for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      const VkDeviceQueueCreateInfo *qci = &pCreateInfo->pQueueCreateInfos[i];

      for (uint32_t q = 0; q < qci->queueCount; q++) {
         struct vtvk_queue *queue =
            vk_zalloc(&device->vk.alloc, sizeof(*queue), 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
         if (!queue) {
            result = vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
            goto fail_queues;
         }

         result = vk_queue_init(&queue->vk, &device->vk, qci, q);
         if (result != VK_SUCCESS) {
            vk_free(&device->vk.alloc, queue);
            goto fail_queues;
         }

         queue->vk.driver_submit = vtvk_queue_submit;
      }
   }

   *pDevice = vtvk_device_to_handle(device);
   return VK_SUCCESS;

fail_queues:
   vk_foreach_queue_safe(queue, &device->vk) {
      struct vtvk_queue *vt_queue = container_of(queue, struct vtvk_queue, vk);
      vk_queue_finish(queue);
      vk_free(&device->vk.alloc, vt_queue);
   }
   vk_device_finish(&device->vk);
fail_device:
   vk_free(pAllocator ? pAllocator : &physical_device->vk.instance->alloc,
           device);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);

   if (!device)
      return;

   vk_foreach_queue_safe(queue, &device->vk) {
      struct vtvk_queue *vt_queue = container_of(queue, struct vtvk_queue, vk);
      vk_queue_finish(queue);
      vk_free(&device->vk.alloc, vt_queue);
   }

   vtvk_driver_backend_finish(device);
   vk_device_finish(&device->vk);
   vk_free(&device->vk.alloc, device);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_QueueWaitIdle(VkQueue queue)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_DeviceWaitIdle(VkDevice device)
{
   return VK_SUCCESS;
}
