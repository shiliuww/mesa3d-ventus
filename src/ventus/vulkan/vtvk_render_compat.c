#include "vtvk_private.h"
#include "vtvk_memory.h"

#include <string.h>

#include "vk_common_entrypoints.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"

struct vtvk_sampler {
   uint32_t unused;
};

struct vtvk_render_pass {
   uint32_t attachment_count;
   uint32_t subpass_count;
};

struct vtvk_framebuffer {
   VkRenderPass render_pass;
   uint32_t attachment_count;
   uint32_t width;
   uint32_t height;
   uint32_t layers;
};
struct vtvk_swapchain {
   uint32_t image_count;
   VkImage images[3];
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
vtvk_CreateSampler(VkDevice _device,
                   const VkSamplerCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator,
                   VkSampler *pSampler)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   (void)pCreateInfo;

   struct vtvk_sampler *sampler =
      vk_zalloc(alloc, sizeof(*sampler), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!sampler)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pSampler = (VkSampler)vtvk_ptr_to_handle(sampler);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroySampler(VkDevice _device,
                    VkSampler sampler,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)sampler));
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateRenderPass(VkDevice _device,
                      const VkRenderPassCreateInfo *pCreateInfo,
                      const VkAllocationCallbacks *pAllocator,
                      VkRenderPass *pRenderPass)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;

   struct vtvk_render_pass *render_pass =
      vk_zalloc(alloc, sizeof(*render_pass), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!render_pass)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   render_pass->attachment_count = pCreateInfo->attachmentCount;
   render_pass->subpass_count = pCreateInfo->subpassCount;
   *pRenderPass = (VkRenderPass)vtvk_ptr_to_handle(render_pass);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateRenderPass2(VkDevice _device,
                       const VkRenderPassCreateInfo2 *pCreateInfo,
                       const VkAllocationCallbacks *pAllocator,
                       VkRenderPass *pRenderPass)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;

   struct vtvk_render_pass *render_pass =
      vk_zalloc(alloc, sizeof(*render_pass), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!render_pass)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   render_pass->attachment_count = pCreateInfo->attachmentCount;
   render_pass->subpass_count = pCreateInfo->subpassCount;
   *pRenderPass = (VkRenderPass)vtvk_ptr_to_handle(render_pass);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyRenderPass(VkDevice _device,
                       VkRenderPass renderPass,
                       const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)renderPass));
}

VKAPI_ATTR void VKAPI_CALL
vtvk_GetRenderAreaGranularity(VkDevice device,
                              VkRenderPass renderPass,
                              VkExtent2D *pGranularity)
{
   (void)device;
   (void)renderPass;
   *pGranularity = (VkExtent2D){ 1, 1 };
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateFramebuffer(VkDevice _device,
                       const VkFramebufferCreateInfo *pCreateInfo,
                       const VkAllocationCallbacks *pAllocator,
                       VkFramebuffer *pFramebuffer)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;

   struct vtvk_framebuffer *framebuffer =
      vk_zalloc(alloc, sizeof(*framebuffer), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!framebuffer)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   framebuffer->render_pass = pCreateInfo->renderPass;
   framebuffer->attachment_count = pCreateInfo->attachmentCount;
   framebuffer->width = pCreateInfo->width;
   framebuffer->height = pCreateInfo->height;
   framebuffer->layers = pCreateInfo->layers;
   *pFramebuffer = (VkFramebuffer)vtvk_ptr_to_handle(framebuffer);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyFramebuffer(VkDevice _device,
                        VkFramebuffer framebuffer,
                        const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)framebuffer));
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateSwapchainKHR(VkDevice _device,
                        const VkSwapchainCreateInfoKHR *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator,
                        VkSwapchainKHR *pSwapchain)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;

   struct vtvk_swapchain *swapchain =
      vk_zalloc(alloc, sizeof(*swapchain), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!swapchain)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   swapchain->image_count = pCreateInfo->minImageCount;
   if (swapchain->image_count < 2)
      swapchain->image_count = 2;
   if (swapchain->image_count > ARRAY_SIZE(swapchain->images))
      swapchain->image_count = ARRAY_SIZE(swapchain->images);

   for (uint32_t i = 0; i < swapchain->image_count; i++) {
      VkImageCreateInfo ici = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D,
         .format = pCreateInfo->imageFormat,
         .extent = { pCreateInfo->imageExtent.width,
                     pCreateInfo->imageExtent.height, 1 },
         .mipLevels = 1,
         .arrayLayers = pCreateInfo->imageArrayLayers,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = pCreateInfo->imageUsage,
      };
      VkResult result = vtvk_CreateImage(_device, &ici, pAllocator,
                                         &swapchain->images[i]);
      if (result != VK_SUCCESS) {
         for (uint32_t j = 0; j < i; j++)
            vtvk_DestroyImage(_device, swapchain->images[j], pAllocator);
         vk_free(alloc, swapchain);
         return result;
      }
   }

   *pSwapchain = (VkSwapchainKHR)vtvk_ptr_to_handle(swapchain);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroySwapchainKHR(VkDevice _device,
                         VkSwapchainKHR swapchain,
                         const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_swapchain *sc = vtvk_handle_to_ptr((uint64_t)swapchain);

   if (!sc)
      return;

   for (uint32_t i = 0; i < sc->image_count; i++)
      vtvk_DestroyImage(_device, sc->images[i], pAllocator);
   vk_free(alloc, sc);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetSwapchainImagesKHR(VkDevice device,
                           VkSwapchainKHR swapchain,
                           uint32_t *pSwapchainImageCount,
                           VkImage *pSwapchainImages)
{
   struct vtvk_swapchain *sc = vtvk_handle_to_ptr((uint64_t)swapchain);
   VK_OUTARRAY_MAKE_TYPED(VkImage, out, pSwapchainImages,
                          pSwapchainImageCount);

   for (uint32_t i = 0; sc && i < sc->image_count; i++) {
      vk_outarray_append_typed(VkImage, &out, image) {
         *image = sc->images[i];
      }
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_AcquireNextImageKHR(VkDevice device,
                         VkSwapchainKHR swapchain,
                         uint64_t timeout,
                         VkSemaphore semaphore,
                         VkFence fence,
                         uint32_t *pImageIndex)
{
   (void)device;
   (void)swapchain;
   (void)timeout;
   (void)semaphore;
   (void)fence;
   *pImageIndex = 0;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_AcquireNextImage2KHR(VkDevice device,
                          const VkAcquireNextImageInfoKHR *pAcquireInfo,
                          uint32_t *pImageIndex)
{
   return vtvk_AcquireNextImageKHR(device, pAcquireInfo->swapchain,
                                   pAcquireInfo->timeout,
                                   pAcquireInfo->semaphore,
                                   pAcquireInfo->fence, pImageIndex);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pPresentInfo)
{
   (void)queue;
   if (pPresentInfo && pPresentInfo->pResults) {
      for (uint32_t i = 0; i < pPresentInfo->swapchainCount; i++)
         ((VkResult *)pPresentInfo->pResults)[i] = VK_SUCCESS;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdSetViewport(VkCommandBuffer commandBuffer,
                    uint32_t firstViewport,
                    uint32_t viewportCount,
                    const VkViewport *pViewports)
{
   (void)commandBuffer;
   (void)firstViewport;
   (void)viewportCount;
   (void)pViewports;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdSetScissor(VkCommandBuffer commandBuffer,
                   uint32_t firstScissor,
                   uint32_t scissorCount,
                   const VkRect2D *pScissors)
{
   (void)commandBuffer;
   (void)firstScissor;
   (void)scissorCount;
   (void)pScissors;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBindVertexBuffers(VkCommandBuffer commandBuffer,
                          uint32_t firstBinding,
                          uint32_t bindingCount,
                          const VkBuffer *pBuffers,
                          const VkDeviceSize *pOffsets)
{
   (void)commandBuffer;
   (void)firstBinding;
   (void)bindingCount;
   (void)pBuffers;
   (void)pOffsets;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer,
                           uint32_t firstBinding,
                           uint32_t bindingCount,
                           const VkBuffer *pBuffers,
                           const VkDeviceSize *pOffsets,
                           const VkDeviceSize *pSizes,
                           const VkDeviceSize *pStrides)
{
   (void)pSizes;
   (void)pStrides;
   vtvk_CmdBindVertexBuffers(commandBuffer, firstBinding, bindingCount,
                             pBuffers, pOffsets);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBindIndexBuffer(VkCommandBuffer commandBuffer,
                        VkBuffer buffer,
                        VkDeviceSize offset,
                        VkIndexType indexType)
{
   (void)commandBuffer;
   (void)buffer;
   (void)offset;
   (void)indexType;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBindIndexBuffer2(VkCommandBuffer commandBuffer,
                         VkBuffer buffer,
                         VkDeviceSize offset,
                         VkDeviceSize size,
                         VkIndexType indexType)
{
   (void)size;
   vtvk_CmdBindIndexBuffer(commandBuffer, buffer, offset, indexType);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdDraw(VkCommandBuffer commandBuffer,
             uint32_t vertexCount,
             uint32_t instanceCount,
             uint32_t firstVertex,
             uint32_t firstInstance)
{
   (void)commandBuffer;
   (void)vertexCount;
   (void)instanceCount;
   (void)firstVertex;
   (void)firstInstance;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdDrawIndexed(VkCommandBuffer commandBuffer,
                    uint32_t indexCount,
                    uint32_t instanceCount,
                    uint32_t firstIndex,
                    int32_t vertexOffset,
                    uint32_t firstInstance)
{
   (void)commandBuffer;
   (void)indexCount;
   (void)instanceCount;
   (void)firstIndex;
   (void)vertexOffset;
   (void)firstInstance;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdDrawIndirect(VkCommandBuffer commandBuffer,
                     VkBuffer buffer,
                     VkDeviceSize offset,
                     uint32_t drawCount,
                     uint32_t stride)
{
   (void)commandBuffer;
   (void)buffer;
   (void)offset;
   (void)drawCount;
   (void)stride;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer,
                            VkBuffer buffer,
                            VkDeviceSize offset,
                            uint32_t drawCount,
                            uint32_t stride)
{
   (void)commandBuffer;
   (void)buffer;
   (void)offset;
   (void)drawCount;
   (void)stride;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBeginRenderPass(VkCommandBuffer commandBuffer,
                        const VkRenderPassBeginInfo *pRenderPassBegin,
                        VkSubpassContents contents)
{
   (void)commandBuffer;
   (void)pRenderPassBegin;
   (void)contents;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBeginRenderPass2(VkCommandBuffer commandBuffer,
                         const VkRenderPassBeginInfo *pRenderPassBegin,
                         const VkSubpassBeginInfo *pSubpassBeginInfo)
{
   (void)pSubpassBeginInfo;
   vtvk_CmdBeginRenderPass(commandBuffer, pRenderPassBegin,
                           VK_SUBPASS_CONTENTS_INLINE);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdNextSubpass(VkCommandBuffer commandBuffer,
                    VkSubpassContents contents)
{
   (void)commandBuffer;
   (void)contents;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdNextSubpass2(VkCommandBuffer commandBuffer,
                     const VkSubpassBeginInfo *pSubpassBeginInfo,
                     const VkSubpassEndInfo *pSubpassEndInfo)
{
   (void)commandBuffer;
   (void)pSubpassBeginInfo;
   (void)pSubpassEndInfo;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdEndRenderPass(VkCommandBuffer commandBuffer)
{
   (void)commandBuffer;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdEndRenderPass2(VkCommandBuffer commandBuffer,
                       const VkSubpassEndInfo *pSubpassEndInfo)
{
   (void)pSubpassEndInfo;
   vtvk_CmdEndRenderPass(commandBuffer);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBeginRendering(VkCommandBuffer commandBuffer,
                       const VkRenderingInfo *pRenderingInfo)
{
   (void)commandBuffer;
   (void)pRenderingInfo;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   (void)commandBuffer;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdEndRendering2KHR(VkCommandBuffer commandBuffer,
                         const VkRenderingEndInfoKHR *pRenderingEndInfo)
{
   (void)pRenderingEndInfo;
   vtvk_CmdEndRendering(commandBuffer);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdCopyImage(VkCommandBuffer commandBuffer,
                  VkImage srcImage,
                  VkImageLayout srcImageLayout,
                  VkImage dstImage,
                  VkImageLayout dstImageLayout,
                  uint32_t regionCount,
                  const VkImageCopy *pRegions)
{
   (void)commandBuffer;
   (void)srcImageLayout;
   (void)dstImageLayout;
   struct vtvk_image *src = vtvk_handle_to_ptr((uint64_t)srcImage);
   struct vtvk_image *dst = vtvk_handle_to_ptr((uint64_t)dstImage);
   if (!src || !dst || !src->map || !dst->map)
      return;

   for (uint32_t r = 0; r < regionCount; r++) {
      const VkImageCopy *region = &pRegions[r];
      uint32_t width = region->extent.width;
      uint32_t height = region->extent.height;
      uint32_t depth = region->extent.depth ? region->extent.depth : 1;

      for (uint32_t z = 0; z < depth; z++) {
         for (uint32_t y = 0; y < height; y++) {
            VkDeviceSize src_offset =
               (((VkDeviceSize)(region->srcOffset.z + z) * src->extent.height +
                 (region->srcOffset.y + y)) * src->extent.width +
                region->srcOffset.x) * 4;
            VkDeviceSize dst_offset =
               (((VkDeviceSize)(region->dstOffset.z + z) * dst->extent.height +
                 (region->dstOffset.y + y)) * dst->extent.width +
                region->dstOffset.x) * 4;
            VkDeviceSize bytes = (VkDeviceSize)width * 4;
            if (src_offset + bytes <= src->size &&
                dst_offset + bytes <= dst->size)
               memcpy((uint8_t *)dst->map + dst_offset,
                      (uint8_t *)src->map + src_offset, bytes);
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdCopyBuffer(VkCommandBuffer commandBuffer,
                   VkBuffer srcBuffer,
                   VkBuffer dstBuffer,
                   uint32_t regionCount,
                   const VkBufferCopy *pRegions)
{
   (void)commandBuffer;
   for (uint32_t i = 0; i < regionCount; i++) {
      void *src = vtvk_buffer_host_ptr(srcBuffer, pRegions[i].srcOffset,
                                       pRegions[i].size);
      void *dst = vtvk_buffer_host_ptr(dstBuffer, pRegions[i].dstOffset,
                                       pRegions[i].size);
      if (src && dst)
         memcpy(dst, src, pRegions[i].size);
   }
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdCopyBufferToImage(VkCommandBuffer commandBuffer,
                          VkBuffer srcBuffer,
                          VkImage dstImage,
                          VkImageLayout dstImageLayout,
                          uint32_t regionCount,
                          const VkBufferImageCopy *pRegions)
{
   (void)commandBuffer;
   (void)srcBuffer;
   (void)dstImage;
   (void)dstImageLayout;
   (void)regionCount;
   (void)pRegions;
}
