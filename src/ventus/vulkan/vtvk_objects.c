#include "vtvk_private.h"

#include <string.h>

#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"

struct vtvk_memory {
   VkDeviceSize size;
   void *map;
};

struct vtvk_buffer {
   VkDeviceSize size;
   VkDeviceMemory memory;
   VkDeviceSize memory_offset;
   void *map;
   struct vtvk_buffer *next;
};

struct vtvk_image {
   VkExtent3D extent;
   VkFormat format;
   VkDeviceSize size;
   VkDeviceMemory memory;
   VkDeviceSize memory_offset;
   void *map;
};

struct vtvk_image_view {
   VkImage image;
   VkFormat format;
};

struct vtvk_fence {
   bool signaled;
};

struct vtvk_semaphore {
   uint64_t value;
};

struct vtvk_descriptor_set_layout {
   uint32_t binding_count;
};

struct vtvk_pipeline_layout {
   uint32_t set_layout_count;
};

struct vtvk_descriptor_pool {
   uint32_t max_sets;
};

struct vtvk_descriptor_set {
   VkDescriptorSetLayout layout;
   struct {
      VkDescriptorType type;
      VkAccelerationStructureKHR acceleration_structure;
      VkDescriptorImageInfo image;
      VkDescriptorBufferInfo buffer;
   } bindings[VTVK_MAX_DESCRIPTOR_BINDINGS];
};

struct vtvk_swapchain {
   uint32_t image_count;
   VkImage images[3];
};

static struct vtvk_buffer *vtvk_buffer_list;

static uint32_t
vtvk_format_bytes_per_pixel(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_B8G8R8A8_SNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
      return 4;
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16_UNORM:
   case VK_FORMAT_R16G16_SNORM:
   case VK_FORMAT_R16G16_SFLOAT:
      return 4;
   case VK_FORMAT_R16G16B16A16_UNORM:
   case VK_FORMAT_R16G16B16A16_SNORM:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      return 8;
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32G32B32A32_SFLOAT:
      return 16;
   default:
      return 4;
   }
}

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

VkDeviceAddress
vtvk_buffer_get_device_address(VkBuffer _buffer)
{
   struct vtvk_buffer *buffer = vtvk_handle_to_ptr((uint64_t)_buffer);
   return buffer ? (VkDeviceAddress)(uintptr_t)buffer : 0;
}

void *
vtvk_buffer_host_ptr(VkBuffer _buffer, VkDeviceSize offset, VkDeviceSize size)
{
   struct vtvk_buffer *buffer = vtvk_handle_to_ptr((uint64_t)_buffer);
   if (!buffer || !buffer->map || offset > buffer->size)
      return NULL;
   if (size > buffer->size - offset)
      return NULL;
   return (uint8_t *)buffer->map + offset;
}

void *
vtvk_buffer_host_ptr_from_addr(VkDeviceAddress address, VkDeviceSize size)
{
   for (struct vtvk_buffer *buffer = vtvk_buffer_list; buffer;
        buffer = buffer->next) {
      VkDeviceAddress base = (VkDeviceAddress)(uintptr_t)buffer;
      if (address < base)
         continue;

      VkDeviceSize offset = address - base;
      if (offset > buffer->size || size > buffer->size - offset)
         continue;

      return vtvk_buffer_host_ptr((VkBuffer)(uintptr_t)buffer, offset, size);
   }

   return NULL;
}

VkPipeline
vtvk_cmd_buffer_bound_rt_pipeline(VkCommandBuffer commandBuffer)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   return cmd ? cmd->bound_rt_pipeline : VK_NULL_HANDLE;
}

VkDescriptorSet
vtvk_cmd_buffer_bound_descriptor_set(VkCommandBuffer commandBuffer,
                                     uint32_t set)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   if (!cmd || set >= VTVK_MAX_BOUND_DESCRIPTOR_SETS)
      return VK_NULL_HANDLE;
   return cmd->bound_descriptor_sets[set];
}

VkAccelerationStructureKHR
vtvk_descriptor_set_acceleration_structure(VkDescriptorSet descriptorSet,
                                           uint32_t binding)
{
   struct vtvk_descriptor_set *set =
      vtvk_handle_to_ptr((uint64_t)descriptorSet);
   if (!set || binding >= VTVK_MAX_DESCRIPTOR_BINDINGS)
      return VK_NULL_HANDLE;

   return set->bindings[binding].acceleration_structure;
}

VkImage
vtvk_descriptor_set_image(VkDescriptorSet descriptorSet, uint32_t binding)
{
   struct vtvk_descriptor_set *set =
      vtvk_handle_to_ptr((uint64_t)descriptorSet);
   if (!set || binding >= VTVK_MAX_DESCRIPTOR_BINDINGS)
      return VK_NULL_HANDLE;

   struct vtvk_image_view *view =
      vtvk_handle_to_ptr((uint64_t)set->bindings[binding].image.imageView);
   return view ? view->image : VK_NULL_HANDLE;
}

void *
vtvk_descriptor_set_buffer_ptr(VkDescriptorSet descriptorSet, uint32_t binding,
                               VkDeviceSize *range_out)
{
   struct vtvk_descriptor_set *set =
      vtvk_handle_to_ptr((uint64_t)descriptorSet);
   if (!set || binding >= VTVK_MAX_DESCRIPTOR_BINDINGS)
      return NULL;

   VkDescriptorBufferInfo *info = &set->bindings[binding].buffer;
   VkDeviceSize range = info->range;
   if (range == VK_WHOLE_SIZE) {
      struct vtvk_buffer *buffer = vtvk_handle_to_ptr((uint64_t)info->buffer);
      range = buffer && info->offset <= buffer->size ?
                 buffer->size - info->offset : 0;
   }
   if (range_out)
      *range_out = range;
   return vtvk_buffer_host_ptr(info->buffer, info->offset, range);
}

void *
vtvk_image_host_ptr(VkImage _image, VkExtent3D *extent_out,
                    VkFormat *format_out)
{
   struct vtvk_image *image = vtvk_handle_to_ptr((uint64_t)_image);
   if (!image || !image->map)
      return NULL;

   if (extent_out)
      *extent_out = image->extent;
   if (format_out)
      *format_out = image->format;
   return image->map;
}

static void
vtvk_signal_fence(VkFence _fence)
{
   struct vtvk_fence *fence = vtvk_handle_to_ptr((uint64_t)_fence);
   if (fence)
      fence->signaled = true;
}

static VkResult
vtvk_cmd_buffer_create(struct vk_command_pool *pool,
                       VkCommandBufferLevel level,
                       struct vk_command_buffer **cmd_buffer_out)
{
   struct vtvk_cmd_buffer *cmd =
      vk_zalloc(&pool->alloc, sizeof(*cmd), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_command_buffer_init(pool, &cmd->vk, &vtvk_command_buffer_ops, level);
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }

   *cmd_buffer_out = &cmd->vk;
   return VK_SUCCESS;
}

static void
vtvk_cmd_buffer_reset(struct vk_command_buffer *cmd_buffer,
                      VkCommandBufferResetFlags flags)
{
   (void)flags;
   vk_command_buffer_reset(cmd_buffer);
}

static void
vtvk_cmd_buffer_destroy(struct vk_command_buffer *cmd_buffer)
{
   struct vtvk_cmd_buffer *cmd =
      container_of(cmd_buffer, struct vtvk_cmd_buffer, vk);
   struct vk_command_pool *pool = cmd_buffer->pool;

   vk_command_buffer_finish(cmd_buffer);
   vk_free(&pool->alloc, cmd);
}

const struct vk_command_buffer_ops vtvk_command_buffer_ops = {
   .create = vtvk_cmd_buffer_create,
   .reset = vtvk_cmd_buffer_reset,
   .destroy = vtvk_cmd_buffer_destroy,
};

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_AllocateMemory(VkDevice _device,
                    const VkMemoryAllocateInfo *pAllocateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkDeviceMemory *pMemory)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;

   struct vtvk_memory *mem =
      vk_zalloc(alloc, sizeof(*mem), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!mem)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   mem->size = pAllocateInfo->allocationSize;
   mem->map = vk_alloc(alloc, mem->size ? mem->size : 1, 16,
                       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!mem->map) {
      vk_free(alloc, mem);
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   *pMemory = (VkDeviceMemory)vtvk_ptr_to_handle(mem);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_FreeMemory(VkDevice _device, VkDeviceMemory memory,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_memory *mem = vtvk_handle_to_ptr((uint64_t)memory);

   if (!mem)
      return;

   vk_free(alloc, mem->map);
   vk_free(alloc, mem);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_MapMemory2KHR(VkDevice device,
                   const VkMemoryMapInfo *pMemoryMapInfo,
                   void **ppData)
{
   struct vtvk_memory *mem =
      vtvk_handle_to_ptr((uint64_t)pMemoryMapInfo->memory);

   if (!mem || pMemoryMapInfo->offset > mem->size)
      return VK_ERROR_MEMORY_MAP_FAILED;

   *ppData = (char *)mem->map + pMemoryMapInfo->offset;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_UnmapMemory2KHR(VkDevice device,
                     const VkMemoryUnmapInfo *pMemoryUnmapInfo)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateBuffer(VkDevice _device,
                  const VkBufferCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *pAllocator,
                  VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;

   struct vtvk_buffer *buffer =
      vk_zalloc(alloc, sizeof(*buffer), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!buffer)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   buffer->size = pCreateInfo->size;
   buffer->next = vtvk_buffer_list;
   vtvk_buffer_list = buffer;
   *pBuffer = (VkBuffer)vtvk_ptr_to_handle(buffer);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyBuffer(VkDevice _device, VkBuffer buffer,
                   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_buffer *victim = vtvk_handle_to_ptr((uint64_t)buffer);
   struct vtvk_buffer **link = &vtvk_buffer_list;
   while (*link) {
      if (*link == victim) {
         *link = victim->next;
         break;
      }
      link = &(*link)->next;
   }
   vk_free(alloc, victim);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateImage(VkDevice _device,
                 const VkImageCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator,
                 VkImage *pImage)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;

   struct vtvk_image *image =
      vk_zalloc(alloc, sizeof(*image), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!image)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   image->extent = pCreateInfo->extent;
   image->format = pCreateInfo->format;
   image->size =
      (VkDeviceSize)image->extent.width * image->extent.height *
      image->extent.depth * vtvk_format_bytes_per_pixel(image->format);
   *pImage = (VkImage)vtvk_ptr_to_handle(image);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyImage(VkDevice _device, VkImage image,
                  const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)image));
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateImageView(VkDevice _device,
                     const VkImageViewCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkImageView *pView)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_image_view *view =
      vk_zalloc(alloc, sizeof(*view), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!view)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   view->image = pCreateInfo->image;
   view->format = pCreateInfo->format;
   *pView = (VkImageView)vtvk_ptr_to_handle(view);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyImageView(VkDevice _device,
                      VkImageView imageView,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)imageView));
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
   (void)submitCount;
   (void)pSubmits;
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
   (void)submitCount;
   (void)pSubmits;
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
vtvk_CreateDescriptorSetLayout(VkDevice _device,
                               const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                               const VkAllocationCallbacks *pAllocator,
                               VkDescriptorSetLayout *pSetLayout)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_descriptor_set_layout *layout =
      vk_zalloc(alloc, sizeof(*layout), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!layout)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   layout->binding_count = pCreateInfo->bindingCount;
   *pSetLayout = (VkDescriptorSetLayout)vtvk_ptr_to_handle(layout);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyDescriptorSetLayout(VkDevice _device,
                                VkDescriptorSetLayout descriptorSetLayout,
                                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)descriptorSetLayout));
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreatePipelineLayout(VkDevice _device,
                          const VkPipelineLayoutCreateInfo *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkPipelineLayout *pPipelineLayout)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_pipeline_layout *layout =
      vk_zalloc(alloc, sizeof(*layout), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!layout)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   layout->set_layout_count = pCreateInfo->setLayoutCount;
   *pPipelineLayout = (VkPipelineLayout)vtvk_ptr_to_handle(layout);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyPipelineLayout(VkDevice _device,
                           VkPipelineLayout pipelineLayout,
                           const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)pipelineLayout));
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateDescriptorPool(VkDevice _device,
                          const VkDescriptorPoolCreateInfo *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_descriptor_pool *pool =
      vk_zalloc(alloc, sizeof(*pool), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!pool)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   pool->max_sets = pCreateInfo->maxSets;
   *pDescriptorPool = (VkDescriptorPool)vtvk_ptr_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyDescriptorPool(VkDevice _device,
                           VkDescriptorPool descriptorPool,
                           const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)descriptorPool));
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_AllocateDescriptorSets(VkDevice _device,
                            const VkDescriptorSetAllocateInfo *pAllocateInfo,
                            VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      struct vtvk_descriptor_set *set =
         vk_zalloc(&device->vk.alloc, sizeof(*set), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!set)
         return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

      set->layout = pAllocateInfo->pSetLayouts[i];
      pDescriptorSets[i] = (VkDescriptorSet)vtvk_ptr_to_handle(set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_FreeDescriptorSets(VkDevice _device,
                        VkDescriptorPool descriptorPool,
                        uint32_t descriptorSetCount,
                        const VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   (void)descriptorPool;
   for (uint32_t i = 0; i < descriptorSetCount; i++)
      vk_free(&device->vk.alloc, vtvk_handle_to_ptr((uint64_t)pDescriptorSets[i]));
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_UpdateDescriptorSets(VkDevice device,
                          uint32_t descriptorWriteCount,
                          const VkWriteDescriptorSet *pDescriptorWrites,
                          uint32_t descriptorCopyCount,
                          const VkCopyDescriptorSet *pDescriptorCopies)
{
   (void)device;
   for (uint32_t i = 0; i < descriptorWriteCount; i++) {
      const VkWriteDescriptorSet *write = &pDescriptorWrites[i];
      struct vtvk_descriptor_set *set =
         vtvk_handle_to_ptr((uint64_t)write->dstSet);
      if (!set || write->dstBinding >= VTVK_MAX_DESCRIPTOR_BINDINGS ||
          write->descriptorCount == 0)
         continue;

      set->bindings[write->dstBinding].type = write->descriptorType;
      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
         const VkWriteDescriptorSetAccelerationStructureKHR *as_info =
            vk_find_struct_const(write->pNext,
                                 WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR);
         if (as_info && as_info->accelerationStructureCount)
            set->bindings[write->dstBinding].acceleration_structure =
               as_info->pAccelerationStructures[0];
         break;
      }
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         if (write->pImageInfo)
            set->bindings[write->dstBinding].image = write->pImageInfo[0];
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         if (write->pBufferInfo)
            set->bindings[write->dstBinding].buffer = write->pBufferInfo[0];
         break;
      default:
         break;
      }
   }

   for (uint32_t i = 0; i < descriptorCopyCount; i++) {
      const VkCopyDescriptorSet *copy = &pDescriptorCopies[i];
      struct vtvk_descriptor_set *src =
         vtvk_handle_to_ptr((uint64_t)copy->srcSet);
      struct vtvk_descriptor_set *dst =
         vtvk_handle_to_ptr((uint64_t)copy->dstSet);
      if (!src || !dst ||
          copy->srcBinding >= VTVK_MAX_DESCRIPTOR_BINDINGS ||
          copy->dstBinding >= VTVK_MAX_DESCRIPTOR_BINDINGS ||
          copy->descriptorCount == 0)
         continue;
      dst->bindings[copy->dstBinding] = src->bindings[copy->srcBinding];
   }
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
vtvk_GetBufferMemoryRequirements2(
   VkDevice device,
   const VkBufferMemoryRequirementsInfo2 *pInfo,
   VkMemoryRequirements2 *pMemoryRequirements)
{
   struct vtvk_buffer *buffer = vtvk_handle_to_ptr((uint64_t)pInfo->buffer);
   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements){
      .size = buffer ? buffer->size : 0,
      .alignment = 64,
      .memoryTypeBits = 1,
   };
}

VKAPI_ATTR void VKAPI_CALL
vtvk_GetImageMemoryRequirements2(
   VkDevice device,
   const VkImageMemoryRequirementsInfo2 *pInfo,
   VkMemoryRequirements2 *pMemoryRequirements)
{
   struct vtvk_image *image = vtvk_handle_to_ptr((uint64_t)pInfo->image);
   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements){
      .size = image ? image->size : 0,
      .alignment = 64,
      .memoryTypeBits = 1,
   };
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_BindBufferMemory2(VkDevice device,
                       uint32_t bindInfoCount,
                       const VkBindBufferMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; i++) {
      struct vtvk_buffer *buffer =
         vtvk_handle_to_ptr((uint64_t)pBindInfos[i].buffer);
      struct vtvk_memory *mem =
         vtvk_handle_to_ptr((uint64_t)pBindInfos[i].memory);

      if (!buffer || !mem || pBindInfos[i].memoryOffset > mem->size)
         return VK_ERROR_UNKNOWN;

      buffer->memory = pBindInfos[i].memory;
      buffer->memory_offset = pBindInfos[i].memoryOffset;
      buffer->map = (uint8_t *)mem->map + pBindInfos[i].memoryOffset;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_BindImageMemory2(VkDevice device,
                      uint32_t bindInfoCount,
                      const VkBindImageMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; i++) {
      struct vtvk_image *image =
         vtvk_handle_to_ptr((uint64_t)pBindInfos[i].image);
      struct vtvk_memory *mem =
         vtvk_handle_to_ptr((uint64_t)pBindInfos[i].memory);

      if (!image || !mem || pBindInfos[i].memoryOffset > mem->size)
         return VK_ERROR_UNKNOWN;

      if (image->size > mem->size - pBindInfos[i].memoryOffset)
         return VK_ERROR_UNKNOWN;

      image->memory = pBindInfos[i].memory;
      image->memory_offset = pBindInfos[i].memoryOffset;
      image->map = (uint8_t *)mem->map + pBindInfos[i].memoryOffset;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                        const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(vk_command_buffer, cmd_buffer, commandBuffer);
   vk_command_buffer_begin(cmd_buffer, pBeginInfo);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(vk_command_buffer, cmd_buffer, commandBuffer);
   return vk_command_buffer_end(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdPipelineBarrier(VkCommandBuffer commandBuffer,
                        VkPipelineStageFlags srcStageMask,
                        VkPipelineStageFlags dstStageMask,
                        VkDependencyFlags dependencyFlags,
                        uint32_t memoryBarrierCount,
                        const VkMemoryBarrier *pMemoryBarriers,
                        uint32_t bufferMemoryBarrierCount,
                        const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                        uint32_t imageMemoryBarrierCount,
                        const VkImageMemoryBarrier *pImageMemoryBarriers)
{
   (void)commandBuffer;
   (void)srcStageMask;
   (void)dstStageMask;
   (void)dependencyFlags;
   (void)memoryBarrierCount;
   (void)pMemoryBarriers;
   (void)bufferMemoryBarrierCount;
   (void)pBufferMemoryBarriers;
   (void)imageMemoryBarrierCount;
   (void)pImageMemoryBarriers;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                         const VkDependencyInfo *pDependencyInfo)
{
   (void)commandBuffer;
   (void)pDependencyInfo;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBindPipeline(VkCommandBuffer commandBuffer,
                     VkPipelineBindPoint pipelineBindPoint,
                     VkPipeline pipeline)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   if (cmd && pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
      cmd->bound_rt_pipeline = pipeline;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBindDescriptorSets(VkCommandBuffer commandBuffer,
                           VkPipelineBindPoint pipelineBindPoint,
                           VkPipelineLayout layout,
                           uint32_t firstSet,
                           uint32_t descriptorSetCount,
                           const VkDescriptorSet *pDescriptorSets,
                           uint32_t dynamicOffsetCount,
                           const uint32_t *pDynamicOffsets)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   (void)layout;
   (void)dynamicOffsetCount;
   (void)pDynamicOffsets;
   if (!cmd || pipelineBindPoint != VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
      return;

   for (uint32_t i = 0; i < descriptorSetCount; i++) {
      uint32_t set = firstSet + i;
      if (set < VTVK_MAX_BOUND_DESCRIPTOR_SETS)
         cmd->bound_descriptor_sets[set] = pDescriptorSets[i];
   }
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
