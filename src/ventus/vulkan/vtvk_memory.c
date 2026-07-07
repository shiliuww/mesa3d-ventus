#include "vtvk_memory.h"

#include <string.h>

#include "vk_common_entrypoints.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"

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
VKAPI_ATTR VkDeviceAddress VKAPI_CALL
vtvk_GetBufferDeviceAddressKHR(VkDevice device,
                               const VkBufferDeviceAddressInfo *pInfo)
{
   (void)device;
   return pInfo ? vtvk_buffer_get_device_address(pInfo->buffer) : 0;
}

VKAPI_ATTR uint64_t VKAPI_CALL
vtvk_GetBufferOpaqueCaptureAddressKHR(VkDevice device,
                                      const VkBufferDeviceAddressInfo *pInfo)
{
   (void)device;
   (void)pInfo;
   return 0;
}

VKAPI_ATTR uint64_t VKAPI_CALL
vtvk_GetDeviceMemoryOpaqueCaptureAddressKHR(
   VkDevice device,
   const VkDeviceMemoryOpaqueCaptureAddressInfo *pInfo)
{
   (void)device;
   (void)pInfo;
   return 0;
}
