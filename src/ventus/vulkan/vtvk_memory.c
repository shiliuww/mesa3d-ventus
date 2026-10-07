#include "vtvk_memory.h"

#include <inttypes.h>
#include <string.h>

#include "util/log.h"
#include "vk_common_entrypoints.h"
#include "vk_util.h"
#include "vtvk_driver_bridge.h"
#include "vtvk_entrypoints.h"

static struct vtvk_buffer *vtvk_buffer_list;

uint32_t
vtvk_format_bytes_per_pixel(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_B8G8R8A8_SNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_R8G8B8A8_SRGB:
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
   /* Device addresses name bound backing storage, not the adjacent Vulkan
    * wrapper objects.  Using a wrapper address with buffer->size creates
    * overlapping fake ranges and can resolve an address to the wrong buffer.
    */
   if (!buffer ||
       !(buffer->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ||
       !buffer->memory)
      return 0;
   struct vtvk_memory *mem =
      vtvk_handle_to_ptr((uint64_t)buffer->memory);
   if (!mem ||
       !(mem->allocate_flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT))
      return 0;
   return buffer->device_address;
}

VkDeviceAddress
vtvk_buffer_get_bound_device_address(VkBuffer _buffer)
{
   struct vtvk_buffer *buffer = vtvk_handle_to_ptr((uint64_t)_buffer);
   return buffer ? buffer->device_address : 0;
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
vtvk_buffer_host_ptr_from_addr_with_usage(
   VkDeviceAddress address, VkDeviceSize size,
   VkBufferUsageFlags required_usage)
{
   for (struct vtvk_buffer *buffer = vtvk_buffer_list; buffer;
        buffer = buffer->next) {
      if (!buffer->map ||
          (buffer->usage & required_usage) != required_usage)
         continue;

      VkDeviceAddress base = buffer->device_address;
      if (!base)
         continue;
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
vtvk_buffer_host_ptr_from_addr(VkDeviceAddress address, VkDeviceSize size)
{
   return vtvk_buffer_host_ptr_from_addr_with_usage(address, size, 0);
}

bool
vtvk_buffer_resolve_device_address(VkDeviceAddress address,
                                   void **host_base_out,
                                   VkDeviceAddress *host_address_out,
                                   VkDeviceSize *size_out,
                                   VkDeviceSize *offset_out)
{
   for (struct vtvk_buffer *buffer = vtvk_buffer_list; buffer;
        buffer = buffer->next) {
      if (!buffer->map)
         continue;

      VkDeviceAddress base = buffer->device_address;
      if (!base)
         continue;
      if (address < base || address - base >= buffer->size)
         continue;

      if (host_base_out)
         *host_base_out = buffer->map;
      if (host_address_out)
         *host_address_out = base;
      if (size_out)
         *size_out = buffer->size;
      if (offset_out)
         *offset_out = address - base;
      return true;
   }

   return false;
}

bool
vtvk_buffer_host_allocation_from_addr(VkDeviceAddress address,
                                      const void **base_out,
                                      VkDeviceSize *size_out,
                                      VkDeviceSize *offset_out)
{
   void *host_base = NULL;
   if (!vtvk_buffer_resolve_device_address(address, &host_base, NULL,
                                           size_out, offset_out))
      return false;
   if (base_out)
      *base_out = host_base;
   return true;
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

VkDeviceAddress
vtvk_image_get_device_address(VkImage _image)
{
   struct vtvk_image *image = vtvk_handle_to_ptr((uint64_t)_image);
   return image ? image->device_address : 0;
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
   mem->device = device;
   const VkMemoryAllocateFlagsInfo *flags_info =
      vk_find_struct_const(pAllocateInfo->pNext, MEMORY_ALLOCATE_FLAGS_INFO);
   mem->allocate_flags = flags_info ? flags_info->flags : 0;
   if (mem->allocate_flags &
       VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT) {
      vk_free(alloc, mem);
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   mem->map = vk_alloc(alloc, mem->size ? mem->size : 1, 16,
                       VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!mem->map) {
      vk_free(alloc, mem);
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   mem->backend_size = mem->size ? mem->size : 1;
   if (!vtvk_driver_memory_alloc(device, mem->backend_size, &mem->dva_base)) {
      vk_free(alloc, mem->map);
      vk_free(alloc, mem);
      return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   mem->backend_allocated = true;
   mem->next = device->memory_list;
   device->memory_list = mem;

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

   struct vtvk_memory **link = &device->memory_list;
   while (*link) {
      if (*link == mem) {
         *link = mem->next;
         break;
      }
      link = &(*link)->next;
   }
   if (mem->backend_allocated &&
       !vtvk_driver_memory_free(device, mem->backend_size, mem->dva_base))
      mesa_loge("ventus-vk: failed to release device memory at 0x%" PRIx64,
                (uint64_t)mem->dva_base);
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
   buffer->usage = pCreateInfo->usage;
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

   const bool color = vtvk_format_is_texture_color(pCreateInfo->format);
   const bool volume = pCreateInfo->imageType == VK_IMAGE_TYPE_3D;
   const uint32_t max_dimension = volume ? 2048 : 16384;
   uint32_t max_levels = 1;
   uint32_t dimension = MAX3(pCreateInfo->extent.width, pCreateInfo->extent.height, pCreateInfo->extent.depth);
   while (dimension > 1) { dimension >>= 1; ++max_levels; }
   if (pCreateInfo->pNext || (pCreateInfo->flags & ~VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ||
       ((pCreateInfo->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
        (volume || !color || pCreateInfo->extent.width != pCreateInfo->extent.height ||
         pCreateInfo->arrayLayers < 6 || !vtvk_array_usage_supported(pCreateInfo->usage))) ||
       (pCreateInfo->imageType != VK_IMAGE_TYPE_2D && !volume) ||
       (volume && (!color || pCreateInfo->arrayLayers != 1 ||
                   !vtvk_array_usage_supported(pCreateInfo->usage))) ||
       !pCreateInfo->arrayLayers || pCreateInfo->arrayLayers > 2048 ||
       (pCreateInfo->arrayLayers > 1 &&
        (!color || !vtvk_array_usage_supported(pCreateInfo->usage))) ||
       pCreateInfo->samples != VK_SAMPLE_COUNT_1_BIT ||
       !pCreateInfo->extent.width || !pCreateInfo->extent.height ||
       pCreateInfo->extent.width > max_dimension || pCreateInfo->extent.height > max_dimension ||
       !pCreateInfo->extent.depth || pCreateInfo->extent.depth > max_dimension ||
       (!volume && pCreateInfo->extent.depth != 1) || !pCreateInfo->mipLevels ||
       pCreateInfo->mipLevels > max_levels ||
       pCreateInfo->mipLevels > VTVK_MAX_MIP_LEVELS ||
       (!color && pCreateInfo->mipLevels != 1) ||
       (vtvk_format_is_srgb(pCreateInfo->format) &&
        !vtvk_srgb_usage_supported(pCreateInfo->usage)))
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   struct vtvk_image *image =
      vk_zalloc(alloc, sizeof(*image), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!image)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   image->extent = pCreateInfo->extent;
   image->format = pCreateInfo->format;
   image->mip_levels = pCreateInfo->mipLevels;
   image->array_layers = pCreateInfo->arrayLayers;
   image->flags = pCreateInfo->flags;
   image->type = pCreateInfo->imageType;
   uint32_t width = image->extent.width, height = image->extent.height, depth = image->extent.depth;
   for (uint32_t level = 0; level < image->mip_levels; ++level) {
      struct vtvk_image_level *mip = &image->levels[level];
      mip->width = width;
      mip->height = height;
      mip->row_pitch = width * vtvk_format_bytes_per_pixel(image->format);
      mip->offset = image->size;
      mip->depth = depth;
      mip->slice_pitch = mip->row_pitch * height;
      mip->size = (VkDeviceSize)mip->slice_pitch * depth;
      image->size += mip->size * image->array_layers;
      width = MAX2(1, width >> 1);
      height = MAX2(1, height >> 1);
      depth = MAX2(1, depth >> 1);
   }
   if ((volume || image->array_layers > 1) && image->size > UINT32_MAX) {
      vk_free(alloc, image);
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }
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
   struct vtvk_image *image = vtvk_handle_to_ptr((uint64_t)pCreateInfo->image);
   const VkImageSubresourceRange *range = &pCreateInfo->subresourceRange;
   const uint32_t count = image && range->baseMipLevel < image->mip_levels &&
      range->levelCount == VK_REMAINING_MIP_LEVELS ?
      image->mip_levels - range->baseMipLevel : range->levelCount;
   const uint32_t layers = image && range->baseArrayLayer < image->array_layers &&
      range->layerCount == VK_REMAINING_ARRAY_LAYERS ?
      image->array_layers - range->baseArrayLayer : range->layerCount;
   const bool color = image && vtvk_format_is_texture_color(image->format);
   if (!image || pCreateInfo->pNext || pCreateInfo->flags ||
       (pCreateInfo->viewType != VK_IMAGE_VIEW_TYPE_2D &&
        pCreateInfo->viewType != VK_IMAGE_VIEW_TYPE_2D_ARRAY &&
        pCreateInfo->viewType != VK_IMAGE_VIEW_TYPE_CUBE &&
        pCreateInfo->viewType != VK_IMAGE_VIEW_TYPE_3D) ||
       ((pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_3D) != (image->type == VK_IMAGE_TYPE_3D)) ||
       pCreateInfo->format != image->format ||
       range->baseArrayLayer >= image->array_layers || !layers ||
       layers > image->array_layers - range->baseArrayLayer ||
       (pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_2D && layers != 1) ||
       (pCreateInfo->viewType == VK_IMAGE_VIEW_TYPE_CUBE &&
        (layers != 6 || !(image->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT))) ||
       (!color && (range->baseArrayLayer || layers != 1 ||
                    pCreateInfo->viewType != VK_IMAGE_VIEW_TYPE_2D)) ||
       range->baseMipLevel >= image->mip_levels || !count ||
       count > image->mip_levels - range->baseMipLevel ||
       (color && range->aspectMask != VK_IMAGE_ASPECT_COLOR_BIT) ||
       (unsigned)pCreateInfo->components.r > VK_COMPONENT_SWIZZLE_A ||
       (unsigned)pCreateInfo->components.g > VK_COMPONENT_SWIZZLE_A ||
       (unsigned)pCreateInfo->components.b > VK_COMPONENT_SWIZZLE_A ||
       (unsigned)pCreateInfo->components.a > VK_COMPONENT_SWIZZLE_A ||
       (!color && !vtvk_components_identity(pCreateInfo->components)))
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   struct vtvk_image_view *view =
      vk_zalloc(alloc, sizeof(*view), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!view)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   view->image = pCreateInfo->image;
   view->format = pCreateInfo->format;
   view->base_mip = range->baseMipLevel;
   view->level_count = count;
   view->base_layer = range->baseArrayLayer;
   view->layer_count = layers;
   view->type = pCreateInfo->viewType;
   view->components = pCreateInfo->components;
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
vtvk_GetImageSubresourceLayout(VkDevice device, VkImage _image,
                              const VkImageSubresource *subresource,
                              VkSubresourceLayout *layout)
{
   struct vtvk_image *image = vtvk_handle_to_ptr((uint64_t)_image);
   *layout = (VkSubresourceLayout){0};
   if (!image || subresource->mipLevel >= image->mip_levels ||
       subresource->arrayLayer >= image->array_layers)
      return;
   const struct vtvk_image_level *mip = &image->levels[subresource->mipLevel];
   *layout = (VkSubresourceLayout){
      .offset = mip->offset + (VkDeviceSize)subresource->arrayLayer * mip->size,
      .size = mip->size,
      .rowPitch = mip->row_pitch,
      .arrayPitch = mip->size,
      .depthPitch = mip->slice_pitch,
   };
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

      if (!buffer || !mem || pBindInfos[i].memoryOffset > mem->size ||
          buffer->size > mem->size - pBindInfos[i].memoryOffset)
         return VK_ERROR_UNKNOWN;

      buffer->memory = pBindInfos[i].memory;
      buffer->memory_offset = pBindInfos[i].memoryOffset;
      buffer->map = (uint8_t *)mem->map + pBindInfos[i].memoryOffset;
      buffer->device_address = mem->dva_base + pBindInfos[i].memoryOffset;
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
      image->device_address = mem->dva_base + pBindInfos[i].memoryOffset;
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
