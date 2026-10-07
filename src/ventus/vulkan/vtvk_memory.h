#ifndef VTVK_MEMORY_H
#define VTVK_MEMORY_H

#include "vtvk_private.h"

static inline bool
vtvk_format_is_srgb(VkFormat format)
{
   return format == VK_FORMAT_R8G8B8A8_SRGB || format == VK_FORMAT_B8G8R8A8_SRGB;
}

static inline bool
vtvk_format_is_texture_color(VkFormat format)
{
   return format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_B8G8R8A8_UNORM ||
          vtvk_format_is_srgb(format);
}

static inline bool
vtvk_srgb_usage_supported(VkImageUsageFlags usage)
{
   return !(usage & ~(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT));
}

static inline bool
vtvk_components_identity(VkComponentMapping mapping)
{
   return (mapping.r == VK_COMPONENT_SWIZZLE_IDENTITY || mapping.r == VK_COMPONENT_SWIZZLE_R) &&
          (mapping.g == VK_COMPONENT_SWIZZLE_IDENTITY || mapping.g == VK_COMPONENT_SWIZZLE_G) &&
          (mapping.b == VK_COMPONENT_SWIZZLE_IDENTITY || mapping.b == VK_COMPONENT_SWIZZLE_B) &&
          (mapping.a == VK_COMPONENT_SWIZZLE_IDENTITY || mapping.a == VK_COMPONENT_SWIZZLE_A);
}

struct vtvk_memory {
   struct vtvk_device *device;
   VkDeviceSize size;
   void *map;
   VkDeviceAddress dva_base;
   VkDeviceSize backend_size;
   VkMemoryAllocateFlags allocate_flags;
   bool backend_allocated;
   struct vtvk_memory *next;
};

struct vtvk_buffer {
   VkDeviceSize size;
   VkBufferUsageFlags usage;
   VkDeviceMemory memory;
   VkDeviceSize memory_offset;
   void *map;
   VkDeviceAddress device_address;
   struct vtvk_buffer *next;
};

#define VTVK_MAX_MIP_LEVELS 15
struct vtvk_image_level {
   uint32_t width, height, row_pitch, depth, slice_pitch;
   VkDeviceSize offset, size;
};

struct vtvk_image {
   VkExtent3D extent;
   VkFormat format;
   VkDeviceSize size;
   VkDeviceMemory memory;
   VkDeviceSize memory_offset;
   void *map;
   VkDeviceAddress device_address;
   VkImageCreateFlags flags;
   VkImageType type;
   uint32_t mip_levels, array_layers;
   struct vtvk_image_level levels[VTVK_MAX_MIP_LEVELS];
};

struct vtvk_image_view {
   VkImage image;
   VkFormat format;
   uint32_t base_mip, level_count, base_layer, layer_count;
   VkImageViewType type;
   VkComponentMapping components;
};

/* Array texture profile does not extend attachment or storage-image paths. */
static inline bool
vtvk_array_usage_supported(VkImageUsageFlags usage)
{
   return !(usage & ~(VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT));
}

#endif
