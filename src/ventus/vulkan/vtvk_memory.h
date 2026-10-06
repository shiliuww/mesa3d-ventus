#ifndef VTVK_MEMORY_H
#define VTVK_MEMORY_H

#include "vtvk_private.h"

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
   uint32_t width, height, row_pitch;
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
   uint32_t mip_levels;
   struct vtvk_image_level levels[VTVK_MAX_MIP_LEVELS];
};

struct vtvk_image_view {
   VkImage image;
   VkFormat format;
   uint32_t base_mip, level_count;
};

#endif
