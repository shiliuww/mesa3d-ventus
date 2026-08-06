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

struct vtvk_image {
   VkExtent3D extent;
   VkFormat format;
   VkDeviceSize size;
   VkDeviceMemory memory;
   VkDeviceSize memory_offset;
   void *map;
   VkDeviceAddress device_address;
};

struct vtvk_image_view {
   VkImage image;
   VkFormat format;
};

#endif
