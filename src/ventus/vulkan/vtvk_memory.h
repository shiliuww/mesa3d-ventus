#ifndef VTVK_MEMORY_H
#define VTVK_MEMORY_H

#include "vtvk_private.h"

struct vtvk_memory {
   VkDeviceSize size;
   void *map;
};

struct vtvk_buffer {
   VkDeviceSize size;
   VkBufferUsageFlags usage;
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

#endif
