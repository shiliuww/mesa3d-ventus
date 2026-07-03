#ifndef VTVK_PRIVATE_H
#define VTVK_PRIVATE_H

#include "util/macros.h"
#include "vk_alloc.h"
#include "vk_device.h"
#include "vk_command_buffer.h"
#include "vk_instance.h"
#include "vk_log.h"
#include "vk_physical_device.h"
#include "vk_queue.h"

#define VTVK_VENDOR_ID 0x5654
#define VTVK_DEVICE_ID 0x5254
#define VTVK_QUEUE_FAMILY 0
#define VTVK_MAX_DESCRIPTOR_BINDINGS 32
#define VTVK_MAX_BOUND_DESCRIPTOR_SETS 4

#ifdef __cplusplus
extern "C" {
#endif

struct vtvk_instance {
   struct vk_instance vk;
};

struct vtvk_physical_device {
   struct vk_physical_device vk;

   VkMemoryHeap memory_heaps[1];
   VkMemoryType memory_types[1];

   const struct vk_sync_type *sync_types[1];
};

struct vtvk_queue {
   struct vk_queue vk;
};

struct vtvk_device {
   struct vk_device vk;
};

struct vtvk_cmd_buffer {
   struct vk_command_buffer vk;

   VkPipeline bound_rt_pipeline;
   VkDescriptorSet bound_descriptor_sets[VTVK_MAX_BOUND_DESCRIPTOR_SETS];
};

VK_DEFINE_HANDLE_CASTS(vtvk_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(vtvk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_DEFINE_HANDLE_CASTS(vtvk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

VkResult vtvk_enumerate_physical_devices(struct vk_instance *vk_instance);
void vtvk_destroy_physical_device(struct vk_physical_device *vk_pdev);

extern const struct vk_command_buffer_ops vtvk_command_buffer_ops;

VkDeviceAddress vtvk_buffer_get_device_address(VkBuffer buffer);
void *vtvk_buffer_host_ptr(VkBuffer buffer, VkDeviceSize offset,
                           VkDeviceSize size);
void *vtvk_buffer_host_ptr_from_addr(VkDeviceAddress address,
                                     VkDeviceSize size);
VkPipeline vtvk_cmd_buffer_bound_rt_pipeline(VkCommandBuffer commandBuffer);
VkDescriptorSet vtvk_cmd_buffer_bound_descriptor_set(VkCommandBuffer commandBuffer,
                                                     uint32_t set);
VkAccelerationStructureKHR
vtvk_descriptor_set_acceleration_structure(VkDescriptorSet descriptorSet,
                                           uint32_t binding);
VkImage vtvk_descriptor_set_image(VkDescriptorSet descriptorSet,
                                  uint32_t binding);
void *vtvk_descriptor_set_buffer_ptr(VkDescriptorSet descriptorSet,
                                     uint32_t binding,
                                     VkDeviceSize *range_out);
void *vtvk_image_host_ptr(VkImage image, VkExtent3D *extent_out,
                          VkFormat *format_out);

#ifdef __cplusplus
}
#endif

#endif
