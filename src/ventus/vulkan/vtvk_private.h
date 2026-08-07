#ifndef VTVK_PRIVATE_H
#define VTVK_PRIVATE_H

#include <stdint.h>

#include "util/macros.h"
#include "vk_alloc.h"
#include "vk_device.h"
#include "vk_command_buffer.h"
#include "vk_instance.h"
#include "vk_log.h"
#include "vk_physical_device.h"
#include "vk_query_pool.h"
#include "vk_queue.h"
#include "ventus/compiler/vt_rt_descriptor_abi.h"
#include "vtvk_rt_profile.h"

#define VTVK_VENDOR_ID 0x5654
#define VTVK_DEVICE_ID 0x5254
#define VTVK_QUEUE_FAMILY 0
#define VTVK_MAX_DESCRIPTOR_BINDINGS VT_RT_DESCRIPTOR_BINDINGS_PER_SET
#define VTVK_MAX_BOUND_DESCRIPTOR_SETS VT_RT_DESCRIPTOR_SET_COUNT
#define VTVK_MAX_PUSH_CONSTANT_SIZE 256

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
   bool rt_uses_rv64;
};

struct vtvk_queue {
   struct vk_queue vk;
};

struct vtvk_memory;

struct vtvk_device {
   struct vk_device vk;
   bool rt_uses_rv64;
   void *driver_backend;
   struct vtvk_memory *memory_list;
};

struct vtvk_sampler {
   VkFilter mag_filter;
   VkFilter min_filter;
   VkSamplerAddressMode address_mode_u;
   VkSamplerAddressMode address_mode_v;
   VkBool32 compare_enable;
   VkBool32 unnormalized_coordinates;
   float min_lod;
   float max_lod;
};

struct vtvk_cmd_buffer {
   struct vk_command_buffer vk;

   VkPipeline bound_rt_pipeline;
   VkDescriptorSet bound_descriptor_sets[VTVK_MAX_BOUND_DESCRIPTOR_SETS];
   uint8_t push_constants[VTVK_MAX_PUSH_CONSTANT_SIZE];
   uint32_t push_constants_size;
   VkShaderStageFlags push_constants_stages;
};

struct vtvk_acceleration_structure {
   VkAccelerationStructureTypeKHR type;
   VkBuffer buffer;
   VkDeviceSize offset;
   VkDeviceSize size;
   VkDeviceAddress address;
   void *data;
   uint8_t *snapshot;
   VkDeviceSize snapshot_size;
   VkBuildAccelerationStructureFlagsKHR build_flags;
   VkAccelerationStructureTypeKHR built_type;
   bool built;
};

struct vtvk_query {
   uint64_t value;
   bool available;
};

struct vtvk_query_pool {
   struct vk_query_pool vk;
   struct vtvk_query queries[0];
};

struct vtvk_descriptor_binding_info {
   bool valid;
   VkDescriptorType type;
   VkAccelerationStructureKHR acceleration_structure;
   VkImage image;
   VkDescriptorImageInfo image_info;
   VkDescriptorBufferInfo buffer;
};

VK_DEFINE_HANDLE_CASTS(vtvk_instance, vk.base, VkInstance,
                       VK_OBJECT_TYPE_INSTANCE)
VK_DEFINE_HANDLE_CASTS(vtvk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)
VK_DEFINE_HANDLE_CASTS(vtvk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)
VK_DEFINE_NONDISP_HANDLE_CASTS(vtvk_query_pool, vk.base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)

VkResult vtvk_enumerate_physical_devices(struct vk_instance *vk_instance);
void vtvk_destroy_physical_device(struct vk_physical_device *vk_pdev);

extern const struct vk_command_buffer_ops vtvk_command_buffer_ops;

VkDeviceAddress vtvk_buffer_get_device_address(VkBuffer buffer);
VkDeviceAddress vtvk_buffer_get_bound_device_address(VkBuffer buffer);
VkDeviceAddress vtvk_image_get_device_address(VkImage image);
void *vtvk_buffer_host_ptr(VkBuffer buffer, VkDeviceSize offset,
                           VkDeviceSize size);
void *vtvk_buffer_host_ptr_from_addr(VkDeviceAddress address,
                                     VkDeviceSize size);
void *vtvk_buffer_host_ptr_from_addr_with_usage(
   VkDeviceAddress address, VkDeviceSize size,
   VkBufferUsageFlags required_usage);
bool vtvk_buffer_resolve_device_address(VkDeviceAddress address,
                                        void **host_base_out,
                                        VkDeviceAddress *host_address_out,
                                        VkDeviceSize *size_out,
                                        VkDeviceSize *offset_out);
bool vtvk_buffer_host_allocation_from_addr(VkDeviceAddress address,
                                           const void **base_out,
                                           VkDeviceSize *size_out,
                                           VkDeviceSize *offset_out);
VkPipeline vtvk_cmd_buffer_bound_rt_pipeline(VkCommandBuffer commandBuffer);
VkDescriptorSet vtvk_cmd_buffer_bound_descriptor_set(VkCommandBuffer commandBuffer,
                                                     uint32_t set);
const void *vtvk_cmd_buffer_push_constants(VkCommandBuffer commandBuffer,
                                           uint32_t *size_out,
                                           VkShaderStageFlags *stages_out);
bool vtvk_descriptor_set_get_binding(VkDescriptorSet descriptorSet,
                                     uint32_t binding,
                                     struct vtvk_descriptor_binding_info *out);
void *vtvk_image_host_ptr(VkImage image, VkExtent3D *extent_out,
                          VkFormat *format_out);
uint32_t vtvk_format_bytes_per_pixel(VkFormat format);
bool vtvk_sampler_is_linear_repeat(VkSampler sampler);
struct vtvk_acceleration_structure *
vtvk_find_as_by_address(VkDeviceAddress address);

#ifdef __cplusplus
}
#endif

#endif
