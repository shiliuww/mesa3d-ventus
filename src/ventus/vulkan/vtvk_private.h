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
#include "vk_queue.h"

#define VTVK_VENDOR_ID 0x5654
#define VTVK_DEVICE_ID 0x5254
#define VTVK_QUEUE_FAMILY 0
#define VTVK_MAX_DESCRIPTOR_BINDINGS 32
#define VTVK_MAX_BOUND_DESCRIPTOR_SETS 4
#define VTVK_RT_DESCRIPTOR_SLOT_COUNT \
   (VTVK_MAX_DESCRIPTOR_BINDINGS * VTVK_MAX_BOUND_DESCRIPTOR_SETS)
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
};

struct vtvk_queue {
   struct vk_queue vk;
};

struct vtvk_device {
   struct vk_device vk;
};

/* Dense device slots are derived from the API layout, never from the binding
 * number. Copies in sets/pipelines survive destruction of the layout handle. */
struct vtvk_descriptor_layout {
   uint32_t count;
   VkDescriptorSetLayoutBinding bindings[VTVK_MAX_DESCRIPTOR_BINDINGS];
   uint32_t slots[VTVK_MAX_DESCRIPTOR_BINDINGS];
};
struct vtvk_descriptor_binding_info;
struct vtvk_pipeline_layout_data {
   uint32_t count;
   struct vtvk_descriptor_layout sets[VTVK_MAX_BOUND_DESCRIPTOR_SETS];
};
bool vtvk_get_pipeline_layout(VkPipelineLayout layout,
                              struct vtvk_pipeline_layout_data *out);
int vtvk_descriptor_slot(const struct vtvk_descriptor_layout *layout,
                         uint32_t binding);
bool vtvk_descriptor_set_get_slot(VkDescriptorSet set, uint32_t slot,
                                  struct vtvk_descriptor_binding_info *out);
bool vtvk_descriptor_set_layout(VkDescriptorSet set,
                                struct vtvk_descriptor_layout *out);

struct vtvk_sampler {
   VkFilter mag_filter;
   VkFilter min_filter;
   VkSamplerAddressMode address_mode_u;
   VkSamplerAddressMode address_mode_v;
};

void vtvk_cmd_buffer_record_copy(VkCommandBuffer commandBuffer, void *dst,
                                  const void *src, size_t size);

enum vtvk_recorded_command_type {
   VTVK_RECORDED_COMMAND_BUILD_AS,
   VTVK_RECORDED_COMMAND_TRACE_RAYS,
   VTVK_RECORDED_COMMAND_COPY,
};

struct vtvk_recorded_as_build {
   uint32_t info_count;
   VkAccelerationStructureBuildGeometryInfoKHR *infos;
   VkAccelerationStructureGeometryKHR *geometries;
   VkAccelerationStructureBuildRangeInfoKHR *ranges;
};

struct vtvk_recorded_trace_rays {
   VkPipeline pipeline;
   VkDescriptorSet descriptor_set;
   VkDescriptorSet descriptor_sets[VTVK_MAX_BOUND_DESCRIPTOR_SETS];
   uint32_t dynamic_offsets[VTVK_MAX_BOUND_DESCRIPTOR_SETS][VTVK_MAX_DESCRIPTOR_BINDINGS];
   uint8_t push_constants[VTVK_MAX_PUSH_CONSTANT_SIZE];
   uint32_t push_constants_size;
   VkStridedDeviceAddressRegionKHR raygen_sbt;
   VkStridedDeviceAddressRegionKHR miss_sbt;
   VkStridedDeviceAddressRegionKHR hit_sbt;
   VkStridedDeviceAddressRegionKHR callable_sbt;
   bool has_raygen_sbt;
   bool has_miss_sbt;
   bool has_hit_sbt;
   bool has_callable_sbt;
   uint32_t width;
   uint32_t height;
   uint32_t depth;
};

struct vtvk_recorded_command {
   enum vtvk_recorded_command_type type;
   union {
      struct vtvk_recorded_as_build build_as;
      struct vtvk_recorded_trace_rays trace_rays;
      struct { void *dst; const void *src; size_t size; } copy;
   } u;
};

struct vtvk_cmd_buffer {
   struct vk_command_buffer vk;

   VkPipeline bound_rt_pipeline;
   VkDescriptorSet bound_descriptor_sets[VTVK_MAX_BOUND_DESCRIPTOR_SETS];
   uint32_t dynamic_offsets[VTVK_MAX_BOUND_DESCRIPTOR_SETS][VTVK_MAX_DESCRIPTOR_BINDINGS];
   uint8_t push_constants[VTVK_MAX_PUSH_CONSTANT_SIZE];
   uint32_t push_constants_size;

   struct vtvk_recorded_command *recorded_commands;
   uint32_t recorded_command_count;
   uint32_t recorded_command_capacity;
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
   /* Host metadata only; not part of the serialized VTAS ABI. */
   uint32_t bvh_stack_required;
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

VkResult vtvk_enumerate_physical_devices(struct vk_instance *vk_instance);
void vtvk_destroy_physical_device(struct vk_physical_device *vk_pdev);

extern const struct vk_command_buffer_ops vtvk_command_buffer_ops;

VkDeviceAddress vtvk_buffer_get_device_address(VkBuffer buffer);
void *vtvk_buffer_host_ptr(VkBuffer buffer, VkDeviceSize offset,
                           VkDeviceSize size);
void *vtvk_buffer_host_ptr_from_addr(VkDeviceAddress address,
                                     VkDeviceSize size);
bool vtvk_buffer_host_allocation_from_addr(VkDeviceAddress address,
                                           const void **base_out,
                                           VkDeviceSize *size_out,
                                           VkDeviceSize *offset_out);
VkPipeline vtvk_cmd_buffer_bound_rt_pipeline(VkCommandBuffer commandBuffer);
VkDescriptorSet vtvk_cmd_buffer_bound_descriptor_set(VkCommandBuffer commandBuffer,
                                                     uint32_t set);
const void *vtvk_cmd_buffer_push_constants(VkCommandBuffer commandBuffer,
                                           uint32_t *size_out);
VkResult vtvk_cmd_buffer_record_as_build(
   VkCommandBuffer commandBuffer,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos);
VkResult vtvk_cmd_buffer_record_trace_rays(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   uint32_t width, uint32_t height, uint32_t depth);
VkResult vtvk_cmd_buffer_execute(VkCommandBuffer commandBuffer);
VkResult vtvk_execute_build_acceleration_structures(
   struct vtvk_device *device,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos);
VkResult vtvk_execute_trace_rays(const struct vtvk_recorded_trace_rays *record);
VkAccelerationStructureKHR
vtvk_descriptor_set_acceleration_structure(VkDescriptorSet descriptorSet,
                                           uint32_t binding);
VkImage vtvk_descriptor_set_image(VkDescriptorSet descriptorSet,
                                  uint32_t binding);
void *vtvk_descriptor_set_buffer_ptr(VkDescriptorSet descriptorSet,
                                     uint32_t binding,
                                     VkDeviceSize *range_out);
bool vtvk_descriptor_set_get_binding(VkDescriptorSet descriptorSet,
                                     uint32_t binding,
                                     struct vtvk_descriptor_binding_info *out);
void *vtvk_image_host_ptr(VkImage image, VkExtent3D *extent_out,
                          VkFormat *format_out);
struct vtvk_acceleration_structure *
vtvk_find_as_by_address(VkDeviceAddress address);

#ifdef __cplusplus
}
#endif

#endif
