#include "vtvk_pipeline.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "util/log.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vtvk_driver_bridge.h"
#include "vtvk_entrypoints.h"

static void *
vtvk_handle_to_ptr(uint64_t handle)
{
   return (void *)(uintptr_t)handle;
}

enum vtvk_rt_recorded_command_type {
   VTVK_RT_RECORDED_TRACE_RAYS,
   VTVK_RT_RECORDED_TRACE_RAYS_INDIRECT,
   VTVK_RT_RECORDED_TRACE_RAYS_INDIRECT2,
};

struct vtvk_rt_recorded_command {
   struct vtvk_recorded_command base;
   enum vtvk_rt_recorded_command_type type;
   VkPipeline pipeline;
   VkDescriptorSet descriptor_sets[VTVK_MAX_BOUND_DESCRIPTOR_SETS];
   uint8_t push_constants[VTVK_MAX_PUSH_CONSTANT_SIZE];
   uint32_t push_constants_size;
   VkShaderStageFlags push_constant_stages;
   VkStridedDeviceAddressRegionKHR raygen_sbt;
   VkStridedDeviceAddressRegionKHR miss_sbt;
   VkStridedDeviceAddressRegionKHR hit_sbt;
   VkStridedDeviceAddressRegionKHR callable_sbt;
   VkDeviceAddress indirect_address;
   uint32_t width;
   uint32_t height;
   uint32_t depth;
};

static VkResult vtvk_execute_recorded_rt_command(
   struct vtvk_cmd_buffer *cmd,
   const struct vtvk_recorded_command *command);

static struct vtvk_rt_recorded_command *
vtvk_record_rt_command(VkCommandBuffer command_buffer,
                       enum vtvk_rt_recorded_command_type type)
{
   struct vtvk_cmd_buffer *cmd =
      (struct vtvk_cmd_buffer *)vtvk_handle_to_ptr((uint64_t)command_buffer);
   if (!cmd)
      return nullptr;

   struct vtvk_rt_recorded_command *record =
      (struct vtvk_rt_recorded_command *)vtvk_cmd_buffer_alloc_command(
         cmd, sizeof(*record), vtvk_execute_recorded_rt_command, nullptr);
   if (!record)
      return nullptr;

   record->type = type;
   record->pipeline = cmd->bound_rt_pipeline;
   memcpy(record->descriptor_sets, cmd->bound_descriptor_sets,
          sizeof(record->descriptor_sets));
   memcpy(record->push_constants, cmd->push_constants,
          sizeof(record->push_constants));
   record->push_constants_size = cmd->push_constants_size;
   record->push_constant_stages = cmd->push_constants_stages;

   return record;
}

static VkStridedDeviceAddressRegionKHR
vtvk_copy_sbt_region(const VkStridedDeviceAddressRegionKHR *region)
{
   return region ? *region : VkStridedDeviceAddressRegionKHR{};
}

static bool
vtvk_collect_rt_descriptor(const struct vtvk_descriptor_binding_info *info,
                           uint32_t set,
                           uint32_t binding,
                           std::vector<vtvk_rt_descriptor_source> *out)
{
   struct vtvk_rt_descriptor_source desc = {};
   desc.type = info->type;
   desc.set = set;
   desc.binding = binding;

   switch (info->type) {
   case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
      desc.acceleration_structure = info->acceleration_structure;
      break;
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      desc.image = info->image;
      desc.sampler = info->image_info.sampler;
      break;
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      desc.buffer = info->buffer.buffer;
      desc.offset = info->buffer.offset;
      desc.range = info->buffer.range;
      break;
   default:
      mesa_loge("ventus-vk: unsupported RT descriptor set=%u binding=%u type=%u",
                set, binding, (unsigned)info->type);
      return false;
   }

   out->push_back(desc);
   return true;
}

static bool
vtvk_execute_trace_rays(
   struct vtvk_cmd_buffer *cmd,
   const struct vtvk_rt_recorded_command *record,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   uint32_t width,
   uint32_t height,
   uint32_t depth)
{
   static bool logged;

   struct vtvk_pipeline *pipeline =
      vtvk_pipeline_from_handle(record->pipeline);
   VkAccelerationStructureKHR tlas_handle = VK_NULL_HANDLE;
   VkImage storage_image = VK_NULL_HANDLE;
   VkExtent3D image_extent = {};
   VkFormat image_format = VK_FORMAT_UNDEFINED;
   VkDeviceSize camera_range = 0;
   void *camera_data = nullptr;

   std::vector<vtvk_rt_descriptor_source> descriptors;
   descriptors.reserve(VT_RT_DESCRIPTOR_TABLE_SLOT_COUNT);
   for (uint32_t set = 0; set < VTVK_MAX_BOUND_DESCRIPTOR_SETS; set++) {
      VkDescriptorSet descriptor_set = record->descriptor_sets[set];
      for (uint32_t binding = 0; binding < VTVK_MAX_DESCRIPTOR_BINDINGS;
           binding++) {
         struct vtvk_descriptor_binding_info info = {};
         if (!vtvk_descriptor_set_get_binding(descriptor_set, binding, &info) ||
             !vtvk_collect_rt_descriptor(&info, set, binding, &descriptors))
            continue;

         if (info.type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR &&
             tlas_handle == VK_NULL_HANDLE) {
            tlas_handle = info.acceleration_structure;
         } else if (info.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE &&
                    storage_image == VK_NULL_HANDLE) {
            storage_image = info.image;
         } else if (info.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
                    !camera_data) {
            camera_range = info.buffer.range;
            camera_data =
               vtvk_buffer_host_ptr(info.buffer.buffer, info.buffer.offset,
                                    camera_range);
         }
      }
   }

   struct vtvk_acceleration_structure *tlas =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)tlas_handle);
   void *storage_image_data =
      vtvk_image_host_ptr(storage_image, &image_extent, &image_format);

   const uint32_t push_constant_size = record->push_constants_size;
   const VkShaderStageFlags push_constant_stages =
      record->push_constant_stages;
   const void *push_constant_data =
      push_constant_size ? record->push_constants : nullptr;

   if (!pipeline || !tlas || !tlas->address || !storage_image_data) {
      mesa_loge("ventus-vk: vkCmdTraceRaysKHR missing Spike dispatch state "
                "pipeline=%p tlas=0x%" PRIx64 " image=%p camera=%p",
                (void *)pipeline, tlas ? (uint64_t)tlas->address : 0,
                storage_image_data, camera_data);
      return false;
   }

   if (!logged) {
      logged = true;
      mesa_logi("ventus-vk: vkCmdTraceRaysKHR collected Spike dispatch state "
                "%ux%ux%u tlas=0x%" PRIx64 " image=%ux%u format=%u "
                "descriptors=%zu scratch=%u sbt={raygen:0x%" PRIx64 ",miss:0x%" PRIx64
                ",hit:0x%" PRIx64 "} push=%u",
                width, height, depth, (uint64_t)tlas->address,
                image_extent.width, image_extent.height, image_format,
                descriptors.size(), pipeline->per_lane_scratch_size,
                pRaygenShaderBindingTable ?
                   (uint64_t)pRaygenShaderBindingTable->deviceAddress : 0,
                pMissShaderBindingTable ?
                   (uint64_t)pMissShaderBindingTable->deviceAddress : 0,
                pHitShaderBindingTable ?
                   (uint64_t)pHitShaderBindingTable->deviceAddress : 0,
                push_constant_size);
   }

   struct vtvk_driver_trace_rays_info trace = {};
   trace.device = cmd && cmd->vk.base.device ?
      container_of(cmd->vk.base.device, struct vtvk_device, vk) : nullptr;
   trace.tlas = tlas;
   trace.storage_image_data = storage_image_data;
   trace.image_extent = image_extent;
   trace.image_format = image_format;
   trace.camera_data = camera_data;
   trace.camera_range = camera_range;
   trace.raygen_sbt = pRaygenShaderBindingTable;
   trace.miss_sbt = pMissShaderBindingTable;
   trace.hit_sbt = pHitShaderBindingTable;
   trace.callable_sbt = pCallableShaderBindingTable;
   trace.width = width;
   trace.height = height;
   trace.depth = depth;
   trace.per_lane_scratch_size = pipeline->per_lane_scratch_size;
   trace.hit_attribute_size_bytes = pipeline->hit_attribute_size_bytes;
   trace.continuation_frame_size = pipeline->continuation_frame_size;
   trace.max_ray_recursion_depth = pipeline->max_ray_recursion_depth;
   trace.execution_profile = pipeline->rt_execution_profile;
   trace.stage_role = pipeline->rt_stage_role;
   trace.state_descriptor_set = pipeline->rt_state_descriptor_set;
   trace.state_descriptor_binding = pipeline->rt_state_descriptor_binding;
   trace.rt_uses_rv64 = pipeline->rt_uses_rv64;
   trace.kernel_path = pipeline->generated_kernel_path;
   trace.kernel_entry = pipeline->generated_kernel_entry;
   trace.resume_kernel_paths = pipeline->resume_kernel_paths;
   trace.resume_kernel_entries = pipeline->resume_kernel_entries;
   trace.resume_kernel_count = pipeline->resume_kernel_count;
   trace.callback_kernel_paths = pipeline->callback_kernel_paths;
   trace.callback_kernel_entries = pipeline->callback_kernel_entries;
   trace.callback_kernel_groups = pipeline->callback_kernel_groups;
   trace.callback_kernel_stages = pipeline->callback_kernel_stages;
   trace.callback_kernel_count = pipeline->callback_kernel_count;
   trace.descriptors = descriptors.data();
   trace.descriptor_count = descriptors.size();
   trace.push_constants = push_constant_data;
   trace.push_constants_size = push_constant_size;
   trace.push_constants_stages = push_constant_stages;

   return vtvk_driver_trace_rays(&trace);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdTraceRaysKHR(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   uint32_t width,
   uint32_t height,
   uint32_t depth)
{
   struct vtvk_rt_recorded_command *record = vtvk_record_rt_command(
      commandBuffer, VTVK_RT_RECORDED_TRACE_RAYS);
   if (!record)
      return;

   record->raygen_sbt = vtvk_copy_sbt_region(pRaygenShaderBindingTable);
   record->miss_sbt = vtvk_copy_sbt_region(pMissShaderBindingTable);
   record->hit_sbt = vtvk_copy_sbt_region(pHitShaderBindingTable);
   record->callable_sbt = vtvk_copy_sbt_region(pCallableShaderBindingTable);
   record->width = width;
   record->height = height;
   record->depth = depth;
   mesa_logi("ventus-vk: recorded vkCmdTraceRaysKHR %ux%ux%u", width,
             height, depth);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdTraceRaysIndirectKHR(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   VkDeviceAddress indirectDeviceAddress)
{
   struct vtvk_rt_recorded_command *record = vtvk_record_rt_command(
      commandBuffer, VTVK_RT_RECORDED_TRACE_RAYS_INDIRECT);
   if (!record)
      return;

   record->raygen_sbt = vtvk_copy_sbt_region(pRaygenShaderBindingTable);
   record->miss_sbt = vtvk_copy_sbt_region(pMissShaderBindingTable);
   record->hit_sbt = vtvk_copy_sbt_region(pHitShaderBindingTable);
   record->callable_sbt = vtvk_copy_sbt_region(pCallableShaderBindingTable);
   record->indirect_address = indirectDeviceAddress;
   mesa_logi("ventus-vk: recorded vkCmdTraceRaysIndirectKHR address=0x%" PRIx64,
             (uint64_t)indirectDeviceAddress);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdTraceRaysIndirect2KHR(VkCommandBuffer commandBuffer,
                              VkDeviceAddress indirectDeviceAddress)
{
   struct vtvk_rt_recorded_command *record = vtvk_record_rt_command(
      commandBuffer, VTVK_RT_RECORDED_TRACE_RAYS_INDIRECT2);
   if (!record)
      return;

   record->indirect_address = indirectDeviceAddress;
   mesa_logi("ventus-vk: recorded vkCmdTraceRaysIndirect2KHR address=0x%" PRIx64,
             (uint64_t)indirectDeviceAddress);
}

static bool
vtvk_execute_indirect_trace(struct vtvk_cmd_buffer *cmd,
                            const struct vtvk_rt_recorded_command *record)
{
   if (record->indirect_address & 3) {
      mesa_loge("ventus-vk: indirect RT dispatch address is not 4-byte "
                "aligned: 0x%" PRIx64,
                (uint64_t)record->indirect_address);
      return false;
   }

   const VkTraceRaysIndirectCommandKHR *command =
      (const VkTraceRaysIndirectCommandKHR *)
         vtvk_buffer_host_ptr_from_addr_with_usage(
            record->indirect_address, sizeof(*command),
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
   if (!command) {
      mesa_loge("ventus-vk: indirect RT dispatch address is not a complete "
                "INDIRECT_BUFFER range: 0x%" PRIx64,
                (uint64_t)record->indirect_address);
      return false;
   }

   mesa_logi("ventus-vk: queue submit resolved vkCmdTraceRaysIndirectKHR "
             "%ux%ux%u from 0x%" PRIx64,
             command->width, command->height, command->depth,
             (uint64_t)record->indirect_address);
   return vtvk_execute_trace_rays(
      cmd, record, &record->raygen_sbt, &record->miss_sbt,
      &record->hit_sbt, &record->callable_sbt,
      command->width, command->height, command->depth);
}

static bool
vtvk_execute_indirect2_trace(struct vtvk_cmd_buffer *cmd,
                             const struct vtvk_rt_recorded_command *record)
{
   if (record->indirect_address & 3) {
      mesa_loge("ventus-vk: indirect2 RT dispatch address is not 4-byte "
                "aligned: 0x%" PRIx64,
                (uint64_t)record->indirect_address);
      return false;
   }

   const VkTraceRaysIndirectCommand2KHR *command =
      (const VkTraceRaysIndirectCommand2KHR *)
         vtvk_buffer_host_ptr_from_addr_with_usage(
            record->indirect_address, sizeof(*command),
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
   if (!command) {
      mesa_loge("ventus-vk: indirect2 RT dispatch address is not a complete "
                "INDIRECT_BUFFER range: 0x%" PRIx64,
                (uint64_t)record->indirect_address);
      return false;
   }

   const VkStridedDeviceAddressRegionKHR raygen = {
      .deviceAddress = command->raygenShaderRecordAddress,
      .stride = command->raygenShaderRecordSize,
      .size = command->raygenShaderRecordSize,
   };
   const VkStridedDeviceAddressRegionKHR miss = {
      .deviceAddress = command->missShaderBindingTableAddress,
      .stride = command->missShaderBindingTableStride,
      .size = command->missShaderBindingTableSize,
   };
   const VkStridedDeviceAddressRegionKHR hit = {
      .deviceAddress = command->hitShaderBindingTableAddress,
      .stride = command->hitShaderBindingTableStride,
      .size = command->hitShaderBindingTableSize,
   };
   const VkStridedDeviceAddressRegionKHR callable = {
      .deviceAddress = command->callableShaderBindingTableAddress,
      .stride = command->callableShaderBindingTableStride,
      .size = command->callableShaderBindingTableSize,
   };

   mesa_logi("ventus-vk: vkCmdTraceRaysIndirect2KHR resolved %ux%ux%u and "
             "SBT regions at queue submit from 0x%" PRIx64,
             command->width, command->height, command->depth,
             (uint64_t)record->indirect_address);
   return vtvk_execute_trace_rays(cmd, record, &raygen, &miss, &hit,
                                  &callable, command->width,
                                  command->height, command->depth);
}

static VkResult
vtvk_execute_recorded_rt_command(
   struct vtvk_cmd_buffer *cmd,
   const struct vtvk_recorded_command *command)
{
   const struct vtvk_rt_recorded_command *record =
      (const struct vtvk_rt_recorded_command *)command;
   bool success = false;
   switch (record->type) {
   case VTVK_RT_RECORDED_TRACE_RAYS:
      mesa_logi("ventus-vk: queue submit executing vkCmdTraceRaysKHR "
                "%ux%ux%u", record->width, record->height, record->depth);
      success = vtvk_execute_trace_rays(
         cmd, record, &record->raygen_sbt, &record->miss_sbt,
         &record->hit_sbt, &record->callable_sbt,
         record->width, record->height, record->depth);
      break;
   case VTVK_RT_RECORDED_TRACE_RAYS_INDIRECT:
      success = vtvk_execute_indirect_trace(cmd, record);
      break;
   case VTVK_RT_RECORDED_TRACE_RAYS_INDIRECT2:
      success = vtvk_execute_indirect2_trace(cmd, record);
      break;
   }
   return success ? VK_SUCCESS : VK_ERROR_DEVICE_LOST;
}
