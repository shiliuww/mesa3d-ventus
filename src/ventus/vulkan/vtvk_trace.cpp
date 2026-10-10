#include "vtvk_pipeline.h"

#include <cstdint>
#include <vector>

#include "util/log.h"
#include "vtvk_driver_bridge.h"
#include "vtvk_entrypoints.h"

static bool
vtvk_collect_rt_descriptor(VkDescriptorSet descriptor_set, uint32_t set, uint32_t binding,
                           uint32_t dynamic_offset,
                           std::vector<vtvk_rt_descriptor_source> *out)
{
   struct vtvk_descriptor_binding_info info = {};
   if (!vtvk_descriptor_set_get_slot(descriptor_set, binding, &info))
      return false;

   struct vtvk_rt_descriptor_source desc = {};
   desc.type = info.type;
   desc.set = set;
   desc.binding = binding;

   switch (info.type) {
   case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
      desc.acceleration_structure = info.acceleration_structure;
      break;
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      desc.image = info.image;
      desc.sampler = info.image_info.sampler;
      break;
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      desc.buffer = info.buffer.buffer;
      desc.offset = info.buffer.offset + dynamic_offset;
      if (info.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
         desc.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      if (info.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
         desc.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      desc.range = info.buffer.range;
      break;
   default:
      mesa_loge("ventus-vk: unsupported RT descriptor set=0 binding=%u type=%u",
                binding, (unsigned)info.type);
      return false;
   }

   out->push_back(desc);
   return true;
}

VkResult
vtvk_execute_trace_rays(const struct vtvk_recorded_trace_rays *record)
{
   struct vtvk_pipeline *pipeline = vtvk_pipeline_from_handle(record->pipeline);
   if (!pipeline)
      return VK_ERROR_INITIALIZATION_FAILED;
   if (!record->width || !record->height || !record->depth)
      return VK_SUCCESS;
   std::vector<vtvk_rt_descriptor_source> descriptors;
   for (uint32_t set = 0; set < pipeline->resource_layout.count; set++) {
      for (uint32_t slot = 0; slot < VTVK_MAX_DESCRIPTOR_BINDINGS; slot++)
         vtvk_collect_rt_descriptor(record->descriptor_sets[set], set, slot,
                                    record->dynamic_offsets[set][slot], &descriptors);
   }
   const VkStridedDeviceAddressRegionKHR *raygen_sbt = record->has_raygen_sbt ? &record->raygen_sbt : nullptr;
   const VkStridedDeviceAddressRegionKHR *miss_sbt = record->has_miss_sbt ? &record->miss_sbt : nullptr;
   const VkStridedDeviceAddressRegionKHR *hit_sbt = record->has_hit_sbt ? &record->hit_sbt : nullptr;
   const VkStridedDeviceAddressRegionKHR *callable_sbt = record->has_callable_sbt ? &record->callable_sbt : nullptr;
   struct vtvk_driver_trace_rays_info trace = {};
   trace.push_constants = record->push_constants;
   trace.push_constants_size = record->push_constants_size;
   trace.raygen_sbt = raygen_sbt;
   trace.miss_sbt = miss_sbt;
   trace.hit_sbt = hit_sbt;
   trace.callable_sbt = callable_sbt;
   trace.width = record->width;
   trace.height = record->height;
   trace.depth = record->depth;
   trace.per_lane_scratch_size = pipeline->per_lane_scratch_size;
   trace.private_stack_base_bytes = pipeline->private_stack_base_bytes;
   trace.hit_attribute_size_bytes = pipeline->hit_attribute_size_bytes;
   trace.continuation_frame_size = pipeline->continuation_frame_size;
   trace.max_ray_recursion_depth = pipeline->max_ray_recursion_depth;
   trace.execution_profile = pipeline->rt_execution_profile;
   trace.stage_role = pipeline->rt_stage_role;
   trace.state_descriptor_set = pipeline->rt_state_descriptor_set;
   trace.state_descriptor_binding = pipeline->rt_state_descriptor_binding;
   trace.kernel_path = pipeline->generated_kernel_path;
   trace.kernel_entry = pipeline->generated_kernel_entry;
   trace.resume_kernel_paths = pipeline->resume_kernel_paths;
   trace.resume_kernel_entries = pipeline->resume_kernel_entries;
   trace.resume_kernel_count = pipeline->resume_kernel_count;
   trace.callback_kernel_paths = pipeline->callback_kernel_paths;
   trace.callback_kernel_entries = pipeline->callback_kernel_entries;
   trace.callback_kernel_groups = pipeline->callback_kernel_groups;
   trace.callback_kernel_stages = pipeline->callback_kernel_stages;
   trace.callback_kernel_lds_sizes = pipeline->callback_kernel_lds_sizes;
   trace.callback_kernel_count = pipeline->callback_kernel_count;
   trace.descriptors = descriptors.data();
   trace.descriptor_count = descriptors.size();

   return vtvk_driver_trace_rays(&trace) ? VK_SUCCESS : VK_ERROR_DEVICE_LOST;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdTraceRaysKHR(VkCommandBuffer commandBuffer,
                     const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
                     const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
                     const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
                     const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
                     uint32_t width,
                     uint32_t height,
                     uint32_t depth)
{
   VkResult result = vtvk_cmd_buffer_record_trace_rays(
      commandBuffer, pRaygenShaderBindingTable, pMissShaderBindingTable,
      pHitShaderBindingTable, pCallableShaderBindingTable, width, height, depth);
   if (result != VK_SUCCESS)
      mesa_loge("ventus-vk: failed to record vkCmdTraceRaysKHR (%d)", result);
}
