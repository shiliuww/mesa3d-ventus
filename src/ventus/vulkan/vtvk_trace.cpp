#include "vtvk_pipeline.h"

#include <cstdint>
#include <vector>

#include "util/log.h"
#include "vtvk_driver_bridge.h"
#include "vtvk_entrypoints.h"

static void *
vtvk_handle_to_ptr(uint64_t handle)
{
   return (void *)(uintptr_t)handle;
}

static bool
vtvk_collect_rt_descriptor(VkDescriptorSet descriptor_set, uint32_t binding,
                           std::vector<vtvk_rt_descriptor_source> *out)
{
   struct vtvk_descriptor_binding_info info = {};
   if (!vtvk_descriptor_set_get_binding(descriptor_set, binding, &info))
      return false;

   struct vtvk_rt_descriptor_source desc = {};
   desc.type = info.type;
   desc.set = 0;
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
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      desc.buffer = info.buffer.buffer;
      desc.offset = info.buffer.offset;
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
   static bool logged;

   VkPipeline pipeline_handle =
      vtvk_cmd_buffer_bound_rt_pipeline(commandBuffer);
   struct vtvk_pipeline *pipeline =
      vtvk_pipeline_from_handle(pipeline_handle);
   VkDescriptorSet descriptor_set =
      vtvk_cmd_buffer_bound_descriptor_set(commandBuffer, 0);
   VkAccelerationStructureKHR tlas_handle =
      vtvk_descriptor_set_acceleration_structure(descriptor_set, 0);
   struct vtvk_acceleration_structure *tlas =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)tlas_handle);

   VkExtent3D image_extent = {};
   VkFormat image_format = VK_FORMAT_UNDEFINED;
   VkImage storage_image = vtvk_descriptor_set_image(descriptor_set, 1);
   void *storage_image_data =
      vtvk_image_host_ptr(storage_image, &image_extent, &image_format);

   VkDeviceSize camera_range = 0;
   void *camera_data =
      vtvk_descriptor_set_buffer_ptr(descriptor_set, 2, &camera_range);
   uint32_t push_constants_size = 0;
   const void *push_constants =
      vtvk_cmd_buffer_push_constants(commandBuffer, &push_constants_size);

   std::vector<vtvk_rt_descriptor_source> descriptors;
   descriptors.reserve(5);
   for (uint32_t binding = 0; binding <= 4; binding++)
      vtvk_collect_rt_descriptor(descriptor_set, binding, &descriptors);

   if (!pipeline || !tlas || !tlas->address || !storage_image_data ||
       !camera_data) {
      mesa_loge("ventus-vk: vkCmdTraceRaysKHR missing Spike dispatch state "
                "pipeline=%p tlas=0x%" PRIx64 " image=%p camera=%p",
                (void *)pipeline, tlas ? (uint64_t)tlas->address : 0,
                storage_image_data, camera_data);
      return;
   }

   if (!logged) {
      logged = true;
      mesa_logi("ventus-vk: vkCmdTraceRaysKHR collected Spike dispatch state "
                "%ux%ux%u tlas=0x%" PRIx64 " image=%ux%u format=%u "
                "descriptors=%zu scratch=%u sbt={raygen:0x%" PRIx64 ",miss:0x%" PRIx64
                ",hit:0x%" PRIx64 "}",
                width, height, depth, (uint64_t)tlas->address,
                image_extent.width, image_extent.height, image_format,
                descriptors.size(), pipeline->per_lane_scratch_size,
                pRaygenShaderBindingTable ?
                   (uint64_t)pRaygenShaderBindingTable->deviceAddress : 0,
                pMissShaderBindingTable ?
                   (uint64_t)pMissShaderBindingTable->deviceAddress : 0,
                pHitShaderBindingTable ?
                   (uint64_t)pHitShaderBindingTable->deviceAddress : 0);
   }

   struct vtvk_driver_trace_rays_info trace = {};
   trace.tlas = tlas;
   trace.storage_image_data = storage_image_data;
   trace.image_extent = image_extent;
   trace.image_format = image_format;
   trace.camera_data = camera_data;
   trace.camera_range = camera_range;
   trace.push_constants = push_constants;
   trace.push_constants_size = push_constants_size;
   trace.raygen_sbt = pRaygenShaderBindingTable;
   trace.miss_sbt = pMissShaderBindingTable;
   trace.hit_sbt = pHitShaderBindingTable;
   trace.callable_sbt = pCallableShaderBindingTable;
   trace.width = width;
   trace.height = height;
   trace.depth = depth;
   trace.per_lane_scratch_size = pipeline->per_lane_scratch_size;
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

   vtvk_driver_trace_rays(&trace);
}
