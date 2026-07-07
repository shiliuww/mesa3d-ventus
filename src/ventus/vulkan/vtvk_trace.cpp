#include "vtvk_pipeline.h"

#include <cstdint>

#include "util/log.h"
#include "vtvk_driver_bridge.h"
#include "vtvk_entrypoints.h"

static void *
vtvk_handle_to_ptr(uint64_t handle)
{
   return (void *)(uintptr_t)handle;
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
                "scratch=%u sbt={raygen:0x%" PRIx64 ",miss:0x%" PRIx64
                ",hit:0x%" PRIx64 "}",
                width, height, depth, (uint64_t)tlas->address,
                image_extent.width, image_extent.height, image_format,
                pipeline->per_lane_scratch_size,
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
   trace.raygen_sbt = pRaygenShaderBindingTable;
   trace.miss_sbt = pMissShaderBindingTable;
   trace.hit_sbt = pHitShaderBindingTable;
   trace.callable_sbt = pCallableShaderBindingTable;
   trace.width = width;
   trace.height = height;
   trace.depth = depth;
   trace.per_lane_scratch_size = pipeline->per_lane_scratch_size;
   trace.kernel_path = pipeline->generated_kernel_path;
   trace.kernel_entry = pipeline->generated_kernel_entry;

   vtvk_driver_trace_rays(&trace);
}
