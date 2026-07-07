#include "vtvk_pipeline.h"

#include "vk_pipeline.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateGraphicsPipelines(VkDevice _device,
                             VkPipelineCache pipelineCache,
                             uint32_t createInfoCount,
                             const VkGraphicsPipelineCreateInfo *pCreateInfos,
                             const VkAllocationCallbacks *pAllocator,
                             VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   (void)pipelineCache;

   VkResult final_result = VK_SUCCESS;
   for (uint32_t i = 0; i < createInfoCount; i++) {
      VkPipelineCreateFlags2KHR flags =
         vk_graphics_pipeline_create_flags(&pCreateInfos[i]);
      struct vtvk_pipeline *pipeline =
         (struct vtvk_pipeline *)vk_pipeline_zalloc(
            &device->vk, &vtvk_pipeline_ops, VK_PIPELINE_BIND_POINT_GRAPHICS,
            flags, pAllocator, sizeof(*pipeline));

      if (!pipeline) {
         pPipelines[i] = VK_NULL_HANDLE;
         final_result = vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

         if (pCreateInfos[i].flags &
             VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT) {
            for (uint32_t j = i + 1; j < createInfoCount; j++)
               pPipelines[j] = VK_NULL_HANDLE;
            break;
         }
         continue;
      }

      for (uint32_t s = 0; s < pCreateInfos[i].stageCount; s++)
         pipeline->vk.stages |= pCreateInfos[i].pStages[s].stage;

      pPipelines[i] = vk_pipeline_to_handle(&pipeline->vk);
   }

   return final_result;
}
