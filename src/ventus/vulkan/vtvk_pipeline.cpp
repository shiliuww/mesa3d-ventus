#include "vtvk_pipeline.h"

#include <cstdlib>

#include "util/ralloc.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"

struct vtvk_pipeline *
vtvk_pipeline_from_handle(VkPipeline pipeline)
{
   VK_FROM_HANDLE(vk_pipeline, vk_pipeline, pipeline);
   return vk_pipeline ? container_of(vk_pipeline, struct vtvk_pipeline, vk) : nullptr;
}

void
vtvk_pipeline_destroy(struct vk_device *device,
                      struct vk_pipeline *vk_pipeline,
                      const VkAllocationCallbacks *pAllocator)
{
   struct vtvk_pipeline *pipeline =
      container_of(vk_pipeline, struct vtvk_pipeline, vk);

   delete pipeline->llvm_module;
   free(pipeline->generated_kernel_path);
   for (uint32_t i = 0; i < pipeline->resume_kernel_count; i++)
      free(pipeline->resume_kernel_paths[i]);
   free(pipeline->resume_kernel_paths);
   free(pipeline->resume_kernel_entries);
   for (uint32_t i = 0; i < pipeline->callback_kernel_count; i++)
      free(pipeline->callback_kernel_paths[i]);
   free(pipeline->callback_kernel_paths);
   free(pipeline->callback_kernel_entries);
   free(pipeline->callback_kernel_groups);
   free(pipeline->callback_kernel_stages);
   ralloc_free(pipeline->nir_mem_ctx);
   vk_free2(&device->alloc, pAllocator, pipeline->shaders);
   vk_free2(&device->alloc, pAllocator, pipeline->group_handles);
   vk_pipeline_free(device, pAllocator, vk_pipeline);
}

const struct vk_pipeline_ops vtvk_pipeline_ops = {
   .destroy = vtvk_pipeline_destroy,
};

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyPipeline(VkDevice _device,
                     VkPipeline _pipeline,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   VK_FROM_HANDLE(vk_pipeline, pipeline, _pipeline);

   if (!pipeline)
      return;

   pipeline->ops->destroy(&device->vk, pipeline, pAllocator);
}
