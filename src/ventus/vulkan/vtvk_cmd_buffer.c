#include "vtvk_private.h"

#include <string.h>

#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"

static void *
vtvk_handle_to_ptr(uint64_t handle)
{
   return (void *)(uintptr_t)handle;
}

static uint64_t
vtvk_ptr_to_handle(void *ptr)
{
   return (uint64_t)(uintptr_t)ptr;
}
static VkPipelineBindPoint
vtvk_bind_point_from_shader_stages(VkShaderStageFlags stage_flags)
{
   const VkShaderStageFlags rt_stages =
      VK_SHADER_STAGE_RAYGEN_BIT_KHR |
      VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
      VK_SHADER_STAGE_MISS_BIT_KHR |
      VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
      VK_SHADER_STAGE_CALLABLE_BIT_KHR;

   return (stage_flags & rt_stages) ?
             VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR :
             VK_PIPELINE_BIND_POINT_GRAPHICS;
}

VkPipeline
vtvk_cmd_buffer_bound_rt_pipeline(VkCommandBuffer commandBuffer)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   return cmd ? cmd->bound_rt_pipeline : VK_NULL_HANDLE;
}

VkDescriptorSet
vtvk_cmd_buffer_bound_descriptor_set(VkCommandBuffer commandBuffer,
                                     uint32_t set)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   if (!cmd || set >= VTVK_MAX_BOUND_DESCRIPTOR_SETS)
      return VK_NULL_HANDLE;
   return cmd->bound_descriptor_sets[set];
}

const void *
vtvk_cmd_buffer_push_constants(VkCommandBuffer commandBuffer,
                               uint32_t *size_out)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   if (size_out)
      *size_out = cmd ? cmd->push_constants_size : 0;
   return cmd ? cmd->push_constants : NULL;
}
static VkResult
vtvk_cmd_buffer_create(struct vk_command_pool *pool,
                       VkCommandBufferLevel level,
                       struct vk_command_buffer **cmd_buffer_out)
{
   struct vtvk_cmd_buffer *cmd =
      vk_zalloc(&pool->alloc, sizeof(*cmd), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_command_buffer_init(pool, &cmd->vk, &vtvk_command_buffer_ops, level);
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }

   *cmd_buffer_out = &cmd->vk;
   return VK_SUCCESS;
}

static void
vtvk_cmd_buffer_reset(struct vk_command_buffer *cmd_buffer,
                      VkCommandBufferResetFlags flags)
{
   (void)flags;
   vk_command_buffer_reset(cmd_buffer);
   struct vtvk_cmd_buffer *cmd =
      container_of(cmd_buffer, struct vtvk_cmd_buffer, vk);
   cmd->bound_rt_pipeline = VK_NULL_HANDLE;
   memset(cmd->bound_descriptor_sets, 0, sizeof(cmd->bound_descriptor_sets));
   memset(cmd->push_constants, 0, sizeof(cmd->push_constants));
   cmd->push_constants_size = 0;
}

static void
vtvk_cmd_buffer_destroy(struct vk_command_buffer *cmd_buffer)
{
   struct vtvk_cmd_buffer *cmd =
      container_of(cmd_buffer, struct vtvk_cmd_buffer, vk);
   struct vk_command_pool *pool = cmd_buffer->pool;

   vk_command_buffer_finish(cmd_buffer);
   vk_free(&pool->alloc, cmd);
}

const struct vk_command_buffer_ops vtvk_command_buffer_ops = {
   .create = vtvk_cmd_buffer_create,
   .reset = vtvk_cmd_buffer_reset,
   .destroy = vtvk_cmd_buffer_destroy,
};
VKAPI_ATTR VkResult VKAPI_CALL
vtvk_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                        const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(vk_command_buffer, cmd_buffer, commandBuffer);
   vk_command_buffer_begin(cmd_buffer, pBeginInfo);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(vk_command_buffer, cmd_buffer, commandBuffer);
   return vk_command_buffer_end(cmd_buffer);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdPipelineBarrier(VkCommandBuffer commandBuffer,
                        VkPipelineStageFlags srcStageMask,
                        VkPipelineStageFlags dstStageMask,
                        VkDependencyFlags dependencyFlags,
                        uint32_t memoryBarrierCount,
                        const VkMemoryBarrier *pMemoryBarriers,
                        uint32_t bufferMemoryBarrierCount,
                        const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                        uint32_t imageMemoryBarrierCount,
                        const VkImageMemoryBarrier *pImageMemoryBarriers)
{
   (void)commandBuffer;
   (void)srcStageMask;
   (void)dstStageMask;
   (void)dependencyFlags;
   (void)memoryBarrierCount;
   (void)pMemoryBarriers;
   (void)bufferMemoryBarrierCount;
   (void)pBufferMemoryBarriers;
   (void)imageMemoryBarrierCount;
   (void)pImageMemoryBarriers;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                         const VkDependencyInfo *pDependencyInfo)
{
   (void)commandBuffer;
   (void)pDependencyInfo;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBindPipeline(VkCommandBuffer commandBuffer,
                     VkPipelineBindPoint pipelineBindPoint,
                     VkPipeline pipeline)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   if (cmd && pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
      cmd->bound_rt_pipeline = pipeline;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBindDescriptorSets(VkCommandBuffer commandBuffer,
                           VkPipelineBindPoint pipelineBindPoint,
                           VkPipelineLayout layout,
                           uint32_t firstSet,
                           uint32_t descriptorSetCount,
                           const VkDescriptorSet *pDescriptorSets,
                           uint32_t dynamicOffsetCount,
                           const uint32_t *pDynamicOffsets)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   (void)layout;
   (void)dynamicOffsetCount;
   (void)pDynamicOffsets;
   if (!cmd || pipelineBindPoint != VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
      return;

   for (uint32_t i = 0; i < descriptorSetCount; i++) {
      uint32_t set = firstSet + i;
      if (set < VTVK_MAX_BOUND_DESCRIPTOR_SETS)
         cmd->bound_descriptor_sets[set] = pDescriptorSets[i];
   }
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBindDescriptorSets2(VkCommandBuffer commandBuffer,
                            const VkBindDescriptorSetsInfo *pBindDescriptorSetsInfo)
{
   vtvk_CmdBindDescriptorSets(commandBuffer,
                              vtvk_bind_point_from_shader_stages(
                                 pBindDescriptorSetsInfo->stageFlags),
                              pBindDescriptorSetsInfo->layout,
                              pBindDescriptorSetsInfo->firstSet,
                              pBindDescriptorSetsInfo->descriptorSetCount,
                              pBindDescriptorSetsInfo->pDescriptorSets,
                              pBindDescriptorSetsInfo->dynamicOffsetCount,
                              pBindDescriptorSetsInfo->pDynamicOffsets);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdPushConstants(VkCommandBuffer commandBuffer,
                      VkPipelineLayout layout,
                      VkShaderStageFlags stageFlags,
                      uint32_t offset,
                      uint32_t size,
                      const void *pValues)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   (void)layout;
   if (!cmd || !(stageFlags & (VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                               VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                               VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                               VK_SHADER_STAGE_MISS_BIT_KHR |
                               VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
                               VK_SHADER_STAGE_CALLABLE_BIT_KHR)) ||
       !pValues || offset > VTVK_MAX_PUSH_CONSTANT_SIZE ||
       size > VTVK_MAX_PUSH_CONSTANT_SIZE - offset)
      return;

   memcpy(cmd->push_constants + offset, pValues, size);
   cmd->push_constants_size = MAX2(cmd->push_constants_size, offset + size);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdPushConstants2(VkCommandBuffer commandBuffer,
                       const VkPushConstantsInfo *pPushConstantsInfo)
{
   if (!pPushConstantsInfo)
      return;
   vtvk_CmdPushConstants(commandBuffer, pPushConstantsInfo->layout,
                         pPushConstantsInfo->stageFlags,
                         pPushConstantsInfo->offset,
                         pPushConstantsInfo->size,
                         pPushConstantsInfo->pValues);
}
