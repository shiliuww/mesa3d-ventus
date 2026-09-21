#include "vtvk_private.h"

#include <stdlib.h>
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

static void
vtvk_cmd_buffer_discard_commands(struct vtvk_cmd_buffer *cmd)
{
   for (uint32_t i = 0; i < cmd->recorded_command_count; i++) {
      struct vtvk_recorded_command *record = &cmd->recorded_commands[i];
      if (record->type == VTVK_RECORDED_COMMAND_BUILD_AS) {
         free(record->u.build_as.infos);
         free(record->u.build_as.geometries);
         free(record->u.build_as.ranges);
      }
   }
   free(cmd->recorded_commands);
   cmd->recorded_commands = NULL;
   cmd->recorded_command_count = 0;
   cmd->recorded_command_capacity = 0;
}

static VkResult
vtvk_cmd_buffer_append_command(struct vtvk_cmd_buffer *cmd,
                               const struct vtvk_recorded_command *record)
{
   if (cmd->recorded_command_count == cmd->recorded_command_capacity) {
      uint32_t capacity = cmd->recorded_command_capacity ?
         cmd->recorded_command_capacity * 2 : 8;
      struct vtvk_recorded_command *commands = realloc(
         cmd->recorded_commands, sizeof(*commands) * capacity);
      if (!commands)
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      cmd->recorded_commands = commands;
      cmd->recorded_command_capacity = capacity;
   }
   cmd->recorded_commands[cmd->recorded_command_count++] = *record;
   return VK_SUCCESS;
}
void
vtvk_cmd_buffer_record_copy(VkCommandBuffer commandBuffer, void *dst,
                            const void *src, size_t size)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   struct vtvk_recorded_command record = {
      .type = VTVK_RECORDED_COMMAND_COPY,
      .u.copy = { .dst = dst, .src = src, .size = size },
   };
   VkResult result = vtvk_cmd_buffer_append_command(cmd, &record);
   if (result != VK_SUCCESS)
      vk_command_buffer_set_error(&cmd->vk, result);
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

VkResult
vtvk_cmd_buffer_record_as_build(
   VkCommandBuffer commandBuffer,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   if (!cmd || !infoCount || !pInfos || !ppBuildRangeInfos)
      return VK_ERROR_INITIALIZATION_FAILED;

   struct vtvk_recorded_command record = {
      .type = VTVK_RECORDED_COMMAND_BUILD_AS,
   };
   struct vtvk_recorded_as_build *build = &record.u.build_as;
   build->info_count = infoCount;
   uint32_t geometry_count = 0;
   for (uint32_t i = 0; i < infoCount; i++)
      geometry_count += pInfos[i].geometryCount;
   build->infos = calloc(infoCount, sizeof(*build->infos));
   build->geometries = calloc(geometry_count, sizeof(*build->geometries));
   build->ranges = calloc(geometry_count, sizeof(*build->ranges));
   if (!build->infos || !build->geometries || !build->ranges)
      goto oom;

   uint32_t offset = 0;
   for (uint32_t i = 0; i < infoCount; i++) {
      if (!pInfos[i].geometryCount || !ppBuildRangeInfos[i])
         goto unsupported;
      build->infos[i] = pInfos[i];
      build->infos[i].pGeometries = &build->geometries[offset];
      build->infos[i].ppGeometries = NULL;
      for (uint32_t g = 0; g < pInfos[i].geometryCount; g++, offset++) {
         const VkAccelerationStructureGeometryKHR *geometry =
            pInfos[i].pGeometries ? &pInfos[i].pGeometries[g] :
            (pInfos[i].ppGeometries ? pInfos[i].ppGeometries[g] : NULL);
         if (!geometry)
            goto unsupported;
         build->geometries[offset] = *geometry;
         build->ranges[offset] = ppBuildRangeInfos[i][g];
      }
   }

   VkResult result = vtvk_cmd_buffer_append_command(cmd, &record);
   if (result == VK_SUCCESS)
      return VK_SUCCESS;
oom:
   free(build->infos);
   free(build->geometries);
   free(build->ranges);
   return VK_ERROR_OUT_OF_HOST_MEMORY;
unsupported:
   free(build->infos);
   free(build->geometries);
   free(build->ranges);
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VkResult
vtvk_cmd_buffer_record_trace_rays(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   uint32_t width, uint32_t height, uint32_t depth)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   if (!cmd)
      return VK_ERROR_INITIALIZATION_FAILED;

   struct vtvk_recorded_command record = {
      .type = VTVK_RECORDED_COMMAND_TRACE_RAYS,
      .u.trace_rays = {
         .pipeline = cmd->bound_rt_pipeline,
         .descriptor_set = cmd->bound_descriptor_sets[0],
         .push_constants_size = cmd->push_constants_size,
         .width = width,
         .height = height,
         .depth = depth,
      },
   };
   struct vtvk_recorded_trace_rays *trace = &record.u.trace_rays;
   memcpy(trace->descriptor_sets, cmd->bound_descriptor_sets, sizeof(trace->descriptor_sets));
   memcpy(trace->dynamic_offsets, cmd->dynamic_offsets, sizeof(trace->dynamic_offsets));
   memcpy(trace->push_constants, cmd->push_constants,
          sizeof(trace->push_constants));
   if (pRaygenShaderBindingTable) {
      trace->raygen_sbt = *pRaygenShaderBindingTable;
      trace->has_raygen_sbt = true;
   }
   if (pMissShaderBindingTable) {
      trace->miss_sbt = *pMissShaderBindingTable;
      trace->has_miss_sbt = true;
   }
   if (pHitShaderBindingTable) {
      trace->hit_sbt = *pHitShaderBindingTable;
      trace->has_hit_sbt = true;
   }
   if (pCallableShaderBindingTable) {
      trace->callable_sbt = *pCallableShaderBindingTable;
      trace->has_callable_sbt = true;
   }
   return vtvk_cmd_buffer_append_command(cmd, &record);
}

VkResult
vtvk_cmd_buffer_execute(VkCommandBuffer commandBuffer)
{
   struct vtvk_cmd_buffer *cmd = vtvk_handle_to_ptr((uint64_t)commandBuffer);
   if (!cmd)
      return VK_ERROR_INITIALIZATION_FAILED;
   struct vtvk_device *device =
      container_of(cmd->vk.base.device, struct vtvk_device, vk);

   for (uint32_t i = 0; i < cmd->recorded_command_count; i++) {
      struct vtvk_recorded_command *record = &cmd->recorded_commands[i];
      if (record->type == VTVK_RECORDED_COMMAND_BUILD_AS) {
         struct vtvk_recorded_as_build *build = &record->u.build_as;
         const VkAccelerationStructureBuildRangeInfoKHR *ranges[build->info_count];
         uint32_t offset = 0;
         for (uint32_t j = 0; j < build->info_count; j++) {
            ranges[j] = &build->ranges[offset];
            offset += build->infos[j].geometryCount;
         }
         VkResult result = vtvk_execute_build_acceleration_structures(
            device, build->info_count, build->infos, ranges);
         if (result != VK_SUCCESS)
            return result;
      } else if (record->type == VTVK_RECORDED_COMMAND_COPY) {
         memcpy(record->u.copy.dst, record->u.copy.src, record->u.copy.size);
      } else if (record->type == VTVK_RECORDED_COMMAND_TRACE_RAYS) {
         VkResult result = vtvk_execute_trace_rays(&record->u.trace_rays);
         if (result != VK_SUCCESS)
            return result;
      }
   }
   return VK_SUCCESS;
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
   vtvk_cmd_buffer_discard_commands(cmd);
   cmd->bound_rt_pipeline = VK_NULL_HANDLE;
   memset(cmd->bound_descriptor_sets, 0, sizeof(cmd->bound_descriptor_sets));
   memset(cmd->dynamic_offsets, 0, sizeof(cmd->dynamic_offsets));
   memset(cmd->push_constants, 0, sizeof(cmd->push_constants));
   cmd->push_constants_size = 0;
}

static void
vtvk_cmd_buffer_destroy(struct vk_command_buffer *cmd_buffer)
{
   struct vtvk_cmd_buffer *cmd =
      container_of(cmd_buffer, struct vtvk_cmd_buffer, vk);
   struct vk_command_pool *pool = cmd_buffer->pool;

   vtvk_cmd_buffer_discard_commands(cmd);
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
   vtvk_cmd_buffer_discard_commands(
      container_of(cmd_buffer, struct vtvk_cmd_buffer, vk));
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
   uint32_t offset_index = 0;
   if (!cmd || pipelineBindPoint != VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
      return;

   for (uint32_t i = 0; i < descriptorSetCount; i++) {
      uint32_t set = firstSet + i;
      if (set < VTVK_MAX_BOUND_DESCRIPTOR_SETS) {
         cmd->bound_descriptor_sets[set] = pDescriptorSets[i];
         memset(cmd->dynamic_offsets[set], 0, sizeof(cmd->dynamic_offsets[set]));
         struct vtvk_descriptor_layout set_layout;
         if (!vtvk_descriptor_set_layout(pDescriptorSets[i], &set_layout))
            continue;
         for (uint32_t b = 0; b < set_layout.count; b++) {
            VkDescriptorType type = set_layout.bindings[b].descriptorType;
            if (type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC &&
                type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
               continue;
            for (uint32_t e = 0; e < set_layout.bindings[b].descriptorCount; e++)
               if (offset_index < dynamicOffsetCount)
                  cmd->dynamic_offsets[set][set_layout.slots[b] + e] = pDynamicOffsets[offset_index++];
         }
      }
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
