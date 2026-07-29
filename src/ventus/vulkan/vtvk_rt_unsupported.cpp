#include "vtvk_private.h"

#include "vk_common_entrypoints.h"
#include "vtvk_entrypoints.h"

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CopyAccelerationStructureKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   (void)device;
   (void)deferredOperation;
   (void)pInfo;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CopyAccelerationStructureToMemoryKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   (void)device;
   (void)deferredOperation;
   (void)pInfo;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CopyMemoryToAccelerationStructureKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   (void)device;
   (void)deferredOperation;
   (void)pInfo;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_WriteAccelerationStructuresPropertiesKHR(
   VkDevice device,
   uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType,
   size_t dataSize,
   void *pData,
   size_t stride)
{
   (void)device;
   (void)accelerationStructureCount;
   (void)pAccelerationStructures;
   (void)queryType;
   (void)dataSize;
   (void)pData;
   (void)stride;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdCopyAccelerationStructureToMemoryKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   (void)commandBuffer;
   (void)pInfo;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdCopyMemoryToAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   (void)commandBuffer;
   (void)pInfo;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdWriteAccelerationStructuresPropertiesKHR(
   VkCommandBuffer commandBuffer,
   uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType,
   VkQueryPool queryPool,
   uint32_t firstQuery)
{
   (void)commandBuffer;
   (void)accelerationStructureCount;
   (void)pAccelerationStructures;
   (void)queryType;
   (void)queryPool;
   (void)firstQuery;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice device,
   const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   (void)device;
   (void)pVersionInfo;
   *pCompatibility = VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetRayTracingCaptureReplayShaderGroupHandlesKHR(
   VkDevice device,
   VkPipeline pipeline,
   uint32_t firstGroup,
   uint32_t groupCount,
   size_t dataSize,
   void *pData)
{
   return vtvk_GetRayTracingShaderGroupHandlesKHR(device, pipeline, firstGroup,
                                                  groupCount, dataSize, pData);
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
   (void)commandBuffer;
   (void)pRaygenShaderBindingTable;
   (void)pMissShaderBindingTable;
   (void)pHitShaderBindingTable;
   (void)pCallableShaderBindingTable;
   (void)indirectDeviceAddress;
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL
vtvk_GetRayTracingShaderGroupStackSizeKHR(
   VkDevice device,
   VkPipeline pipeline,
   uint32_t group,
   VkShaderGroupShaderKHR groupShader)
{
   (void)device;
   (void)pipeline;
   (void)group;
   (void)groupShader;
   return 0;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdSetRayTracingPipelineStackSizeKHR(VkCommandBuffer commandBuffer,
                                          uint32_t pipelineStackSize)
{
   (void)commandBuffer;
   (void)pipelineStackSize;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateDeferredOperationKHR(VkDevice device,
                                const VkAllocationCallbacks *pAllocator,
                                VkDeferredOperationKHR *pDeferredOperation)
{
   return vk_common_CreateDeferredOperationKHR(device, pAllocator,
                                               pDeferredOperation);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyDeferredOperationKHR(VkDevice device,
                                 VkDeferredOperationKHR operation,
                                 const VkAllocationCallbacks *pAllocator)
{
   vk_common_DestroyDeferredOperationKHR(device, operation, pAllocator);
}

VKAPI_ATTR uint32_t VKAPI_CALL
vtvk_GetDeferredOperationMaxConcurrencyKHR(VkDevice device,
                                           VkDeferredOperationKHR operation)
{
   return vk_common_GetDeferredOperationMaxConcurrencyKHR(device, operation);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetDeferredOperationResultKHR(VkDevice device,
                                   VkDeferredOperationKHR operation)
{
   return vk_common_GetDeferredOperationResultKHR(device, operation);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_DeferredOperationJoinKHR(VkDevice device,
                              VkDeferredOperationKHR operation)
{
   return vk_common_DeferredOperationJoinKHR(device, operation);
}
