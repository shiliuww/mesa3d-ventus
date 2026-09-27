/*
 * Experimental Ventus Vulkan RT execution-profile contract.
 *
 * This is intentionally a private pNext structure: it is consumed by the
 * Ventus ICD at VkRayTracingPipelineCreateInfoKHR creation time and is not a
 * Vulkan-standard extension.  Applications that omit it retain the portable
 * megakernel execution profile.
 */
#ifndef VENTUS_VULKAN_RT_PROFILE_H
#define VENTUS_VULKAN_RT_PROFILE_H

#include <vulkan/vulkan_core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VK_STRUCTURE_TYPE_VENTUS_RT_EXECUTION_PROFILE_INFO \
   ((VkStructureType)1000900000)

#define VK_VENTUS_RT_EXECUTION_PROFILE_VERSION_1 1u

/* Shader-private flag consumed by Mesa before RT Local TraceMeta0 is written.
 * A raygen helper may OR it into one traceRayEXT invocation to request the
 * collective group-traverse lowering.  It is not a Vulkan RayFlagsKHR bit and
 * must never be visible to an RT shader-table implementation. */
#define VENTUS_RT_TRACE_FLAG_GROUP_COLLECTIVE_BIT 0x80000000u

typedef enum VkVentusRtExecutionProfile {
   VK_VENTUS_RT_EXECUTION_PROFILE_MEGAKERNEL = 0,
   /* Application-defined primary/shadow/reflection dispatch stages.  This is
    * not a Vulkan-standard extension and does not imply wavefront execution. */
   VK_VENTUS_RT_EXECUTION_PROFILE_STAGED_DISPATCH = 1,
} VkVentusRtExecutionProfile;

typedef enum VkVentusRtStageRole {
   VK_VENTUS_RT_STAGE_ROLE_NONE = 0,
   VK_VENTUS_RT_STAGE_ROLE_PRIMARY = 1,
   VK_VENTUS_RT_STAGE_ROLE_DIRECT_SHADOW = 2,
   VK_VENTUS_RT_STAGE_ROLE_REFLECTION_0 = 3,
   VK_VENTUS_RT_STAGE_ROLE_REFLECTION_1 = 4,
} VkVentusRtStageRole;

typedef struct VkVentusRtExecutionProfileInfo {
   VkStructureType sType;
   const void *pNext;
   uint32_t version;
   VkVentusRtExecutionProfile profile;
   VkVentusRtStageRole stageRole;
   uint32_t stateDescriptorSet;
   uint32_t stateDescriptorBinding;
} VkVentusRtExecutionProfileInfo;

#ifdef __cplusplus
}
#endif

#endif
