/*
 * Experimental Ventus Vulkan RT execution-profile contract.
 *
 * This private pNext is consumed by the Ventus ICD at ray-tracing pipeline
 * creation. Applications that omit it use the standard MegaKernel profile.
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

typedef enum VkVentusRtExecutionProfile {
   VK_VENTUS_RT_EXECUTION_PROFILE_MEGAKERNEL = 0,
   /* Application-defined stages connected through explicit path state. */
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
