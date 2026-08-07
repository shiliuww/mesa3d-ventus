#include "vtvk_rt_profile.h"

static bool
vtvk_rt_stage_role_is_staged(VkVentusRtStageRole role)
{
   switch (role) {
   case VK_VENTUS_RT_STAGE_ROLE_PRIMARY:
   case VK_VENTUS_RT_STAGE_ROLE_DIRECT_SHADOW:
   case VK_VENTUS_RT_STAGE_ROLE_REFLECTION_0:
   case VK_VENTUS_RT_STAGE_ROLE_REFLECTION_1:
      return true;
   default:
      return false;
   }
}

VkResult
vtvk_rt_execution_profile_parse(
   const void *p_next,
   uint32_t max_descriptor_sets,
   uint32_t max_descriptor_bindings,
   struct vtvk_rt_execution_profile_config *config)
{
   if (!config || !max_descriptor_sets || !max_descriptor_bindings)
      return VK_ERROR_INITIALIZATION_FAILED;

   *config = (struct vtvk_rt_execution_profile_config) {
      .profile = VK_VENTUS_RT_EXECUTION_PROFILE_MEGAKERNEL,
      .stage_role = VK_VENTUS_RT_STAGE_ROLE_NONE,
   };

   const VkVentusRtExecutionProfileInfo *profile_info = NULL;
   for (const VkBaseInStructure *node =
           (const VkBaseInStructure *)p_next;
        node; node = node->pNext) {
      if (node->sType != VK_STRUCTURE_TYPE_VENTUS_RT_EXECUTION_PROFILE_INFO)
         continue;
      if (profile_info)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      profile_info = (const VkVentusRtExecutionProfileInfo *)node;
   }

   if (!profile_info)
      return VK_SUCCESS;
   if (profile_info->version != VK_VENTUS_RT_EXECUTION_PROFILE_VERSION_1)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   switch (profile_info->profile) {
   case VK_VENTUS_RT_EXECUTION_PROFILE_MEGAKERNEL:
      if (profile_info->stageRole != VK_VENTUS_RT_STAGE_ROLE_NONE ||
          profile_info->stateDescriptorSet != 0 ||
          profile_info->stateDescriptorBinding != 0)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      break;
   case VK_VENTUS_RT_EXECUTION_PROFILE_STAGED_DISPATCH:
      if (!vtvk_rt_stage_role_is_staged(profile_info->stageRole) ||
          profile_info->stateDescriptorSet >= max_descriptor_sets ||
          profile_info->stateDescriptorBinding >= max_descriptor_bindings)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      break;
   default:
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }

   config->profile = profile_info->profile;
   config->stage_role = profile_info->stageRole;
   config->state_descriptor_set = profile_info->stateDescriptorSet;
   config->state_descriptor_binding = profile_info->stateDescriptorBinding;
   config->explicit_profile = true;
   return VK_SUCCESS;
}

bool
vtvk_rt_legacy_selectors_absent(
   const char *public_selector,
   const char *internal_selector,
   const char *legacy_global,
   const char *legacy_mirror)
{
   return !(public_selector && public_selector[0]) &&
          !(internal_selector && internal_selector[0]) &&
          !legacy_global && !legacy_mirror;
}
