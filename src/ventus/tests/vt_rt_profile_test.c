#include <assert.h>
#include <stddef.h>

#include "ventus/vulkan/vtvk_rt_profile.h"

#define MAX_SETS 4u
#define MAX_BINDINGS 16u

static VkVentusRtExecutionProfileInfo
profile_info(VkVentusRtExecutionProfile profile,
             VkVentusRtStageRole role,
             uint32_t set,
             uint32_t binding)
{
   VkVentusRtExecutionProfileInfo info = {
      .sType = VK_STRUCTURE_TYPE_VENTUS_RT_EXECUTION_PROFILE_INFO,
      .version = VK_VENTUS_RT_EXECUTION_PROFILE_VERSION_1,
      .profile = profile,
      .stageRole = role,
      .stateDescriptorSet = set,
      .stateDescriptorBinding = binding,
   };
   return info;
}

static void
check_profile_parser(void)
{
   struct vtvk_rt_execution_profile_config config;
   assert(vtvk_rt_execution_profile_parse(
             NULL, MAX_SETS, MAX_BINDINGS, &config) == VK_SUCCESS);
   assert(config.profile == VK_VENTUS_RT_EXECUTION_PROFILE_MEGAKERNEL);
   assert(config.stage_role == VK_VENTUS_RT_STAGE_ROLE_NONE);
   assert(!config.explicit_profile);

   VkVentusRtExecutionProfileInfo mega = profile_info(
      VK_VENTUS_RT_EXECUTION_PROFILE_MEGAKERNEL,
      VK_VENTUS_RT_STAGE_ROLE_NONE, 0, 0);
   assert(vtvk_rt_execution_profile_parse(
             &mega, MAX_SETS, MAX_BINDINGS, &config) == VK_SUCCESS);
   assert(config.profile == VK_VENTUS_RT_EXECUTION_PROFILE_MEGAKERNEL);
   assert(config.explicit_profile);

   const VkVentusRtStageRole staged_roles[] = {
      VK_VENTUS_RT_STAGE_ROLE_PRIMARY,
      VK_VENTUS_RT_STAGE_ROLE_DIRECT_SHADOW,
      VK_VENTUS_RT_STAGE_ROLE_REFLECTION_0,
      VK_VENTUS_RT_STAGE_ROLE_REFLECTION_1,
   };
   for (unsigned i = 0; i < sizeof(staged_roles) / sizeof(staged_roles[0]); i++) {
      VkVentusRtExecutionProfileInfo staged = profile_info(
         VK_VENTUS_RT_EXECUTION_PROFILE_STAGED_DISPATCH,
         staged_roles[i], 2, 7);
      assert(vtvk_rt_execution_profile_parse(
                &staged, MAX_SETS, MAX_BINDINGS, &config) == VK_SUCCESS);
      assert(config.profile ==
             VK_VENTUS_RT_EXECUTION_PROFILE_STAGED_DISPATCH);
      assert(config.stage_role == staged_roles[i]);
      assert(config.state_descriptor_set == 2);
      assert(config.state_descriptor_binding == 7);
      assert(config.explicit_profile);
   }

   VkBaseInStructure unknown = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = (const VkBaseInStructure *)&mega,
   };
   assert(vtvk_rt_execution_profile_parse(
             &unknown, MAX_SETS, MAX_BINDINGS, &config) == VK_SUCCESS);

   VkVentusRtExecutionProfileInfo duplicate = mega;
   duplicate.pNext = &mega;
   assert(vtvk_rt_execution_profile_parse(
             &duplicate, MAX_SETS, MAX_BINDINGS, &config) ==
          VK_ERROR_FEATURE_NOT_PRESENT);

   VkVentusRtExecutionProfileInfo invalid = mega;
   invalid.version++;
   assert(vtvk_rt_execution_profile_parse(
             &invalid, MAX_SETS, MAX_BINDINGS, &config) ==
          VK_ERROR_FEATURE_NOT_PRESENT);

   invalid = mega;
   invalid.stageRole = VK_VENTUS_RT_STAGE_ROLE_PRIMARY;
   assert(vtvk_rt_execution_profile_parse(
             &invalid, MAX_SETS, MAX_BINDINGS, &config) ==
          VK_ERROR_FEATURE_NOT_PRESENT);

   invalid = profile_info(VK_VENTUS_RT_EXECUTION_PROFILE_STAGED_DISPATCH,
                          VK_VENTUS_RT_STAGE_ROLE_NONE, 0, 0);
   assert(vtvk_rt_execution_profile_parse(
             &invalid, MAX_SETS, MAX_BINDINGS, &config) ==
          VK_ERROR_FEATURE_NOT_PRESENT);
   invalid.stageRole = VK_VENTUS_RT_STAGE_ROLE_PRIMARY;
   invalid.stateDescriptorSet = MAX_SETS;
   assert(vtvk_rt_execution_profile_parse(
             &invalid, MAX_SETS, MAX_BINDINGS, &config) ==
          VK_ERROR_FEATURE_NOT_PRESENT);
   invalid.stateDescriptorSet = 0;
   invalid.stateDescriptorBinding = MAX_BINDINGS;
   assert(vtvk_rt_execution_profile_parse(
             &invalid, MAX_SETS, MAX_BINDINGS, &config) ==
          VK_ERROR_FEATURE_NOT_PRESENT);
   invalid.profile = (VkVentusRtExecutionProfile)99;
   assert(vtvk_rt_execution_profile_parse(
             &invalid, MAX_SETS, MAX_BINDINGS, &config) ==
          VK_ERROR_FEATURE_NOT_PRESENT);

   assert(vtvk_rt_execution_profile_parse(
             NULL, 0, MAX_BINDINGS, &config) ==
          VK_ERROR_INITIALIZATION_FAILED);
   assert(vtvk_rt_execution_profile_parse(
             NULL, MAX_SETS, 0, &config) ==
          VK_ERROR_INITIALIZATION_FAILED);
   assert(vtvk_rt_execution_profile_parse(
             NULL, MAX_SETS, MAX_BINDINGS, NULL) ==
          VK_ERROR_INITIALIZATION_FAILED);
}

static void
check_legacy_selector_rejection(void)
{
   assert(vtvk_rt_legacy_selectors_absent(NULL, NULL, NULL, NULL));
   assert(!vtvk_rt_legacy_selectors_absent("compat", NULL, NULL, NULL));
   assert(!vtvk_rt_legacy_selectors_absent(NULL, "compat", NULL, NULL));
   assert(!vtvk_rt_legacy_selectors_absent(NULL, "global", NULL, NULL));
   assert(!vtvk_rt_legacy_selectors_absent(NULL, NULL, "0", NULL));
   assert(!vtvk_rt_legacy_selectors_absent(NULL, NULL, NULL, "0"));
}

int
main(void)
{
   check_profile_parser();
   check_legacy_selector_rejection();
   return 0;
}
