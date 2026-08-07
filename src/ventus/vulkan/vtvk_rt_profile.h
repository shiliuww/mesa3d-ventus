#ifndef VTVK_RT_PROFILE_H
#define VTVK_RT_PROFILE_H

#include <stdbool.h>
#include <stdint.h>

#include "ventus/vulkan/ventus_rt_profile.h"

#ifdef __cplusplus
extern "C" {
#endif

struct vtvk_rt_execution_profile_config {
   VkVentusRtExecutionProfile profile;
   VkVentusRtStageRole stage_role;
   uint32_t state_descriptor_set;
   uint32_t state_descriptor_binding;
   bool explicit_profile;
};

VkResult
vtvk_rt_execution_profile_parse(
   const void *p_next,
   uint32_t max_descriptor_sets,
   uint32_t max_descriptor_bindings,
   struct vtvk_rt_execution_profile_config *config);

bool
vtvk_rt_legacy_selectors_absent(
   const char *public_selector,
   const char *internal_selector,
   const char *legacy_global,
   const char *legacy_mirror);

#ifdef __cplusplus
}
#endif

#endif
