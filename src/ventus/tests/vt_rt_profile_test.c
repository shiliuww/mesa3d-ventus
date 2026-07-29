#include <assert.h>
#include <stddef.h>

#include "ventus/vulkan/vtvk_rt_profile.h"

static void
check_profile(const char *selector, const char *legacy_global,
              const char *legacy_mirror,
              enum vtvk_rt_execution_profile expected)
{
   enum vtvk_rt_execution_profile profile = VTVK_RT_EXECUTION_COMPAT;
   assert(vtvk_rt_execution_profile_resolve(
      selector, legacy_global, legacy_mirror, &profile));
   assert(profile == expected);
}

int
main(void)
{
   check_profile(NULL, NULL, NULL, VTVK_RT_EXECUTION_WAVEFRONT_GLOBAL);
   check_profile("", NULL, NULL, VTVK_RT_EXECUTION_WAVEFRONT_GLOBAL);
   check_profile("global", "0", "1", VTVK_RT_EXECUTION_WAVEFRONT_GLOBAL);
   check_profile("compat", "1", "1", VTVK_RT_EXECUTION_COMPAT);
   check_profile("mirror", "1", NULL, VTVK_RT_EXECUTION_WAVEFRONT_MIRROR);

   check_profile(NULL, "1", "1", VTVK_RT_EXECUTION_WAVEFRONT_GLOBAL);
   check_profile(NULL, "true", NULL, VTVK_RT_EXECUTION_WAVEFRONT_GLOBAL);
   check_profile(NULL, "0", "1", VTVK_RT_EXECUTION_COMPAT);
   check_profile(NULL, "false", "1", VTVK_RT_EXECUTION_COMPAT);
   check_profile(NULL, NULL, "1", VTVK_RT_EXECUTION_WAVEFRONT_MIRROR);
   check_profile(NULL, NULL, "off", VTVK_RT_EXECUTION_WAVEFRONT_GLOBAL);

   enum vtvk_rt_execution_profile profile = VTVK_RT_EXECUTION_COMPAT;
   assert(!vtvk_rt_execution_profile_resolve("invalid", NULL, NULL, &profile));
   assert(!vtvk_rt_execution_profile_resolve(NULL, NULL, NULL, NULL));
   return 0;
}
