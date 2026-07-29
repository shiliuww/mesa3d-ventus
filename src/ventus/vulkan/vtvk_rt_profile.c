#include "vtvk_rt_profile.h"

#include <string.h>

static bool
vtvk_profile_value_enabled(const char *value)
{
   return value && value[0] && strcmp(value, "0") != 0 &&
          strcmp(value, "false") != 0 && strcmp(value, "FALSE") != 0 &&
          strcmp(value, "off") != 0 && strcmp(value, "OFF") != 0;
}

bool
vtvk_rt_execution_profile_resolve(const char *selector,
                                  const char *legacy_global,
                                  const char *legacy_mirror,
                                  enum vtvk_rt_execution_profile *profile)
{
   if (!profile)
      return false;

   if (selector && selector[0]) {
      if (strcmp(selector, "compat") == 0)
         *profile = VTVK_RT_EXECUTION_COMPAT;
      else if (strcmp(selector, "mirror") == 0)
         *profile = VTVK_RT_EXECUTION_WAVEFRONT_MIRROR;
      else if (strcmp(selector, "global") == 0)
         *profile = VTVK_RT_EXECUTION_WAVEFRONT_GLOBAL;
      else
         return false;
      return true;
   }

   if (legacy_global) {
      *profile = vtvk_profile_value_enabled(legacy_global) ?
         VTVK_RT_EXECUTION_WAVEFRONT_GLOBAL : VTVK_RT_EXECUTION_COMPAT;
      return true;
   }
   if (vtvk_profile_value_enabled(legacy_mirror)) {
      *profile = VTVK_RT_EXECUTION_WAVEFRONT_MIRROR;
      return true;
   }

   *profile = VTVK_RT_EXECUTION_WAVEFRONT_GLOBAL;
   return true;
}
