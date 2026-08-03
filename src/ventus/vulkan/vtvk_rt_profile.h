#ifndef VTVK_RT_PROFILE_H
#define VTVK_RT_PROFILE_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

enum vtvk_rt_execution_profile {
   VTVK_RT_EXECUTION_COMPAT = 0,
   VTVK_RT_EXECUTION_WAVEFRONT_GLOBAL,
};

bool
vtvk_rt_execution_profile_resolve(const char *selector,
                                  const char *legacy_global,
                                  const char *legacy_mirror,
                                  enum vtvk_rt_execution_profile *profile);

#ifdef __cplusplus
}
#endif

#endif
