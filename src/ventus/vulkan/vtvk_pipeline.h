#ifndef VTVK_PIPELINE_H
#define VTVK_PIPELINE_H

#include "vtvk_private.h"
#include <ventus/vulkan/ventus_rt_profile.h>

#include "vk_pipeline.h"
#include "ventus/compiler/vt_nir.h"

#ifdef __cplusplus
/* X11 platform headers define common tokens such as None/Always as macros.
 * LLVM uses the same names in scoped enums, so clear them before LLVM headers.
 */
#ifdef None
#undef None
#endif
#ifdef Always
#undef Always
#endif
#ifdef Status
#undef Status
#endif
#ifdef True
#undef True
#endif
#ifdef False
#undef False
#endif

#include "llvm/IR/Module.h"

struct vtvk_pipeline {
   struct vk_pipeline vk;

   void *nir_mem_ctx;
   nir_shader **shaders;
   uint32_t shader_count;
   /* NIR CPS split products for the PDS-free root wavefront profile. */
   nir_shader **resume_shaders;
   uint32_t resume_shader_count;
   /* Independently linked executable images for the CPS split products.
    * They are metadata only until the RTcore completion dispatcher schedules
    * them; raygen remains generated_kernel_path. */
   char **resume_kernel_paths;
   uint64_t *resume_kernel_entries;
   uint32_t resume_kernel_count;
   /* Terminal global callback kernels; index is selected by SBT group metadata
    * after RTcore has published a completion record. */
   char **callback_kernel_paths;
   uint64_t *callback_kernel_entries;
   uint32_t *callback_kernel_groups;
   uint32_t *callback_kernel_stages;
   /* Per-callback workgroup LDS reservation.  This is phase-local scratch,
    * never a replacement for the global completion/CPS ABI. */
   uint32_t *callback_kernel_lds_sizes;
   uint32_t callback_kernel_count;

   uint8_t *group_handles;
   uint32_t group_count;

   uint32_t per_lane_scratch_size;
   uint32_t continuation_frame_size;
   uint32_t max_ray_recursion_depth;
   VkVentusRtExecutionProfile rt_execution_profile;
   VkVentusRtStageRole rt_stage_role;
   uint32_t rt_state_descriptor_set;
   uint32_t rt_state_descriptor_binding;
   bool rt_uses_rv64;
   struct vt_nir_pipeline_scratch_info scratch_info;

   llvm::Module *llvm_module;
   char *generated_kernel_path;
   uint64_t generated_kernel_entry;
};

extern const struct vk_pipeline_ops vtvk_pipeline_ops;
struct vtvk_pipeline *vtvk_pipeline_from_handle(VkPipeline pipeline);
void vtvk_pipeline_destroy(struct vk_device *device,
                           struct vk_pipeline *vk_pipeline,
                           const VkAllocationCallbacks *pAllocator);
#endif

#endif
