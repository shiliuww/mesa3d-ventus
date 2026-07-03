#include "vtvk_private.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "compiler/spirv/nir_spirv.h"
#include "nir/nir.h"
#include "util/log.h"
#include "util/ralloc.h"
#include "vk_pipeline.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"
#include "ventus/common/ventus_rt_bvh.h"
#include "ventus/compiler/vt_nir.h"
#include "ventus/llvm/ventus_nir_to_llvm.h"

#include <cfloat>

/* X11 platform headers define common tokens such as None/Always as macros.
 * LLVM uses the same names in scoped enums, so clear them before LLVM headers.
 */
#ifdef None
#undef None
#endif
#ifdef Always
#undef Always
#endif

#include "llvm/IR/Module.h"

struct vtvk_aabb {
   float min[3];
   float max[3];
};

struct vtvk_build_node {
   uint32_t ref;
   struct vtvk_aabb bounds;
};

struct vtvk_pipeline {
   struct vk_pipeline vk;

   void *nir_mem_ctx;
   nir_shader **shaders;
   uint32_t shader_count;

   uint8_t *group_handles;
   uint32_t group_count;

   uint32_t per_lane_scratch_size;
   struct vt_nir_pipeline_scratch_info scratch_info;

   llvm::Module *llvm_module;
};

struct vtvk_acceleration_structure {
   VkAccelerationStructureTypeKHR type;
   VkBuffer buffer;
   VkDeviceSize offset;
   VkDeviceSize size;
   VkDeviceAddress address;
   void *data;
};

static void *
vtvk_handle_to_ptr(uint64_t handle)
{
   return (void *)(uintptr_t)handle;
}

static uint64_t
vtvk_ptr_to_handle(void *ptr)
{
   return (uint64_t)(uintptr_t)ptr;
}

static const struct nir_shader_compiler_options vtvk_nir_options = {};

static const struct spirv_to_nir_options vtvk_spirv_options = {
   .environment = NIR_SPIRV_VULKAN,
   .ubo_addr_format = nir_address_format_vec2_index_32bit_offset,
   .ssbo_addr_format = nir_address_format_vec2_index_32bit_offset,
   .phys_ssbo_addr_format = nir_address_format_64bit_global,
   .push_const_addr_format = nir_address_format_logical,
   .shared_addr_format = nir_address_format_32bit_offset,
   .constant_addr_format = nir_address_format_64bit_global,
};

static void
vtvk_aabb_empty(struct vtvk_aabb *aabb)
{
   for (unsigned i = 0; i < 3; i++) {
      aabb->min[i] = FLT_MAX;
      aabb->max[i] = -FLT_MAX;
   }
}

static void
vtvk_aabb_include_point(struct vtvk_aabb *aabb, const float p[3])
{
   for (unsigned i = 0; i < 3; i++) {
      aabb->min[i] = std::min(aabb->min[i], p[i]);
      aabb->max[i] = std::max(aabb->max[i], p[i]);
   }
}

static void
vtvk_aabb_include_aabb(struct vtvk_aabb *dst, const struct vtvk_aabb *src)
{
   vtvk_aabb_include_point(dst, src->min);
   vtvk_aabb_include_point(dst, src->max);
}

static void
vtvk_store_u16(uint8_t *base, uint32_t offset, uint16_t value)
{
   memcpy(base + offset, &value, sizeof(value));
}

static void
vtvk_store_u32(uint8_t *base, uint32_t offset, uint32_t value)
{
   memcpy(base + offset, &value, sizeof(value));
}

static void
vtvk_store_u64(uint8_t *base, uint32_t offset, uint64_t value)
{
   memcpy(base + offset, &value, sizeof(value));
}

static void
vtvk_store_f32(uint8_t *base, uint32_t offset, float value)
{
   memcpy(base + offset, &value, sizeof(value));
}

static uint32_t
vtvk_load_u32(const uint8_t *base, uint32_t offset)
{
   uint32_t value;
   memcpy(&value, base + offset, sizeof(value));
   return value;
}

static float
vtvk_load_f32(const uint8_t *base, uint32_t offset)
{
   float value;
   memcpy(&value, base + offset, sizeof(value));
   return value;
}

static struct vtvk_aabb
vtvk_header_aabb(const uint8_t *as_data)
{
   struct vtvk_aabb aabb;
   aabb.min[0] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MIN_X);
   aabb.min[1] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MIN_Y);
   aabb.min[2] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MIN_Z);
   aabb.max[0] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MAX_X);
   aabb.max[1] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MAX_Y);
   aabb.max[2] = vtvk_load_f32(as_data, VENTUS_AS_HEADER_ROOT_AABB_MAX_Z);
   return aabb;
}

static void
vtvk_write_header(uint8_t *dst, enum ventus_as_type type, uint32_t root_ref,
                  uint32_t node_count, uint32_t node_region_size,
                  uint32_t primitive_count, uint32_t instance_count,
                  const struct vtvk_aabb *root_aabb)
{
   memset(dst, 0, VENTUS_AS_HEADER_SIZE);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_MAGIC, VENTUS_AS_MAGIC);
   vtvk_store_u16(dst, VENTUS_AS_HEADER_VERSION, VENTUS_AS_VERSION);
   vtvk_store_u16(dst, VENTUS_AS_HEADER_HEADER_SIZE, VENTUS_AS_HEADER_SIZE);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_TYPE, type);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_ROOT_NODE_REF, root_ref);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_NODE_COUNT, node_count);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_NODE_REGION_OFFSET,
                  VENTUS_AS_HEADER_SIZE);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_NODE_REGION_SIZE, node_region_size);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_PRIMITIVE_REGION_OFFSET,
                  VENTUS_AS_HEADER_SIZE);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_PRIMITIVE_COUNT, primitive_count);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_INSTANCE_REGION_OFFSET,
                  VENTUS_AS_HEADER_SIZE);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_INSTANCE_COUNT, instance_count);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MIN_X, root_aabb->min[0]);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MIN_Y, root_aabb->min[1]);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MIN_Z, root_aabb->min[2]);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MAX_X, root_aabb->max[0]);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MAX_Y, root_aabb->max[1]);
   vtvk_store_f32(dst, VENTUS_AS_HEADER_ROOT_AABB_MAX_Z, root_aabb->max[2]);
}

static void
vtvk_write_box4(uint8_t *dst, uint32_t offset,
                const std::vector<struct vtvk_build_node> &children)
{
   uint8_t *node = dst + offset;
   memset(node, 0, VENTUS_AS_BOX4_NODE_SIZE);
   for (uint32_t i = 0; i < 4; i++)
      vtvk_store_u32(node, VENTUS_BOX4_CHILD_REF + i * 4,
                     VENTUS_AS_INVALID_NODE);

   for (uint32_t i = 0; i < children.size(); i++) {
      vtvk_store_u32(node, VENTUS_BOX4_CHILD_REF + i * 4, children[i].ref);
      vtvk_store_f32(node, VENTUS_BOX4_MIN_X + i * 4, children[i].bounds.min[0]);
      vtvk_store_f32(node, VENTUS_BOX4_MIN_Y + i * 4, children[i].bounds.min[1]);
      vtvk_store_f32(node, VENTUS_BOX4_MIN_Z + i * 4, children[i].bounds.min[2]);
      vtvk_store_f32(node, VENTUS_BOX4_MAX_X + i * 4, children[i].bounds.max[0]);
      vtvk_store_f32(node, VENTUS_BOX4_MAX_Y + i * 4, children[i].bounds.max[1]);
      vtvk_store_f32(node, VENTUS_BOX4_MAX_Z + i * 4, children[i].bounds.max[2]);
   }
}

static uint32_t
vtvk_next_node_offset(uint32_t *cursor, uint32_t size)
{
   uint32_t offset = *cursor;
   *cursor += size;
   return offset;
}

static struct vtvk_build_node
vtvk_build_box_tree(uint8_t *dst, uint32_t *cursor,
                    std::vector<struct vtvk_build_node> leaves,
                    uint32_t *node_count)
{
   if (leaves.size() == 1)
      return leaves[0];

   while (leaves.size() > 1) {
      std::vector<struct vtvk_build_node> next;
      for (size_t i = 0; i < leaves.size(); i += 4) {
         std::vector<struct vtvk_build_node> children;
         struct vtvk_aabb bounds;
         vtvk_aabb_empty(&bounds);

         for (size_t j = i; j < leaves.size() && j < i + 4; j++) {
            children.push_back(leaves[j]);
            vtvk_aabb_include_aabb(&bounds, &leaves[j].bounds);
         }

         uint32_t offset = vtvk_next_node_offset(cursor, VENTUS_AS_BOX4_NODE_SIZE);
         vtvk_write_box4(dst, offset, children);
         (*node_count)++;

         next.push_back({
            ventus_bvh_make_node_ref(offset, VENTUS_BVH_NODE_BOX4),
            bounds,
         });
      }
      leaves = next;
   }

   return leaves[0];
}

static struct vtvk_pipeline *
vtvk_pipeline_from_handle(VkPipeline pipeline)
{
   VK_FROM_HANDLE(vk_pipeline, vk_pipeline, pipeline);
   return vk_pipeline ? container_of(vk_pipeline, struct vtvk_pipeline, vk) : nullptr;
}

static void
vtvk_pipeline_destroy(struct vk_device *device,
                      struct vk_pipeline *vk_pipeline,
                      const VkAllocationCallbacks *pAllocator)
{
   struct vtvk_pipeline *pipeline =
      container_of(vk_pipeline, struct vtvk_pipeline, vk);

   delete pipeline->llvm_module;
   ralloc_free(pipeline->nir_mem_ctx);
   vk_free2(&device->alloc, pAllocator, pipeline->shaders);
   vk_free2(&device->alloc, pAllocator, pipeline->group_handles);
   vk_pipeline_free(device, pAllocator, vk_pipeline);
}

static const struct vk_pipeline_ops vtvk_pipeline_ops = {
   .destroy = vtvk_pipeline_destroy,
};

static bool
vtvk_rt_stage_supported(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_RAYGEN ||
          stage == MESA_SHADER_MISS ||
          stage == MESA_SHADER_CLOSEST_HIT ||
          stage == MESA_SHADER_ANY_HIT ||
          stage == MESA_SHADER_INTERSECTION;
}

static VkResult
vtvk_compile_rt_stage(struct vtvk_device *device,
                      struct vtvk_pipeline *pipeline,
                      VkPipelineCreateFlags2KHR pipeline_flags,
                      const VkPipelineShaderStageCreateInfo *stage_info,
                      nir_shader **nir_out)
{
   mesa_shader_stage stage = vk_to_mesa_shader_stage(stage_info->stage);

   if (!vtvk_rt_stage_supported(stage))
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   VkResult result =
      vk_pipeline_shader_stage_to_nir(&device->vk, pipeline_flags, stage_info,
                                      &vtvk_spirv_options, &vtvk_nir_options,
                                      pipeline->nir_mem_ctx, nir_out);
   if (result != VK_SUCCESS)
      return result;

   struct vt_nir_rt_config config = vt_nir_rt_default_config();
   struct vt_nir_rt_abi_layout abi = vt_nir_rt_default_abi_layout();
   vt_nir_rt_config_apply_abi_layout(&config, &abi);

   if (stage == MESA_SHADER_RAYGEN)
      NIR_PASS(_, *nir_out, vt_nir_lower_rt_io_monolithic, &config);

   NIR_PASS(_, *nir_out, vt_nir_lower_rt_abi, &config);
   nir_validate_shader(*nir_out, "after Ventus RT lowering");

   return VK_SUCCESS;
}

static void
vtvk_fill_group_handles(struct vtvk_pipeline *pipeline,
                        const VkRayTracingPipelineCreateInfoKHR *info)
{
   const uint32_t handle_size = 32;
   pipeline->group_count = info->groupCount;

   for (uint32_t g = 0; g < info->groupCount; g++) {
      uint8_t *handle = pipeline->group_handles + g * handle_size;
      const VkRayTracingShaderGroupCreateInfoKHR *group = &info->pGroups[g];

      memcpy(handle, "VTRT", 4);
      memcpy(handle + 4, &g, sizeof(g));
      memcpy(handle + 8, &group->type, sizeof(group->type));
      memcpy(handle + 12, &group->generalShader, sizeof(group->generalShader));
      memcpy(handle + 16, &group->closestHitShader,
             sizeof(group->closestHitShader));
      memcpy(handle + 20, &group->anyHitShader, sizeof(group->anyHitShader));
      memcpy(handle + 24, &group->intersectionShader,
             sizeof(group->intersectionShader));
   }
}

static VkResult
vtvk_create_one_rt_pipeline(VkDevice _device,
                            VkPipelineCreateFlags2KHR pipeline_flags,
                            const VkRayTracingPipelineCreateInfoKHR *info,
                            const VkAllocationCallbacks *pAllocator,
                            VkPipeline *out)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);

   struct vtvk_pipeline *pipeline =
      (struct vtvk_pipeline *)vk_pipeline_zalloc(
         &device->vk, &vtvk_pipeline_ops, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
         pipeline_flags, pAllocator, sizeof(*pipeline));
   if (!pipeline)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   pipeline->nir_mem_ctx = ralloc_context(nullptr);
   pipeline->shader_count = info->stageCount;

   pipeline->shaders =
      (nir_shader **)vk_zalloc2(&device->vk.alloc, pAllocator,
                                sizeof(nir_shader *) * info->stageCount, 8,
                                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   pipeline->group_handles =
      (uint8_t *)vk_zalloc2(&device->vk.alloc, pAllocator,
                            32 * info->groupCount, 8,
                            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);

   if (!pipeline->nir_mem_ctx || !pipeline->shaders || !pipeline->group_handles) {
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   for (uint32_t s = 0; s < info->stageCount; s++) {
      VkResult result =
         vtvk_compile_rt_stage(device, pipeline, pipeline_flags,
                               &info->pStages[s], &pipeline->shaders[s]);
      if (result != VK_SUCCESS) {
         vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
         return result;
      }

      pipeline->vk.stages |= info->pStages[s].stage;
   }

   pipeline->per_lane_scratch_size =
      vt_nir_pipeline_compute_scratch(pipeline->shaders, pipeline->shader_count,
                                      &pipeline->scratch_info);

   struct vt_nir_to_llvm_options llvm_options = {
      .module_name = "ventus_rt_pipeline",
      .target_triple = "riscv32-unknown-elf",
      .data_layout = "e-m:e-p:32:32-i64:64-n32-S128",
   };

   pipeline->llvm_module =
      vt_nir_to_llvm_translate(pipeline->shaders, pipeline->shader_count,
                               &llvm_options);
   if (!pipeline->llvm_module) {
      vtvk_pipeline_destroy(&device->vk, &pipeline->vk, pAllocator);
      return vk_error(device, VK_ERROR_INITIALIZATION_FAILED);
   }

   vtvk_fill_group_handles(pipeline, info);

   mesa_logi("ventus-vk: RT pipeline compiled via SPIR-V -> NIR -> "
             "Ventus lowering -> LLVM; stages=%u groups=%u scratch=%u llvm_functions=%zu",
             info->stageCount, info->groupCount,
             pipeline->per_lane_scratch_size,
             pipeline->llvm_module->size());

   *out = vk_pipeline_to_handle(&pipeline->vk);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateRayTracingPipelinesKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   VkPipelineCache pipelineCache,
   uint32_t createInfoCount,
   const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
   const VkAllocationCallbacks *pAllocator,
   VkPipeline *pPipelines)
{
   (void)deferredOperation;
   (void)pipelineCache;

   VkResult final_result = VK_SUCCESS;

   for (uint32_t i = 0; i < createInfoCount; i++) {
      VkPipelineCreateFlags2KHR flags = vk_rt_pipeline_create_flags(&pCreateInfos[i]);
      VkResult result =
         vtvk_create_one_rt_pipeline(device, flags, &pCreateInfos[i],
                                     pAllocator, &pPipelines[i]);
      if (result != VK_SUCCESS) {
         pPipelines[i] = VK_NULL_HANDLE;
         final_result = result;

         if (!(pCreateInfos[i].flags &
               VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT))
            continue;

         for (uint32_t j = i + 1; j < createInfoCount; j++)
            pPipelines[j] = VK_NULL_HANDLE;
         break;
      }
   }

   return final_result;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetRayTracingShaderGroupHandlesKHR(VkDevice device,
                                        VkPipeline pipeline,
                                        uint32_t firstGroup,
                                        uint32_t groupCount,
                                        size_t dataSize,
                                        void *pData)
{
   (void)device;
   struct vtvk_pipeline *vt_pipeline = vtvk_pipeline_from_handle(pipeline);
   const uint32_t handle_size = 32;

   if (!vt_pipeline || firstGroup > vt_pipeline->group_count ||
       groupCount > vt_pipeline->group_count - firstGroup)
      return VK_ERROR_UNKNOWN;

   size_t copy_size = (size_t)groupCount * handle_size;
   if (dataSize < copy_size)
      return VK_ERROR_UNKNOWN;

   memcpy(pData, vt_pipeline->group_handles + firstGroup * handle_size,
          copy_size);
   return VK_SUCCESS;
}

static const VkAccelerationStructureGeometryKHR *
vtvk_build_geometry(const VkAccelerationStructureBuildGeometryInfoKHR *info,
                    uint32_t index)
{
   return info->pGeometries ? &info->pGeometries[index] : info->ppGeometries[index];
}

static void
vtvk_read_vec3(const uint8_t *base, VkDeviceSize stride, uint32_t index,
               float out[3])
{
   const uint8_t *ptr = base + (VkDeviceSize)index * stride;
   memcpy(out, ptr, sizeof(float) * 3);
}

static void
vtvk_transform_point(const VkTransformMatrixKHR *m, const float in[3],
                     float out[3])
{
   for (unsigned row = 0; row < 3; row++) {
      out[row] = m->matrix[row][0] * in[0] +
                 m->matrix[row][1] * in[1] +
                 m->matrix[row][2] * in[2] +
                 m->matrix[row][3];
   }
}

static struct vtvk_aabb
vtvk_transform_aabb(const VkTransformMatrixKHR *m, const struct vtvk_aabb *in)
{
   struct vtvk_aabb out;
   vtvk_aabb_empty(&out);

   for (unsigned x = 0; x < 2; x++) {
      for (unsigned y = 0; y < 2; y++) {
         for (unsigned z = 0; z < 2; z++) {
            float p[3] = {
               x ? in->max[0] : in->min[0],
               y ? in->max[1] : in->min[1],
               z ? in->max[2] : in->min[2],
            };
            float tp[3];
            vtvk_transform_point(m, p, tp);
            vtvk_aabb_include_point(&out, tp);
         }
      }
   }

   return out;
}

static void
vtvk_write_triangle_leaf(uint8_t *dst, uint32_t offset, const float v0[3],
                         const float v1[3], const float v2[3],
                         uint32_t primitive_id, uint32_t geometry_id,
                         uint32_t sbt_record_offset, uint32_t flags,
                         uint64_t primitive_addr)
{
   uint8_t *node = dst + offset;
   memset(node, 0, VENTUS_AS_TRIANGLE_NODE_SIZE);
   memcpy(node + VENTUS_TRIANGLE_V0, v0, sizeof(float) * 3);
   memcpy(node + VENTUS_TRIANGLE_V1, v1, sizeof(float) * 3);
   memcpy(node + VENTUS_TRIANGLE_V2, v2, sizeof(float) * 3);
   vtvk_store_u32(node, VENTUS_TRIANGLE_PRIMITIVE_ID, primitive_id);
   vtvk_store_u32(node, VENTUS_TRIANGLE_GEOMETRY_ID, geometry_id);
   vtvk_store_u32(node, VENTUS_TRIANGLE_SBT_RECORD_OFFSET, sbt_record_offset);
   vtvk_store_u32(node, VENTUS_TRIANGLE_FLAGS, flags);
   vtvk_store_u64(node, VENTUS_TRIANGLE_PRIMITIVE_ADDR_LO, primitive_addr);
}

static void
vtvk_write_instance_leaf(uint8_t *dst, uint32_t offset,
                         const VkAccelerationStructureInstanceKHR *instance,
                         uint32_t instance_id)
{
   uint8_t *node = dst + offset;
   memset(node, 0, VENTUS_AS_INSTANCE_NODE_SIZE);
   vtvk_store_u64(node, VENTUS_INSTANCE_BLAS_ADDR_LO,
                  instance->accelerationStructureReference);
   vtvk_store_u32(node, VENTUS_INSTANCE_CUSTOM_INDEX,
                  instance->instanceCustomIndex);
   vtvk_store_u32(node, VENTUS_INSTANCE_MASK, instance->mask);
   vtvk_store_u32(node, VENTUS_INSTANCE_SBT_RECORD_OFFSET,
                  instance->instanceShaderBindingTableRecordOffset);
   vtvk_store_u32(node, VENTUS_INSTANCE_FLAGS, instance->flags);
   vtvk_store_u32(node, VENTUS_INSTANCE_INSTANCE_ID, instance_id);

   memcpy(node + VENTUS_INSTANCE_OBJECT_TO_WORLD, instance->transform.matrix,
          sizeof(instance->transform.matrix));

   float identity[3][4] = {
      { 1.0f, 0.0f, 0.0f, 0.0f },
      { 0.0f, 1.0f, 0.0f, 0.0f },
      { 0.0f, 0.0f, 1.0f, 0.0f },
   };
   memcpy(node + VENTUS_INSTANCE_WORLD_TO_OBJECT, identity, sizeof(identity));
}

static VkResult
vtvk_build_blas(struct vtvk_device *device,
                struct vtvk_acceleration_structure *as,
                const VkAccelerationStructureBuildGeometryInfoKHR *info,
                const VkAccelerationStructureBuildRangeInfoKHR *range)
{
   if (info->geometryCount != 1)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   const VkAccelerationStructureGeometryKHR *geom = vtvk_build_geometry(info, 0);
   if (geom->geometryType != VK_GEOMETRY_TYPE_TRIANGLES_KHR)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   const VkAccelerationStructureGeometryTrianglesDataKHR *triangles =
      &geom->geometry.triangles;
   if (triangles->vertexFormat != VK_FORMAT_R32G32B32_SFLOAT ||
       triangles->indexType != VK_INDEX_TYPE_UINT32)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   uint32_t primitive_count = range ? range->primitiveCount : 0;
   if (!primitive_count)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   uint8_t *dst = (uint8_t *)as->data;
   if (!dst)
      return vk_error(device, VK_ERROR_UNKNOWN);

   uint8_t *vertex_base = (uint8_t *)vtvk_buffer_host_ptr_from_addr(
      triangles->vertexData.deviceAddress, triangles->vertexStride *
      (triangles->maxVertex + 1));
   uint32_t *index_base = (uint32_t *)vtvk_buffer_host_ptr_from_addr(
      triangles->indexData.deviceAddress, sizeof(uint32_t) * primitive_count * 3);
   if (!vertex_base || !index_base)
      return vk_error(device, VK_ERROR_UNKNOWN);

   memset(dst, 0, as->size);
   uint32_t cursor = VENTUS_AS_HEADER_SIZE;
   uint32_t node_count = 0;
   std::vector<struct vtvk_build_node> leaves;

   for (uint32_t p = 0; p < primitive_count; p++) {
      uint32_t i0 = index_base[p * 3 + 0];
      uint32_t i1 = index_base[p * 3 + 1];
      uint32_t i2 = index_base[p * 3 + 2];
      float v0[3], v1[3], v2[3];
      vtvk_read_vec3(vertex_base, triangles->vertexStride, i0, v0);
      vtvk_read_vec3(vertex_base, triangles->vertexStride, i1, v1);
      vtvk_read_vec3(vertex_base, triangles->vertexStride, i2, v2);

      struct vtvk_aabb bounds;
      vtvk_aabb_empty(&bounds);
      vtvk_aabb_include_point(&bounds, v0);
      vtvk_aabb_include_point(&bounds, v1);
      vtvk_aabb_include_point(&bounds, v2);

      uint32_t offset =
         vtvk_next_node_offset(&cursor, VENTUS_AS_TRIANGLE_NODE_SIZE);
      vtvk_write_triangle_leaf(dst, offset, v0, v1, v2, p, 0, 0,
                               geom->flags & VK_GEOMETRY_OPAQUE_BIT_KHR ? 1 : 0,
                               triangles->vertexData.deviceAddress +
                                  (VkDeviceAddress)p * triangles->vertexStride);
      node_count++;
      leaves.push_back({
         ventus_bvh_make_node_ref(offset, VENTUS_BVH_NODE_TRIANGLE),
         bounds,
      });
   }

   struct vtvk_build_node root =
      vtvk_build_box_tree(dst, &cursor, leaves, &node_count);
   vtvk_write_header(dst, VENTUS_AS_TYPE_BLAS, root.ref, node_count,
                     cursor - VENTUS_AS_HEADER_SIZE, primitive_count, 0,
                     &root.bounds);
   as->address = (VkDeviceAddress)(uintptr_t)as->data;
   return VK_SUCCESS;
}

static VkResult
vtvk_build_tlas(struct vtvk_device *device,
                struct vtvk_acceleration_structure *as,
                const VkAccelerationStructureBuildGeometryInfoKHR *info,
                const VkAccelerationStructureBuildRangeInfoKHR *range)
{
   if (info->geometryCount != 1)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   const VkAccelerationStructureGeometryKHR *geom = vtvk_build_geometry(info, 0);
   if (geom->geometryType != VK_GEOMETRY_TYPE_INSTANCES_KHR)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   uint32_t instance_count = range ? range->primitiveCount : 0;
   if (!instance_count)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   const VkAccelerationStructureGeometryInstancesDataKHR *instances =
      &geom->geometry.instances;
   if (instances->arrayOfPointers)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   VkAccelerationStructureInstanceKHR *instance_base =
      (VkAccelerationStructureInstanceKHR *)vtvk_buffer_host_ptr_from_addr(
         instances->data.deviceAddress,
         sizeof(VkAccelerationStructureInstanceKHR) * instance_count);
   if (!instance_base)
      return vk_error(device, VK_ERROR_UNKNOWN);

   uint8_t *dst = (uint8_t *)as->data;
   if (!dst)
      return vk_error(device, VK_ERROR_UNKNOWN);

   memset(dst, 0, as->size);
   uint32_t cursor = VENTUS_AS_HEADER_SIZE;
   uint32_t node_count = 0;
   std::vector<struct vtvk_build_node> leaves;

   for (uint32_t i = 0; i < instance_count; i++) {
      const VkAccelerationStructureInstanceKHR *inst = &instance_base[i];
      const uint8_t *blas = (const uint8_t *)(uintptr_t)inst->accelerationStructureReference;
      if (!blas || vtvk_load_u32(blas, VENTUS_AS_HEADER_MAGIC) != VENTUS_AS_MAGIC)
         return vk_error(device, VK_ERROR_UNKNOWN);

      struct vtvk_aabb blas_bounds = vtvk_header_aabb(blas);
      struct vtvk_aabb bounds = vtvk_transform_aabb(&inst->transform,
                                                    &blas_bounds);
      uint32_t offset =
         vtvk_next_node_offset(&cursor, VENTUS_AS_INSTANCE_NODE_SIZE);
      vtvk_write_instance_leaf(dst, offset, inst, i);
      node_count++;
      leaves.push_back({
         ventus_bvh_make_node_ref(offset, VENTUS_BVH_NODE_INSTANCE),
         bounds,
      });
   }

   struct vtvk_build_node root =
      vtvk_build_box_tree(dst, &cursor, leaves, &node_count);
   vtvk_write_header(dst, VENTUS_AS_TYPE_TLAS, root.ref, node_count,
                     cursor - VENTUS_AS_HEADER_SIZE, 0, instance_count,
                     &root.bounds);
   as->address = (VkDeviceAddress)(uintptr_t)as->data;
   return VK_SUCCESS;
}

static VkResult
vtvk_build_as(struct vtvk_device *device,
              const VkAccelerationStructureBuildGeometryInfoKHR *info,
              const VkAccelerationStructureBuildRangeInfoKHR *const *ranges)
{
   struct vtvk_acceleration_structure *as =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)info->dstAccelerationStructure);
   if (!as)
      return vk_error(device, VK_ERROR_UNKNOWN);

   as->data = vtvk_buffer_host_ptr(as->buffer, as->offset, as->size);
   if (!as->data)
      return vk_error(device, VK_ERROR_UNKNOWN);

   const VkAccelerationStructureBuildRangeInfoKHR *range =
      ranges ? ranges[0] : NULL;

   if (info->type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR)
      return vtvk_build_blas(device, as, info, range);

   if (info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR)
      return vtvk_build_tlas(device, as, info, range);

   return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdTraceRaysKHR(VkCommandBuffer commandBuffer,
                     const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
                     const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
                     const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
                     const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
                     uint32_t width,
                     uint32_t height,
                     uint32_t depth)
{
   static bool logged;
   (void)pCallableShaderBindingTable;

   VkPipeline pipeline_handle =
      vtvk_cmd_buffer_bound_rt_pipeline(commandBuffer);
   struct vtvk_pipeline *pipeline =
      vtvk_pipeline_from_handle(pipeline_handle);
   VkDescriptorSet descriptor_set =
      vtvk_cmd_buffer_bound_descriptor_set(commandBuffer, 0);
   VkAccelerationStructureKHR tlas_handle =
      vtvk_descriptor_set_acceleration_structure(descriptor_set, 0);
   struct vtvk_acceleration_structure *tlas =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)tlas_handle);

   VkExtent3D image_extent = {};
   VkFormat image_format = VK_FORMAT_UNDEFINED;
   VkImage storage_image = vtvk_descriptor_set_image(descriptor_set, 1);
   void *storage_image_data =
      vtvk_image_host_ptr(storage_image, &image_extent, &image_format);

   VkDeviceSize camera_range = 0;
   void *camera_data =
      vtvk_descriptor_set_buffer_ptr(descriptor_set, 2, &camera_range);

   if (!pipeline || !tlas || !tlas->address || !storage_image_data ||
       !camera_data) {
      mesa_loge("ventus-vk: vkCmdTraceRaysKHR missing Spike dispatch state "
                "pipeline=%p tlas=0x%" PRIx64 " image=%p camera=%p",
                (void *)pipeline, tlas ? (uint64_t)tlas->address : 0,
                storage_image_data, camera_data);
      return;
   }

   if (!logged) {
      logged = true;
      mesa_logi("ventus-vk: vkCmdTraceRaysKHR collected Spike dispatch state "
                "%ux%ux%u tlas=0x%" PRIx64 " image=%ux%u format=%u "
                "scratch=%u sbt={raygen:0x%" PRIx64 ",miss:0x%" PRIx64
                ",hit:0x%" PRIx64 "}",
                width, height, depth, (uint64_t)tlas->address,
                image_extent.width, image_extent.height, image_format,
                pipeline->per_lane_scratch_size,
                pRaygenShaderBindingTable ?
                   (uint64_t)pRaygenShaderBindingTable->deviceAddress : 0,
                pMissShaderBindingTable ?
                   (uint64_t)pMissShaderBindingTable->deviceAddress : 0,
                pHitShaderBindingTable ?
                   (uint64_t)pHitShaderBindingTable->deviceAddress : 0);
   }
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL
vtvk_GetAccelerationStructureDeviceAddressKHR(
   VkDevice device,
   const VkAccelerationStructureDeviceAddressInfoKHR *pInfo)
{
   (void)device;
   struct vtvk_acceleration_structure *as =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)pInfo->accelerationStructure);
   if (as && !as->address) {
      as->data = vtvk_buffer_host_ptr(as->buffer, as->offset, as->size);
      as->address = (VkDeviceAddress)(uintptr_t)as->data;
   }
   return as ? as->address : 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateAccelerationStructureKHR(
   VkDevice _device,
   const VkAccelerationStructureCreateInfoKHR *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkAccelerationStructureKHR *pAccelerationStructure)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &device->vk.alloc;

   struct vtvk_acceleration_structure *as =
      (struct vtvk_acceleration_structure *)
         vk_zalloc(alloc, sizeof(*as), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!as)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   as->type = pCreateInfo->type;
   as->buffer = pCreateInfo->buffer;
   as->offset = pCreateInfo->offset;
   as->size = pCreateInfo->size;
   as->data = vtvk_buffer_host_ptr(as->buffer, as->offset, as->size);
   as->address = (VkDeviceAddress)(uintptr_t)as->data;

   *pAccelerationStructure =
      (VkAccelerationStructureKHR)vtvk_ptr_to_handle(as);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyAccelerationStructureKHR(
   VkDevice _device,
   VkAccelerationStructureKHR accelerationStructure,
   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc =
      pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)accelerationStructure));
}

VKAPI_ATTR void VKAPI_CALL
vtvk_GetAccelerationStructureBuildSizesKHR(
   VkDevice device,
   VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   (void)device;
   (void)buildType;
   uint64_t primitive_count = 1;
   if (pMaxPrimitiveCounts && pBuildInfo && pBuildInfo->geometryCount)
      primitive_count = pMaxPrimitiveCounts[0] ? pMaxPrimitiveCounts[0] : 1;

   uint64_t leaf_size =
      pBuildInfo && pBuildInfo->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
         ? VENTUS_AS_INSTANCE_NODE_SIZE : VENTUS_AS_TRIANGLE_NODE_SIZE;
   uint64_t internal_count = primitive_count > 1 ? (primitive_count + 2) / 3 : 0;
   uint64_t internal_bytes = internal_count * VENTUS_AS_BOX4_NODE_SIZE;
   uint64_t primitive_bytes = primitive_count * leaf_size;

   pSizeInfo->accelerationStructureSize =
      VENTUS_AS_HEADER_SIZE + primitive_bytes + internal_bytes + 256;
   pSizeInfo->updateScratchSize = 2048 + primitive_bytes;
   pSizeInfo->buildScratchSize = 4096 + primitive_bytes;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct vtvk_device *device =
      container_of(vk_cmd->base.device, struct vtvk_device, vk);

   for (uint32_t i = 0; i < infoCount; i++) {
      VkResult result =
         vtvk_build_as(device, &pInfos[i], ppBuildRangeInfos + i);
      if (result != VK_SUCCESS) {
         mesa_loge("ventus-vk: AS build failed for Ventus BVH ABI v1 (%d)",
                   result);
         return;
      }
   }

   mesa_logi("ventus-vk: AS build wrote Ventus BVH ABI v1 objects=%u",
             infoCount);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBuildAccelerationStructuresIndirectKHR(
   VkCommandBuffer commandBuffer,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkDeviceAddress *pIndirectDeviceAddresses,
   const uint32_t *pIndirectStrides,
   const uint32_t *const *ppMaxPrimitiveCounts)
{
   (void)commandBuffer;
   (void)infoCount;
   (void)pInfos;
   (void)pIndirectDeviceAddresses;
   (void)pIndirectStrides;
   (void)ppMaxPrimitiveCounts;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_BuildAccelerationStructuresKHR(
   VkDevice device,
   VkDeferredOperationKHR deferredOperation,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   (void)device;
   (void)deferredOperation;
   VK_FROM_HANDLE(vtvk_device, vdevice, device);

   for (uint32_t i = 0; i < infoCount; i++) {
      VkResult result =
         vtvk_build_as(vdevice, &pInfos[i], ppBuildRangeInfos + i);
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

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
vtvk_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   (void)commandBuffer;
   (void)pInfo;
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
   (void)device;
   (void)pAllocator;
   *pDeferredOperation = VK_NULL_HANDLE;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyDeferredOperationKHR(VkDevice device,
                                 VkDeferredOperationKHR operation,
                                 const VkAllocationCallbacks *pAllocator)
{
   (void)device;
   (void)operation;
   (void)pAllocator;
}

VKAPI_ATTR uint32_t VKAPI_CALL
vtvk_GetDeferredOperationMaxConcurrencyKHR(VkDevice device,
                                           VkDeferredOperationKHR operation)
{
   (void)device;
   (void)operation;
   return 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetDeferredOperationResultKHR(VkDevice device,
                                   VkDeferredOperationKHR operation)
{
   (void)device;
   (void)operation;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_DeferredOperationJoinKHR(VkDevice device,
                              VkDeferredOperationKHR operation)
{
   (void)device;
   (void)operation;
   return VK_SUCCESS;
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL
vtvk_GetBufferDeviceAddressKHR(VkDevice device,
                               const VkBufferDeviceAddressInfo *pInfo)
{
   (void)device;
   return pInfo ? vtvk_buffer_get_device_address(pInfo->buffer) : 0;
}

VKAPI_ATTR uint64_t VKAPI_CALL
vtvk_GetBufferOpaqueCaptureAddressKHR(VkDevice device,
                                      const VkBufferDeviceAddressInfo *pInfo)
{
   (void)device;
   (void)pInfo;
   return 0;
}

VKAPI_ATTR uint64_t VKAPI_CALL
vtvk_GetDeviceMemoryOpaqueCaptureAddressKHR(
   VkDevice device,
   const VkDeviceMemoryOpaqueCaptureAddressInfo *pInfo)
{
   (void)device;
   (void)pInfo;
   return 0;
}
