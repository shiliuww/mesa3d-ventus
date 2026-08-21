#include "vtvk_private.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "util/log.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"
#include "ventus/common/ventus_rt_bvh.h"
#include "ventus/common/ventus_rt_bvh_v2.h"

struct vtvk_aabb {
   float min[3];
   float max[3];
};

struct vtvk_build_node {
   uint32_t ref;
   struct vtvk_aabb bounds;
};
static std::vector<struct vtvk_acceleration_structure *> vtvk_as_registry;

#define VTVK_AS_DEVICE_ADDRESS_ALIGNMENT 256u

static constexpr uint16_t vtvk_target_as_version = VENTUS_AS_VERSION_V2;

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

static uint8_t *
vtvk_alloc_as_snapshot(VkDeviceSize size)
{
   if (!size || size > SIZE_MAX ||
       size_t(size) > SIZE_MAX - (VTVK_AS_DEVICE_ADDRESS_ALIGNMENT - 1))
      return nullptr;

   const size_t allocation_size =
      (size_t(size) + VTVK_AS_DEVICE_ADDRESS_ALIGNMENT - 1) &
      ~size_t(VTVK_AS_DEVICE_ADDRESS_ALIGNMENT - 1);
   uint8_t *snapshot = (uint8_t *)aligned_alloc(
      VTVK_AS_DEVICE_ADDRESS_ALIGNMENT, allocation_size);
   if (snapshot)
      memset(snapshot, 0, allocation_size);
   return snapshot;
}

static void
vtvk_aabb_empty(struct vtvk_aabb *aabb)
{
   for (unsigned i = 0; i < 3; i++) {
      aabb->min[i] = FLT_MAX;
      aabb->max[i] = -FLT_MAX;
   }
}

static void
vtvk_aabb_zero(struct vtvk_aabb *aabb)
{
   memset(aabb, 0, sizeof(*aabb));
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

static uint16_t
vtvk_load_u16(const uint8_t *base, uint32_t offset)
{
   uint16_t value;
   memcpy(&value, base + offset, sizeof(value));
   return value;
}

static uint64_t
vtvk_load_u64(const uint8_t *base, uint32_t offset)
{
   uint64_t value;
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
vtvk_log_aabb(const char *name, const struct vtvk_aabb *aabb)
{
   mesa_logi("ventus-vk: %s aabb min=(%.6g %.6g %.6g) max=(%.6g %.6g %.6g)",
             name, aabb->min[0], aabb->min[1], aabb->min[2],
             aabb->max[0], aabb->max[1], aabb->max[2]);
}

static void
vtvk_log_transform(const char *name, const float matrix[3][4])
{
   mesa_logi("ventus-vk: %s [%.6g %.6g %.6g %.6g] "
             "[%.6g %.6g %.6g %.6g] [%.6g %.6g %.6g %.6g]",
             name,
             matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3],
             matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3],
             matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]);
}

static void
vtvk_write_header(uint8_t *dst, uint16_t version, enum ventus_as_type type,
                  uint32_t root_ref,
                  uint32_t node_count, uint32_t node_region_size,
                  uint32_t primitive_count, uint32_t instance_count,
                  const struct vtvk_aabb *root_aabb)
{
   memset(dst, 0, VENTUS_AS_HEADER_SIZE);
   vtvk_store_u32(dst, VENTUS_AS_HEADER_MAGIC, VENTUS_AS_MAGIC);
   vtvk_store_u16(dst, VENTUS_AS_HEADER_VERSION, version);
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

static uint32_t
vtvk_align_u32(uint32_t value, uint32_t alignment)
{
   return (value + alignment - 1) & ~(alignment - 1);
}

static void
vtvk_write_v2_child_parent(uint8_t *dst, uint32_t parent_offset,
                           uint32_t child_ref, uint8_t child_index)
{
   const uint32_t child_type = ventus_bvh_node_ref_type(child_ref);
   const uint32_t child_offset = ventus_bvh_node_ref_offset(child_ref);
   uint8_t *child = dst + child_offset;
   if (child_type == VENTUS_BVH_NODE_BOX4) {
      vtvk_store_u32(child, VENTUS_BOX4_V2_PARENT_NODE_REF,
                     ventus_bvh_make_node_ref(parent_offset,
                                              VENTUS_BVH_NODE_BOX4));
      child[VENTUS_BOX4_V2_PARENT_CHILD_INDEX] = child_index;
      child[VENTUS_BOX4_V2_NODE_FLAGS] = 0;
   } else if (child_type == VENTUS_BVH_NODE_INSTANCE) {
      vtvk_store_u32(child, VENTUS_INSTANCE_V2_PARENT_EDGE,
                     parent_offset | child_index);
   }
}

static void
vtvk_write_v2_root(uint8_t *dst, uint32_t root_ref)
{
   if (root_ref == VENTUS_AS_INVALID_NODE)
      return;

   const uint32_t root_type = ventus_bvh_node_ref_type(root_ref);
   uint8_t *root = dst + ventus_bvh_node_ref_offset(root_ref);
   if (root_type == VENTUS_BVH_NODE_BOX4) {
      vtvk_store_u32(root, VENTUS_BOX4_V2_PARENT_NODE_REF,
                     VENTUS_AS_INVALID_NODE);
      root[VENTUS_BOX4_V2_PARENT_CHILD_INDEX] = 0xff;
      root[VENTUS_BOX4_V2_NODE_FLAGS] = VENTUS_BOX4_V2_NODE_FLAG_ROOT;
   } else if (root_type == VENTUS_BVH_NODE_INSTANCE) {
      vtvk_store_u32(root, VENTUS_INSTANCE_V2_PARENT_EDGE,
                     VENTUS_INSTANCE_V2_ROOT_PARENT_EDGE);
   }
}

static void
vtvk_write_box4(uint8_t *dst, uint16_t version, uint32_t offset,
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
      if (version == VENTUS_AS_VERSION_V2)
         vtvk_write_v2_child_parent(dst, offset, children[i].ref, i);
   }
}

static uint32_t
vtvk_next_node_offset(uint32_t *cursor, uint32_t size)
{
   uint32_t offset = *cursor;
   *cursor += size;
   return offset;
}

static uint64_t
vtvk_internal_box_count(uint64_t leaf_count)
{
   uint64_t total = 0;
   uint64_t level_count = leaf_count;

   while (level_count > 1) {
      level_count = (level_count + 3) / 4;
      total += level_count;
   }

   return total;
}

static uint64_t
vtvk_as_required_size(uint16_t version, uint64_t leaf_bytes,
                      uint64_t internal_count)
{
   uint64_t size = VENTUS_AS_HEADER_SIZE + leaf_bytes;
   if (version == VENTUS_AS_VERSION_V2 && internal_count)
      size = (size + VENTUS_AS_BOX4_NODE_SIZE - 1) &
             ~(uint64_t(VENTUS_AS_BOX4_NODE_SIZE) - 1);
   return size + internal_count * VENTUS_AS_BOX4_NODE_SIZE;
}

static struct vtvk_build_node
vtvk_build_box_tree(uint8_t *dst, uint16_t version, uint32_t *cursor,
                    std::vector<struct vtvk_build_node> leaves,
                    uint32_t *node_count)
{
   if (leaves.size() == 1)
      return leaves[0];

   if (version == VENTUS_AS_VERSION_V2)
      *cursor = vtvk_align_u32(*cursor, VENTUS_AS_BOX4_NODE_SIZE);

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
         vtvk_write_box4(dst, version, offset, children);
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

static VkResult
vtvk_validate_built_as(struct vtvk_device *device, const uint8_t *dst,
                       uint32_t used_size, uint16_t version)
{
   if (version != VENTUS_AS_VERSION_V2)
      return VK_SUCCESS;

   struct ventus_vtas_v2_validation_result validation = {};
   if (ventus_vtas_v2_validate(dst, used_size, &validation))
      return VK_SUCCESS;

   mesa_loge("ventus-vk: rejected malformed VTAS V2: %s ref=0x%x detail=0x%x",
             ventus_vtas_v2_validation_error_string(validation.error),
             validation.node_ref, validation.detail);
   return vk_error(device, VK_ERROR_UNKNOWN);
}

struct vtvk_acceleration_structure *
vtvk_find_as_by_address(VkDeviceAddress address)
{
   for (struct vtvk_acceleration_structure *as : vtvk_as_registry) {
      if (as && (as->address == address ||
                 (as->snapshot &&
                  address >= (VkDeviceAddress)(uintptr_t)as->snapshot &&
                  address < (VkDeviceAddress)(uintptr_t)as->snapshot +
                               as->snapshot_size) ||
                 (as->data && (VkDeviceAddress)(uintptr_t)as->data == address) ||
                 (as->data && address >= (VkDeviceAddress)(uintptr_t)as->data &&
                  address < (VkDeviceAddress)(uintptr_t)as->data + as->size)))
         return as;
   }
   return nullptr;
}
static bool
vtvk_env_enabled(const char *name)
{
   const char *value = getenv(name);
   if (!value || !value[0])
      return false;
   return strcmp(value, "0") != 0 &&
          strcmp(value, "false") != 0 &&
          strcmp(value, "FALSE") != 0 &&
          strcmp(value, "off") != 0 &&
          strcmp(value, "OFF") != 0;
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

static bool
vtvk_invert_transform(const VkTransformMatrixKHR *m, float inv[3][4])
{
   const float a00 = m->matrix[0][0], a01 = m->matrix[0][1],
               a02 = m->matrix[0][2];
   const float a10 = m->matrix[1][0], a11 = m->matrix[1][1],
               a12 = m->matrix[1][2];
   const float a20 = m->matrix[2][0], a21 = m->matrix[2][1],
               a22 = m->matrix[2][2];
   const float det =
      a00 * (a11 * a22 - a12 * a21) -
      a01 * (a10 * a22 - a12 * a20) +
      a02 * (a10 * a21 - a11 * a20);

   if (std::fabs(det) < 1.0e-12f)
      return false;

   const float inv_det = 1.0f / det;
   inv[0][0] = (a11 * a22 - a12 * a21) * inv_det;
   inv[0][1] = (a02 * a21 - a01 * a22) * inv_det;
   inv[0][2] = (a01 * a12 - a02 * a11) * inv_det;
   inv[1][0] = (a12 * a20 - a10 * a22) * inv_det;
   inv[1][1] = (a00 * a22 - a02 * a20) * inv_det;
   inv[1][2] = (a02 * a10 - a00 * a12) * inv_det;
   inv[2][0] = (a10 * a21 - a11 * a20) * inv_det;
   inv[2][1] = (a01 * a20 - a00 * a21) * inv_det;
   inv[2][2] = (a00 * a11 - a01 * a10) * inv_det;

   const float t[3] = {
      m->matrix[0][3],
      m->matrix[1][3],
      m->matrix[2][3],
   };
   for (unsigned row = 0; row < 3; row++) {
      inv[row][3] = -(inv[row][0] * t[0] +
                      inv[row][1] * t[1] +
                      inv[row][2] * t[2]);
   }

   return true;
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
vtvk_write_aabb_leaf(uint8_t *dst, uint32_t offset,
                     const struct vtvk_aabb *bounds, uint32_t primitive_id,
                     uint32_t geometry_id, uint32_t sbt_record_offset,
                     uint32_t flags, uint64_t primitive_addr)
{
   uint8_t *node = dst + offset;
   memset(node, 0, VENTUS_AS_AABB_NODE_SIZE);
   memcpy(node + VENTUS_AABB_MIN, bounds->min, sizeof(bounds->min));
   memcpy(node + VENTUS_AABB_MAX, bounds->max, sizeof(bounds->max));
   vtvk_store_u32(node, VENTUS_AABB_PRIMITIVE_ID, primitive_id);
   vtvk_store_u32(node, VENTUS_AABB_GEOMETRY_ID, geometry_id);
   vtvk_store_u32(node, VENTUS_AABB_SBT_RECORD_OFFSET, sbt_record_offset);
   vtvk_store_u32(node, VENTUS_AABB_HIT_KIND, 0xff);
   vtvk_store_u32(node, VENTUS_AABB_FLAGS, flags);
   vtvk_store_u64(node, VENTUS_AABB_PRIMITIVE_ADDR_LO, primitive_addr);
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

   float world_to_object[3][4] = {
      { 1.0f, 0.0f, 0.0f, 0.0f },
      { 0.0f, 1.0f, 0.0f, 0.0f },
      { 0.0f, 0.0f, 1.0f, 0.0f },
   };
   (void)vtvk_invert_transform(&instance->transform, world_to_object);
   memcpy(node + VENTUS_INSTANCE_WORLD_TO_OBJECT, world_to_object,
          sizeof(world_to_object));
}

static VkResult
vtvk_append_aabb_geometry(
   struct vtvk_device *device,
   struct vtvk_acceleration_structure *as, uint8_t *dst,
   const VkAccelerationStructureGeometryKHR *geom,
   const VkAccelerationStructureBuildRangeInfoKHR *range,
   uint32_t geometry_id, uint32_t *cursor, uint32_t *node_count,
   std::vector<struct vtvk_build_node> *leaves)
{
   const VkAccelerationStructureGeometryAabbsDataKHR *aabbs =
      &geom->geometry.aabbs;
   const uint32_t primitive_count = range ? range->primitiveCount : 0;
   if (!primitive_count || aabbs->stride < sizeof(VkAabbPositionsKHR))
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   VkDeviceAddress primitive_addr = aabbs->data.deviceAddress;
   if (range) {
      if (UINT64_MAX - primitive_addr < range->primitiveOffset)
         return vk_error(device, VK_ERROR_UNKNOWN);
      primitive_addr += range->primitiveOffset;
   }

   const uint64_t last_offset =
      uint64_t(primitive_count - 1) * uint64_t(aabbs->stride);
   if (last_offset > SIZE_MAX - sizeof(VkAabbPositionsKHR))
      return vk_error(device, VK_ERROR_UNKNOWN);
   const size_t source_size =
      size_t(last_offset) + sizeof(VkAabbPositionsKHR);
   const uint8_t *source = (const uint8_t *)vtvk_buffer_host_ptr_from_addr(
      primitive_addr, source_size);
   if (!source || !as->data)
      return vk_error(device, VK_ERROR_UNKNOWN);

   for (uint32_t p = 0; p < primitive_count; p++) {
      VkAabbPositionsKHR positions;
      memcpy(&positions, source + uint64_t(p) * aabbs->stride,
             sizeof(positions));

      struct vtvk_aabb bounds = {
         { positions.minX, positions.minY, positions.minZ },
         { positions.maxX, positions.maxY, positions.maxZ },
      };
      for (uint32_t axis = 0; axis < 3; axis++) {
         if (!std::isfinite(bounds.min[axis]) ||
             !std::isfinite(bounds.max[axis]) ||
             bounds.min[axis] > bounds.max[axis])
            return vk_error(device, VK_ERROR_UNKNOWN);
      }

      if (*cursor > as->size ||
          VENTUS_AS_AABB_NODE_SIZE > as->size - *cursor)
         return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      const uint32_t offset =
         vtvk_next_node_offset(cursor, VENTUS_AS_AABB_NODE_SIZE);
      vtvk_write_aabb_leaf(
         dst, offset, &bounds, p, geometry_id, 0,
         geom->flags & VK_GEOMETRY_OPAQUE_BIT_KHR ? 1 : 0,
         primitive_addr + uint64_t(p) * aabbs->stride);
      (*node_count)++;
      leaves->push_back({
         ventus_bvh_make_node_ref(offset, VENTUS_BVH_NODE_AABB),
         bounds,
      });
   }

   return VK_SUCCESS;
}

static VkResult
vtvk_append_triangle_geometry(
   struct vtvk_device *device,
   struct vtvk_acceleration_structure *as, uint8_t *dst,
   const VkAccelerationStructureGeometryKHR *geom,
   const VkAccelerationStructureBuildRangeInfoKHR *range,
   uint32_t geometry_id, uint32_t *cursor, uint32_t *node_count,
   std::vector<struct vtvk_build_node> *leaves)
{
   const VkAccelerationStructureGeometryTrianglesDataKHR *triangles =
      &geom->geometry.triangles;
   if (triangles->vertexFormat != VK_FORMAT_R32G32B32_SFLOAT ||
       triangles->vertexStride < sizeof(float) * 3)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   const uint32_t primitive_count = range ? range->primitiveCount : 0;
   if (!primitive_count)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   uint32_t index_size;
   switch (triangles->indexType) {
   case VK_INDEX_TYPE_UINT16:
      index_size = sizeof(uint16_t);
      break;
   case VK_INDEX_TYPE_UINT32:
      index_size = sizeof(uint32_t);
      break;
   case VK_INDEX_TYPE_NONE_KHR:
      index_size = 0;
      break;
   default:
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }

   const bool indexed = index_size != 0;
   VkDeviceAddress index_addr = triangles->indexData.deviceAddress;
   const uint8_t *index_base = nullptr;
   if (indexed) {
      if (UINT64_MAX - index_addr < range->primitiveOffset)
         return vk_error(device, VK_ERROR_UNKNOWN);
      index_addr += range->primitiveOffset;
      const uint64_t index_count = uint64_t(primitive_count) * 3;
      if (index_count > SIZE_MAX / index_size)
         return vk_error(device, VK_ERROR_UNKNOWN);
      index_base = (const uint8_t *)vtvk_buffer_host_ptr_from_addr(
         index_addr, size_t(index_count) * index_size);
      if (!index_base)
         return vk_error(device, VK_ERROR_UNKNOWN);
   }

   const auto load_index = [&](uint64_t index) {
      uint32_t value = 0;
      if (index_size == sizeof(uint16_t)) {
         uint16_t value16;
         memcpy(&value16, index_base + index * index_size, sizeof(value16));
         value = value16;
      } else {
         memcpy(&value, index_base + index * index_size, sizeof(value));
      }
      return value;
   };

   const uint32_t first_vertex = range->firstVertex;
   uint32_t max_referenced_vertex;
   if (indexed) {
      uint32_t max_index = 0;
      for (uint64_t i = 0; i < uint64_t(primitive_count) * 3; i++)
         max_index = std::max(max_index, load_index(i));
      if (first_vertex > UINT32_MAX - max_index)
         return vk_error(device, VK_ERROR_UNKNOWN);
      max_referenced_vertex = max_index + first_vertex;
   } else {
      const uint64_t last_vertex =
         uint64_t(first_vertex) + uint64_t(primitive_count) * 3 - 1;
      if (last_vertex > UINT32_MAX)
         return vk_error(device, VK_ERROR_UNKNOWN);
      max_referenced_vertex = uint32_t(last_vertex);
   }
   if (max_referenced_vertex > triangles->maxVertex)
      return vk_error(device, VK_ERROR_UNKNOWN);

   VkDeviceAddress vertex_addr = triangles->vertexData.deviceAddress;
   if (!indexed) {
      if (UINT64_MAX - vertex_addr < range->primitiveOffset)
         return vk_error(device, VK_ERROR_UNKNOWN);
      vertex_addr += range->primitiveOffset;
   }
   const uint64_t vertex_bytes =
      uint64_t(triangles->vertexStride) *
      (uint64_t(max_referenced_vertex) + 1);
   if (vertex_bytes > SIZE_MAX)
      return vk_error(device, VK_ERROR_UNKNOWN);
   uint8_t *vertex_base = (uint8_t *)vtvk_buffer_host_ptr_from_addr(
      vertex_addr, size_t(vertex_bytes));
   if (!vertex_base)
      return vk_error(device, VK_ERROR_UNKNOWN);

   VkTransformMatrixKHR transform;
   const VkTransformMatrixKHR *build_transform = nullptr;
   if (triangles->transformData.deviceAddress) {
      VkDeviceAddress transform_addr =
         triangles->transformData.deviceAddress;
      if (UINT64_MAX - transform_addr < range->transformOffset)
         return vk_error(device, VK_ERROR_UNKNOWN);
      transform_addr += range->transformOffset;
      const void *transform_source = vtvk_buffer_host_ptr_from_addr(
         transform_addr, sizeof(transform));
      if (!transform_source)
         return vk_error(device, VK_ERROR_UNKNOWN);
      memcpy(&transform, transform_source, sizeof(transform));
      build_transform = &transform;
   }

   for (uint32_t p = 0; p < primitive_count; p++) {
      const uint64_t primitive_index = uint64_t(p) * 3;
      uint32_t i0, i1, i2;
      if (indexed) {
         i0 = load_index(primitive_index + 0) + first_vertex;
         i1 = load_index(primitive_index + 1) + first_vertex;
         i2 = load_index(primitive_index + 2) + first_vertex;
      } else {
         i0 = uint32_t(first_vertex + primitive_index + 0);
         i1 = uint32_t(first_vertex + primitive_index + 1);
         i2 = uint32_t(first_vertex + primitive_index + 2);
      }
      float v0[3], v1[3], v2[3];
      vtvk_read_vec3(vertex_base, triangles->vertexStride, i0, v0);
      vtvk_read_vec3(vertex_base, triangles->vertexStride, i1, v1);
      vtvk_read_vec3(vertex_base, triangles->vertexStride, i2, v2);
      if (build_transform) {
         float transformed[3];
         vtvk_transform_point(build_transform, v0, transformed);
         memcpy(v0, transformed, sizeof(v0));
         vtvk_transform_point(build_transform, v1, transformed);
         memcpy(v1, transformed, sizeof(v1));
         vtvk_transform_point(build_transform, v2, transformed);
         memcpy(v2, transformed, sizeof(v2));
      }

      struct vtvk_aabb bounds;
      vtvk_aabb_empty(&bounds);
      vtvk_aabb_include_point(&bounds, v0);
      vtvk_aabb_include_point(&bounds, v1);
      vtvk_aabb_include_point(&bounds, v2);

      if (*cursor > as->size ||
          VENTUS_AS_TRIANGLE_NODE_SIZE > as->size - *cursor)
         return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      uint32_t offset =
         vtvk_next_node_offset(cursor, VENTUS_AS_TRIANGLE_NODE_SIZE);
      const VkDeviceAddress primitive_addr =
         indexed ?
         index_addr + primitive_index * index_size :
         vertex_addr + uint64_t(i0) * triangles->vertexStride;
      vtvk_write_triangle_leaf(dst, offset, v0, v1, v2, p, geometry_id, 0,
                               geom->flags & VK_GEOMETRY_OPAQUE_BIT_KHR ? 1 : 0,
                               primitive_addr);
      (*node_count)++;
      leaves->push_back({
         ventus_bvh_make_node_ref(offset, VENTUS_BVH_NODE_TRIANGLE),
         bounds,
      });
   }

   return VK_SUCCESS;
}

static VkResult
vtvk_build_blas(struct vtvk_device *device,
                struct vtvk_acceleration_structure *as,
                const VkAccelerationStructureBuildGeometryInfoKHR *info,
                const VkAccelerationStructureBuildRangeInfoKHR *ranges)
{
   if (!info->geometryCount) {
      if (as->size < VENTUS_AS_HEADER_SIZE)
         return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      struct vtvk_aabb bounds;
      vtvk_aabb_zero(&bounds);
      memset(as->data, 0, as->size);
      vtvk_write_header((uint8_t *)as->data, vtvk_target_as_version,
                        VENTUS_AS_TYPE_BLAS,
                        VENTUS_AS_INVALID_NODE, 0, 0, 0, 0, &bounds);
      return vtvk_validate_built_as(device, (const uint8_t *)as->data,
                                    VENTUS_AS_HEADER_SIZE,
                                    vtvk_target_as_version);
   }
   if (!ranges)
      return vk_error(device, VK_ERROR_UNKNOWN);

   const VkAccelerationStructureGeometryKHR *first_geom =
      vtvk_build_geometry(info, 0);
   if (!first_geom)
      return vk_error(device, VK_ERROR_UNKNOWN);
   const VkGeometryTypeKHR geometry_type = first_geom->geometryType;
   if (geometry_type != VK_GEOMETRY_TYPE_TRIANGLES_KHR &&
       geometry_type != VK_GEOMETRY_TYPE_AABBS_KHR)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   uint64_t primitive_count = 0;
   for (uint32_t geometry_id = 0; geometry_id < info->geometryCount;
        geometry_id++) {
      const VkAccelerationStructureGeometryKHR *geom =
         vtvk_build_geometry(info, geometry_id);
      if (!geom || geom->geometryType != geometry_type)
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      primitive_count += ranges[geometry_id].primitiveCount;
      if (primitive_count > UINT32_MAX)
         return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   const uint64_t leaf_size =
      geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR ?
      VENTUS_AS_AABB_NODE_SIZE : VENTUS_AS_TRIANGLE_NODE_SIZE;
   const uint64_t internal_count = vtvk_internal_box_count(primitive_count);
   const uint64_t required_size = vtvk_as_required_size(
      vtvk_target_as_version, primitive_count * leaf_size, internal_count);
   if (required_size > as->size || required_size > UINT32_MAX)
      return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   uint8_t *dst = (uint8_t *)as->data;
   if (!dst)
      return vk_error(device, VK_ERROR_UNKNOWN);
   memset(dst, 0, as->size);

   uint32_t cursor = VENTUS_AS_HEADER_SIZE;
   uint32_t node_count = 0;
   std::vector<struct vtvk_build_node> leaves;
   leaves.reserve(size_t(primitive_count));

   for (uint32_t geometry_id = 0; geometry_id < info->geometryCount;
        geometry_id++) {
      const VkAccelerationStructureGeometryKHR *geom =
         vtvk_build_geometry(info, geometry_id);
      if (!ranges[geometry_id].primitiveCount)
         continue;
      VkResult result;
      if (geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR) {
         result = vtvk_append_aabb_geometry(
            device, as, dst, geom, &ranges[geometry_id], geometry_id,
            &cursor, &node_count, &leaves);
      } else {
         result = vtvk_append_triangle_geometry(
            device, as, dst, geom, &ranges[geometry_id], geometry_id,
            &cursor, &node_count, &leaves);
      }
      if (result != VK_SUCCESS)
         return result;
   }

   struct vtvk_build_node root;
   if (leaves.empty()) {
      root.ref = VENTUS_AS_INVALID_NODE;
      vtvk_aabb_zero(&root.bounds);
   } else {
      root = vtvk_build_box_tree(dst, vtvk_target_as_version, &cursor, leaves,
                                 &node_count);
   }
   if (vtvk_target_as_version == VENTUS_AS_VERSION_V2)
      vtvk_write_v2_root(dst, root.ref);
   vtvk_write_header(dst, vtvk_target_as_version, VENTUS_AS_TYPE_BLAS,
                     root.ref, node_count,
                     cursor - VENTUS_AS_HEADER_SIZE,
                     uint32_t(primitive_count), 0,
                     &root.bounds);
   VkResult validation = vtvk_validate_built_as(
      device, dst, cursor, vtvk_target_as_version);
   if (validation != VK_SUCCESS)
      return validation;
   if (vtvk_env_enabled("VENTUS_VK_DEBUG_AS")) {
      mesa_logi("ventus-vk: BLAS geometry_count=%u primitive_count=%" PRIu64
                " node_count=%u root=0x%x",
                info->geometryCount, primitive_count, node_count, root.ref);
      vtvk_log_aabb("BLAS root", &root.bounds);
   }
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
   if (!range)
      return vk_error(device, VK_ERROR_UNKNOWN);

   uint8_t *dst = (uint8_t *)as->data;
   if (!dst)
      return vk_error(device, VK_ERROR_UNKNOWN);
   const uint64_t internal_count = vtvk_internal_box_count(instance_count);
   const uint64_t required_size = vtvk_as_required_size(
      vtvk_target_as_version,
      uint64_t(instance_count) * VENTUS_AS_INSTANCE_NODE_SIZE,
      internal_count);
   if (required_size > as->size || required_size > UINT32_MAX)
      return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   if (!instance_count) {
      struct vtvk_aabb bounds;
      vtvk_aabb_zero(&bounds);
      memset(dst, 0, as->size);
      vtvk_write_header(dst, vtvk_target_as_version, VENTUS_AS_TYPE_TLAS,
                        VENTUS_AS_INVALID_NODE,
                        0, 0, 0, 0, &bounds);
      return vtvk_validate_built_as(device, dst, VENTUS_AS_HEADER_SIZE,
                                    vtvk_target_as_version);
   }

   const VkAccelerationStructureGeometryInstancesDataKHR *instances =
      &geom->geometry.instances;
   if (range->primitiveOffset >
       UINT64_MAX - instances->data.deviceAddress)
      return vk_error(device, VK_ERROR_UNKNOWN);
   const VkDeviceAddress instance_data_address =
      instances->data.deviceAddress + range->primitiveOffset;
   const VkAccelerationStructureInstanceKHR *instance_base = nullptr;
   const uint8_t *instance_address_array = nullptr;
   if (instances->arrayOfPointers) {
      instance_address_array =
         (const uint8_t *)vtvk_buffer_host_ptr_from_addr(
            instance_data_address,
            sizeof(VkDeviceAddress) * instance_count);
      if (!instance_address_array)
         return vk_error(device, VK_ERROR_UNKNOWN);
   } else {
      instance_base =
         (const VkAccelerationStructureInstanceKHR *)
            vtvk_buffer_host_ptr_from_addr(
               instance_data_address,
               sizeof(VkAccelerationStructureInstanceKHR) * instance_count);
      if (!instance_base)
         return vk_error(device, VK_ERROR_UNKNOWN);
   }

   memset(dst, 0, as->size);
   uint32_t cursor = VENTUS_AS_HEADER_SIZE;
   uint32_t node_count = 0;
   std::vector<struct vtvk_build_node> leaves;

   for (uint32_t i = 0; i < instance_count; i++) {
      const VkAccelerationStructureInstanceKHR *inst;
      if (instances->arrayOfPointers) {
         VkDeviceAddress instance_address;
         memcpy(&instance_address,
                instance_address_array + i * sizeof(instance_address),
                sizeof(instance_address));
         inst = (const VkAccelerationStructureInstanceKHR *)
            vtvk_buffer_host_ptr_from_addr(
               instance_address, sizeof(VkAccelerationStructureInstanceKHR));
      } else {
         inst = &instance_base[i];
      }
      if (!inst)
         return vk_error(device, VK_ERROR_UNKNOWN);

      const VkDeviceAddress blas_address =
         inst->accelerationStructureReference;
      struct vtvk_acceleration_structure *blas_as =
         vtvk_find_as_by_address(blas_address);
      const uint8_t *blas = blas_as ?
         (blas_as->snapshot ? blas_as->snapshot :
                              (const uint8_t *)blas_as->data) : nullptr;
      if (!blas_as || !blas_as->built || !blas ||
          vtvk_load_u32(blas, VENTUS_AS_HEADER_MAGIC) != VENTUS_AS_MAGIC ||
          vtvk_load_u16(blas, VENTUS_AS_HEADER_VERSION) !=
             vtvk_target_as_version)
         return vk_error(device, VK_ERROR_UNKNOWN);
      if (vtvk_target_as_version == VENTUS_AS_VERSION_V2) {
         struct ventus_vtas_v2_validation_result validation = {};
         const size_t blas_size = blas_as->snapshot ? blas_as->snapshot_size :
                                                      blas_as->size;
         if (!ventus_vtas_v2_validate(blas, blas_size, &validation))
            return vk_error(device, VK_ERROR_UNKNOWN);
      }

      struct vtvk_aabb blas_bounds = vtvk_header_aabb(blas);
      struct vtvk_aabb bounds = vtvk_transform_aabb(&inst->transform,
                                                    &blas_bounds);
      uint32_t offset =
         vtvk_next_node_offset(&cursor, VENTUS_AS_INSTANCE_NODE_SIZE);
      vtvk_write_instance_leaf(dst, offset, inst, i);
      if (vtvk_env_enabled("VENTUS_VK_DEBUG_AS")) {
         mesa_logi("ventus-vk: TLAS instance %u node=0x%x blas_ref=0x%" PRIx64
                   " stored=0x%" PRIx64,
                   i, offset,
                   (uint64_t)inst->accelerationStructureReference,
                   vtvk_load_u64(dst + offset, VENTUS_INSTANCE_BLAS_ADDR_LO));
         vtvk_log_transform("TLAS object_to_world",
                            inst->transform.matrix);
         float world_to_object[3][4];
         memcpy(world_to_object, dst + offset + VENTUS_INSTANCE_WORLD_TO_OBJECT,
                sizeof(world_to_object));
         vtvk_log_transform("TLAS world_to_object", world_to_object);
         vtvk_log_aabb("TLAS instance object bounds", &blas_bounds);
         vtvk_log_aabb("TLAS instance world bounds", &bounds);
      }
      node_count++;
      leaves.push_back({
         ventus_bvh_make_node_ref(offset, VENTUS_BVH_NODE_INSTANCE),
         bounds,
      });
   }

   struct vtvk_build_node root =
      vtvk_build_box_tree(dst, vtvk_target_as_version, &cursor, leaves,
                          &node_count);
   if (vtvk_target_as_version == VENTUS_AS_VERSION_V2)
      vtvk_write_v2_root(dst, root.ref);
   vtvk_write_header(dst, vtvk_target_as_version, VENTUS_AS_TYPE_TLAS,
                     root.ref, node_count,
                     cursor - VENTUS_AS_HEADER_SIZE, 0, instance_count,
                     &root.bounds);
   VkResult validation = vtvk_validate_built_as(
      device, dst, cursor, vtvk_target_as_version);
   if (validation != VK_SUCCESS)
      return validation;
   if (vtvk_env_enabled("VENTUS_VK_DEBUG_AS")) {
      mesa_logi("ventus-vk: TLAS instance_count=%u node_count=%u root=0x%x",
                instance_count, node_count, root.ref);
      vtvk_log_aabb("TLAS root", &root.bounds);
   }
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
   if (as->type != VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR &&
       as->type != info->type)
      return vk_error(device, VK_ERROR_UNKNOWN);

   if (info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
      struct vtvk_acceleration_structure *src =
         (struct vtvk_acceleration_structure *)
            vtvk_handle_to_ptr((uint64_t)info->srcAccelerationStructure);
      if (!src || !src->built ||
          !(src->build_flags &
            VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) ||
          src->build_flags != info->flags ||
          src->built_type != info->type)
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   } else if (info->mode != VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) {
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   }

   as->data = vtvk_buffer_host_ptr(as->buffer, as->offset, as->size);
   if (!as->data)
      return vk_error(device, VK_ERROR_UNKNOWN);

   const VkAccelerationStructureBuildRangeInfoKHR *range =
      ranges ? ranges[0] : NULL;

   VkResult result;
   if (info->type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR)
      result = vtvk_build_blas(device, as, info, range);
   else if (info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR)
      result = vtvk_build_tlas(device, as, info, range);
   else
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   if (result != VK_SUCCESS)
      return result;

   if (!as->snapshot || as->snapshot_size < as->size)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   memcpy(as->snapshot, as->data, as->size);
   as->build_flags = info->flags;
   as->built_type = info->type;
   as->built = true;
   if (vtvk_env_enabled("VENTUS_VK_DEBUG_AS") &&
       info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
      uint32_t root_ref =
         vtvk_load_u32(as->snapshot, VENTUS_AS_HEADER_ROOT_NODE_REF);
      uint32_t root_offset = ventus_bvh_node_ref_offset(root_ref);
      mesa_logi("ventus-vk: TLAS snapshot=%p size=%" PRIu64
                " root=0x%x root_blas=0x%" PRIx64,
                (void *)as->snapshot, (uint64_t)as->snapshot_size, root_ref,
                root_offset < as->snapshot_size ?
                   vtvk_load_u64(as->snapshot + root_offset,
                                 VENTUS_INSTANCE_BLAS_ADDR_LO) : 0);
   }
   return VK_SUCCESS;
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
   as->snapshot = vtvk_alloc_as_snapshot(as->size);
   if (!as->snapshot) {
      vk_free(alloc, as);
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   as->snapshot_size = as->size;
   VkDeviceAddress buffer_address = vtvk_buffer_get_device_address(as->buffer);
   if (!buffer_address || as->offset > UINT64_MAX - buffer_address) {
      free(as->snapshot);
      vk_free(alloc, as);
      return vk_error(device, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
   }
   as->address = buffer_address + as->offset;
   if (as->address % VTVK_AS_DEVICE_ADDRESS_ALIGNMENT) {
      free(as->snapshot);
      vk_free(alloc, as);
      return vk_error(device, VK_ERROR_VALIDATION_FAILED_EXT);
   }

   *pAccelerationStructure =
      (VkAccelerationStructureKHR)vtvk_ptr_to_handle(as);
   vtvk_as_registry.push_back(as);
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
   struct vtvk_acceleration_structure *as =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)accelerationStructure);
   vtvk_as_registry.erase(std::remove(vtvk_as_registry.begin(),
                                      vtvk_as_registry.end(), as),
                          vtvk_as_registry.end());
   if (as)
      free(as->snapshot);
   vk_free(alloc, as);
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
   uint64_t primitive_count = 0;
   uint64_t primitive_bytes = 0;
   if (pBuildInfo &&
       pBuildInfo->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
      primitive_count =
         pMaxPrimitiveCounts && pBuildInfo->geometryCount ?
         pMaxPrimitiveCounts[0] : 1;
      primitive_bytes = primitive_count * VENTUS_AS_INSTANCE_NODE_SIZE;
   } else if (pBuildInfo && pBuildInfo->geometryCount) {
      for (uint32_t i = 0; i < pBuildInfo->geometryCount; i++) {
         const VkAccelerationStructureGeometryKHR *geom =
            vtvk_build_geometry(pBuildInfo, i);
         uint64_t geometry_primitive_count =
            pMaxPrimitiveCounts ? pMaxPrimitiveCounts[i] : 1;
         const uint64_t leaf_size =
            geom && geom->geometryType == VK_GEOMETRY_TYPE_AABBS_KHR ?
            VENTUS_AS_AABB_NODE_SIZE : VENTUS_AS_TRIANGLE_NODE_SIZE;
         primitive_count += geometry_primitive_count;
         primitive_bytes += geometry_primitive_count * leaf_size;
      }
   }
   uint64_t internal_count = vtvk_internal_box_count(primitive_count);

   pSizeInfo->accelerationStructureSize = vtvk_as_required_size(
      vtvk_target_as_version, primitive_bytes, internal_count) + 256;
   pSizeInfo->updateScratchSize = 2048 + primitive_bytes;
   pSizeInfo->buildScratchSize = 4096 + primitive_bytes;
}

struct vtvk_recorded_as_build_command {
   struct vtvk_recorded_command base;
   bool indirect;
   uint32_t info_count;
   uint32_t total_geometry_count;
   VkAccelerationStructureBuildGeometryInfoKHR *infos;
   VkAccelerationStructureGeometryKHR *geometries;
   uint32_t *geometry_offsets;
   VkAccelerationStructureBuildRangeInfoKHR *ranges;
   VkDeviceAddress *indirect_addresses;
   uint32_t *indirect_strides;
   uint32_t *max_primitive_counts;
};

static void
vtvk_destroy_recorded_as_build_command(
   struct vtvk_cmd_buffer *cmd,
   struct vtvk_recorded_command *command)
{
   struct vtvk_recorded_as_build_command *build =
      (struct vtvk_recorded_as_build_command *)command;
   vk_free(&cmd->vk.pool->alloc, build->infos);
   vk_free(&cmd->vk.pool->alloc, build->geometries);
   vk_free(&cmd->vk.pool->alloc, build->geometry_offsets);
   vk_free(&cmd->vk.pool->alloc, build->ranges);
   vk_free(&cmd->vk.pool->alloc, build->indirect_addresses);
   vk_free(&cmd->vk.pool->alloc, build->indirect_strides);
   vk_free(&cmd->vk.pool->alloc, build->max_primitive_counts);
}

static VkResult
vtvk_execute_recorded_as_build_command(
   struct vtvk_cmd_buffer *cmd,
   const struct vtvk_recorded_command *command)
{
   const struct vtvk_recorded_as_build_command *build =
      (const struct vtvk_recorded_as_build_command *)command;
   struct vtvk_device *device =
      container_of(cmd->vk.base.device, struct vtvk_device, vk);

   if (!build->indirect) {
      for (uint32_t i = 0; i < build->info_count; i++) {
         const uint32_t offset = build->geometry_offsets[i];
         const VkAccelerationStructureBuildRangeInfoKHR *ranges =
            build->infos[i].geometryCount ? build->ranges + offset : nullptr;
         VkResult result =
            vtvk_build_as(device, &build->infos[i],
                          ranges ? &ranges : nullptr);
         if (result != VK_SUCCESS) {
            mesa_loge("ventus-vk: AS build failed for Ventus BVH ABI v%u "
                      "object=%u (%d)", vtvk_target_as_version, i, result);
            return result;
         }
      }
      mesa_logi("ventus-vk: queue submit AS build wrote Ventus BVH ABI v%u "
                "objects=%u", vtvk_target_as_version, build->info_count);
      return VK_SUCCESS;
   }

   std::vector<std::vector<VkAccelerationStructureBuildRangeInfoKHR>>
      resolved_ranges(build->info_count);
   for (uint32_t i = 0; i < build->info_count; i++) {
      const uint32_t geometry_count = build->infos[i].geometryCount;
      const VkDeviceAddress address = build->indirect_addresses[i];
      const uint32_t stride = build->indirect_strides[i];
      if ((address & 3) || (stride & 3)) {
         mesa_loge("ventus-vk: indirect AS build address/stride is not "
                   "4-byte aligned: address=0x%" PRIx64 " stride=%u",
                   (uint64_t)address, stride);
         return VK_ERROR_VALIDATION_FAILED_EXT;
      }
      if (!geometry_count)
         continue;

      const VkDeviceSize specified_span =
         VkDeviceSize(geometry_count) * stride;
      const VkDeviceSize loaded_span =
         VkDeviceSize(geometry_count - 1) * stride +
         sizeof(VkAccelerationStructureBuildRangeInfoKHR);
      const VkDeviceSize span = std::max(specified_span, loaded_span);
      const uint8_t *range_bytes =
         (const uint8_t *)vtvk_buffer_host_ptr_from_addr_with_usage(
            address, span, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
      if (!range_bytes) {
         mesa_loge("ventus-vk: indirect AS build %u does not name a complete "
                   "INDIRECT_BUFFER range: address=0x%" PRIx64
                   " span=%" PRIu64,
                   i, (uint64_t)address, (uint64_t)span);
         return VK_ERROR_VALIDATION_FAILED_EXT;
      }

      resolved_ranges[i].resize(geometry_count);
      const uint32_t max_offset = build->geometry_offsets[i];
      for (uint32_t j = 0; j < geometry_count; j++) {
         memcpy(&resolved_ranges[i][j],
                range_bytes + VkDeviceSize(j) * stride,
                sizeof(resolved_ranges[i][j]));
         if (resolved_ranges[i][j].primitiveCount >
             build->max_primitive_counts[max_offset + j]) {
            mesa_loge("ventus-vk: indirect AS build %u geometry %u "
                      "primitiveCount=%u exceeds maximum=%u",
                      i, j, resolved_ranges[i][j].primitiveCount,
                      build->max_primitive_counts[max_offset + j]);
            return VK_ERROR_VALIDATION_FAILED_EXT;
         }
      }
   }

   for (uint32_t i = 0; i < build->info_count; i++) {
      const VkAccelerationStructureBuildRangeInfoKHR *ranges =
         resolved_ranges[i].empty() ? nullptr : resolved_ranges[i].data();
      VkResult result =
         vtvk_build_as(device, &build->infos[i],
                       ranges ? &ranges : nullptr);
      if (result != VK_SUCCESS) {
         mesa_loge("ventus-vk: indirect AS build failed for Ventus BVH "
                   "ABI v%u object=%u (%d)", vtvk_target_as_version, i,
                   result);
         return result;
      }
   }

   mesa_logi("ventus-vk: queue submit indirect AS build wrote Ventus BVH "
             "ABI v%u objects=%u geometries=%u",
             vtvk_target_as_version, build->info_count,
             build->total_geometry_count);
   return VK_SUCCESS;
}

static struct vtvk_recorded_as_build_command *
vtvk_record_as_build_command(
   VkCommandBuffer command_buffer,
   uint32_t info_count,
   const VkAccelerationStructureBuildGeometryInfoKHR *infos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *range_infos,
   const VkDeviceAddress *indirect_addresses,
   const uint32_t *indirect_strides,
   const uint32_t *const *max_primitive_counts)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, command_buffer);
   struct vtvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct vtvk_cmd_buffer, vk);
   const bool indirect = indirect_addresses != nullptr;
   if (!info_count || !infos ||
       (!indirect && !range_infos) ||
       (indirect && (!indirect_strides || !max_primitive_counts))) {
      vk_command_buffer_set_error(vk_cmd, VK_ERROR_VALIDATION_FAILED_EXT);
      return nullptr;
   }

   uint64_t geometry_count64 = 0;
   for (uint32_t i = 0; i < info_count; i++)
      geometry_count64 += infos[i].geometryCount;
   if (geometry_count64 > UINT32_MAX) {
      vk_command_buffer_set_error(vk_cmd, VK_ERROR_OUT_OF_HOST_MEMORY);
      return nullptr;
   }
   const uint32_t geometry_count = (uint32_t)geometry_count64;

   struct vtvk_recorded_as_build_command *build =
      (struct vtvk_recorded_as_build_command *)
         vtvk_cmd_buffer_alloc_command(
            cmd, sizeof(*build), vtvk_execute_recorded_as_build_command,
            vtvk_destroy_recorded_as_build_command);
   if (!build)
      return nullptr;

   const VkAllocationCallbacks *alloc = &cmd->vk.pool->alloc;
   build->indirect = indirect;
   build->info_count = info_count;
   build->total_geometry_count = geometry_count;
   build->infos = (VkAccelerationStructureBuildGeometryInfoKHR *)
      vk_zalloc(alloc, sizeof(*build->infos) * info_count, 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   build->geometry_offsets = (uint32_t *)
      vk_zalloc(alloc, sizeof(*build->geometry_offsets) * info_count, 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (geometry_count) {
      build->geometries = (VkAccelerationStructureGeometryKHR *)
         vk_zalloc(alloc, sizeof(*build->geometries) * geometry_count, 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (indirect) {
         build->max_primitive_counts = (uint32_t *)
            vk_zalloc(alloc,
                      sizeof(*build->max_primitive_counts) * geometry_count, 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      } else {
         build->ranges = (VkAccelerationStructureBuildRangeInfoKHR *)
            vk_zalloc(alloc, sizeof(*build->ranges) * geometry_count, 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      }
   }
   if (indirect) {
      build->indirect_addresses = (VkDeviceAddress *)
         vk_zalloc(alloc, sizeof(*build->indirect_addresses) * info_count, 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      build->indirect_strides = (uint32_t *)
         vk_zalloc(alloc, sizeof(*build->indirect_strides) * info_count, 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   }

   if (!build->infos || !build->geometry_offsets ||
       (geometry_count &&
        (!build->geometries ||
         (indirect ? !build->max_primitive_counts : !build->ranges))) ||
       (indirect &&
        (!build->indirect_addresses || !build->indirect_strides))) {
      vk_command_buffer_set_error(vk_cmd, VK_ERROR_OUT_OF_HOST_MEMORY);
      return nullptr;
   }

   uint32_t geometry_offset = 0;
   for (uint32_t i = 0; i < info_count; i++) {
      build->geometry_offsets[i] = geometry_offset;
      build->infos[i] = infos[i];
      build->infos[i].pNext = nullptr;
      build->infos[i].pGeometries =
         infos[i].geometryCount ? build->geometries + geometry_offset : nullptr;
      build->infos[i].ppGeometries = nullptr;

      if (indirect) {
         build->indirect_addresses[i] = indirect_addresses[i];
         build->indirect_strides[i] = indirect_strides[i];
      }
      for (uint32_t j = 0; j < infos[i].geometryCount; j++) {
         const VkAccelerationStructureGeometryKHR *geometry =
            (infos[i].pGeometries || infos[i].ppGeometries) ?
            vtvk_build_geometry(&infos[i], j) : nullptr;
         if (!geometry ||
             (!indirect && !range_infos[i]) ||
             (indirect && !max_primitive_counts[i])) {
            vk_command_buffer_set_error(vk_cmd,
                                        VK_ERROR_VALIDATION_FAILED_EXT);
            return nullptr;
         }
         build->geometries[geometry_offset + j] = *geometry;
         build->geometries[geometry_offset + j].pNext = nullptr;
         if (indirect) {
            build->max_primitive_counts[geometry_offset + j] =
               max_primitive_counts[i][j];
         } else {
            build->ranges[geometry_offset + j] = range_infos[i][j];
         }
      }
      geometry_offset += infos[i].geometryCount;
   }
   return build;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   vtvk_record_as_build_command(commandBuffer, infoCount, pInfos,
                                ppBuildRangeInfos, nullptr, nullptr, nullptr);
}

static VkResult
vtvk_acceleration_structure_used_size(
   struct vtvk_device *device,
   const struct vtvk_acceleration_structure *as,
   VkDeviceSize *size_out)
{
   if (!as || !as->built || !as->snapshot ||
       as->snapshot_size < VENTUS_AS_HEADER_SIZE ||
       vtvk_load_u32(as->snapshot, VENTUS_AS_HEADER_MAGIC) !=
          VENTUS_AS_MAGIC)
      return vk_error(device, VK_ERROR_UNKNOWN);

   const uint32_t node_region_size =
      vtvk_load_u32(as->snapshot, VENTUS_AS_HEADER_NODE_REGION_SIZE);
   if (node_region_size > as->snapshot_size - VENTUS_AS_HEADER_SIZE)
      return vk_error(device, VK_ERROR_UNKNOWN);

   *size_out = VENTUS_AS_HEADER_SIZE + VkDeviceSize(node_region_size);
   return VK_SUCCESS;
}

static VkResult
vtvk_acceleration_structure_property(
   struct vtvk_device *device,
   const struct vtvk_acceleration_structure *as,
   VkQueryType query_type,
   uint64_t *value_out)
{
   if (!as || !as->built)
      return vk_error(device, VK_ERROR_UNKNOWN);

   if (query_type !=
       VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   if (!(as->build_flags &
         VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR))
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   VkDeviceSize used_size;
   VkResult result =
      vtvk_acceleration_structure_used_size(device, as, &used_size);
   if (result == VK_SUCCESS)
      *value_out = used_size;
   return result;
}

static VkResult
vtvk_copy_acceleration_structure(
   struct vtvk_device *device,
   const VkCopyAccelerationStructureInfoKHR *info)
{
   if (!info ||
       (info->mode != VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR &&
        info->mode != VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR))
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   struct vtvk_acceleration_structure *src =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)info->src);
   struct vtvk_acceleration_structure *dst =
      (struct vtvk_acceleration_structure *)
         vtvk_handle_to_ptr((uint64_t)info->dst);
   if (!src || !dst || src == dst || src->type != dst->type ||
       !src->built || !src->snapshot || !src->snapshot_size)
      return vk_error(device, VK_ERROR_UNKNOWN);

   VkDeviceSize copy_size = src->snapshot_size;
   if (info->mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR) {
      if (!(src->build_flags &
            VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR))
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      VkResult result =
         vtvk_acceleration_structure_used_size(device, src, &copy_size);
      if (result != VK_SUCCESS)
         return result;
   }
   if (dst->size < copy_size)
      return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   void *dst_data = vtvk_buffer_host_ptr(dst->buffer, dst->offset, dst->size);
   if (!dst_data || !dst->snapshot ||
       dst->snapshot_size < copy_size)
      return vk_error(device, VK_ERROR_UNKNOWN);

   memset(dst_data, 0, dst->size);
   memcpy(dst_data, src->snapshot, copy_size);
   memset(dst->snapshot, 0, dst->snapshot_size);
   memcpy(dst->snapshot, src->snapshot, copy_size);
   dst->data = dst_data;
   dst->build_flags = src->build_flags;
   dst->built_type = src->built_type;
   dst->built = true;
   return VK_SUCCESS;
}

struct vtvk_recorded_as_copy_command {
   struct vtvk_recorded_command base;
   VkCopyAccelerationStructureInfoKHR info;
};

static VkResult
vtvk_execute_recorded_as_copy_command(
   struct vtvk_cmd_buffer *cmd,
   const struct vtvk_recorded_command *command)
{
   const struct vtvk_recorded_as_copy_command *copy =
      (const struct vtvk_recorded_as_copy_command *)command;
   struct vtvk_device *device =
      container_of(cmd->vk.base.device, struct vtvk_device, vk);
   VkResult result = vtvk_copy_acceleration_structure(device, &copy->info);
   if (result != VK_SUCCESS)
      mesa_loge("ventus-vk: AS copy failed at queue submit (%d)", result);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct vtvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct vtvk_cmd_buffer, vk);
   if (!pInfo) {
      vk_command_buffer_set_error(vk_cmd, VK_ERROR_VALIDATION_FAILED_EXT);
      return;
   }
   struct vtvk_recorded_as_copy_command *copy =
      (struct vtvk_recorded_as_copy_command *)
         vtvk_cmd_buffer_alloc_command(
            cmd, sizeof(*copy), vtvk_execute_recorded_as_copy_command,
            nullptr);
   if (!copy)
      return;
   copy->info = *pInfo;
   copy->info.pNext = nullptr;
}

struct vtvk_recorded_as_property_command {
   struct vtvk_recorded_command base;
   uint32_t acceleration_structure_count;
   VkAccelerationStructureKHR *acceleration_structures;
   VkQueryType query_type;
   VkQueryPool query_pool;
   uint32_t first_query;
};

static void
vtvk_destroy_recorded_as_property_command(
   struct vtvk_cmd_buffer *cmd,
   struct vtvk_recorded_command *command)
{
   struct vtvk_recorded_as_property_command *property =
      (struct vtvk_recorded_as_property_command *)command;
   vk_free(&cmd->vk.pool->alloc, property->acceleration_structures);
}

static VkResult
vtvk_execute_recorded_as_property_command(
   struct vtvk_cmd_buffer *cmd,
   const struct vtvk_recorded_command *command)
{
   const struct vtvk_recorded_as_property_command *property =
      (const struct vtvk_recorded_as_property_command *)command;
   VK_FROM_HANDLE(vtvk_query_pool, pool, property->query_pool);
   struct vtvk_device *device =
      container_of(cmd->vk.base.device, struct vtvk_device, vk);

   if (!pool || pool->vk.query_type != property->query_type ||
       property->first_query > pool->vk.query_count ||
       property->acceleration_structure_count >
          pool->vk.query_count - property->first_query)
      return VK_ERROR_VALIDATION_FAILED_EXT;

   std::vector<uint64_t> values(property->acceleration_structure_count);
   for (uint32_t i = 0; i < property->acceleration_structure_count; i++) {
      if (pool->queries[property->first_query + i].available)
         return VK_ERROR_VALIDATION_FAILED_EXT;
      const struct vtvk_acceleration_structure *as =
         (const struct vtvk_acceleration_structure *)vtvk_handle_to_ptr(
            (uint64_t)property->acceleration_structures[i]);
      VkResult result = vtvk_acceleration_structure_property(
         device, as, property->query_type, &values[i]);
      if (result != VK_SUCCESS)
         return result;
   }

   for (uint32_t i = 0; i < property->acceleration_structure_count; i++) {
      pool->queries[property->first_query + i].value = values[i];
      pool->queries[property->first_query + i].available = true;
   }
   return VK_SUCCESS;
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
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct vtvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct vtvk_cmd_buffer, vk);
   if (accelerationStructureCount && !pAccelerationStructures) {
      vk_command_buffer_set_error(vk_cmd, VK_ERROR_VALIDATION_FAILED_EXT);
      return;
   }

   struct vtvk_recorded_as_property_command *property =
      (struct vtvk_recorded_as_property_command *)
         vtvk_cmd_buffer_alloc_command(
            cmd, sizeof(*property),
            vtvk_execute_recorded_as_property_command,
            vtvk_destroy_recorded_as_property_command);
   if (!property)
      return;
   property->acceleration_structure_count = accelerationStructureCount;
   property->query_type = queryType;
   property->query_pool = queryPool;
   property->first_query = firstQuery;
   if (accelerationStructureCount) {
      property->acceleration_structures = (VkAccelerationStructureKHR *)
         vk_alloc(&cmd->vk.pool->alloc,
                  sizeof(*property->acceleration_structures) *
                     accelerationStructureCount,
                  8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!property->acceleration_structures) {
         vk_command_buffer_set_error(vk_cmd, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }
      memcpy(property->acceleration_structures, pAccelerationStructures,
             sizeof(*property->acceleration_structures) *
                accelerationStructureCount);
   }
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
   vtvk_record_as_build_command(
      commandBuffer, infoCount, pInfos, nullptr, pIndirectDeviceAddresses,
      pIndirectStrides, ppMaxPrimitiveCounts);
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
