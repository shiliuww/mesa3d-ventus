/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "ventus_rt_bvh_v2.h"

#include <string.h>

#include <unordered_set>
#include <vector>

#include "ventus_rt_bvh.h"

namespace {

struct node_work {
   uint32_t ref;
   uint32_t parent_ref;
   uint8_t parent_child_index;
   bool root;
};

static uint16_t
load_u16(const uint8_t *data, size_t offset)
{
   return uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
}

static uint32_t
load_u32(const uint8_t *data, size_t offset)
{
   return uint32_t(data[offset]) | (uint32_t(data[offset + 1]) << 8) |
          (uint32_t(data[offset + 2]) << 16) |
          (uint32_t(data[offset + 3]) << 24);
}

static bool
fail(struct ventus_vtas_v2_validation_result *result,
     enum ventus_vtas_v2_validation_error error, uint32_t node_ref,
     uint32_t detail)
{
   if (result) {
      result->error = error;
      result->node_ref = node_ref;
      result->detail = detail;
   }
   return false;
}

static size_t
node_size(uint32_t type)
{
   switch (type) {
   case VENTUS_BVH_NODE_BOX4:
      return VENTUS_AS_BOX4_NODE_SIZE;
   case VENTUS_BVH_NODE_TRIANGLE:
      return VENTUS_AS_TRIANGLE_NODE_SIZE;
   case VENTUS_BVH_NODE_INSTANCE:
      return VENTUS_AS_INSTANCE_NODE_SIZE;
   case VENTUS_BVH_NODE_AABB:
      return VENTUS_AS_AABB_NODE_SIZE;
   default:
      return 0;
   }
}

} // namespace

extern "C" bool
ventus_vtas_v2_validate(const void *image, size_t image_size,
                        struct ventus_vtas_v2_validation_result *result)
{
   if (result)
      memset(result, 0, sizeof(*result));
   if (!image || image_size < VENTUS_AS_HEADER_SIZE)
      return fail(result, VENTUS_VTAS_V2_TRUNCATED, 0, 0);

   const uint8_t *data = static_cast<const uint8_t *>(image);
   if (load_u32(data, VENTUS_AS_HEADER_MAGIC) != VENTUS_AS_MAGIC)
      return fail(result, VENTUS_VTAS_V2_BAD_MAGIC, 0, 0);
   if (load_u16(data, VENTUS_AS_HEADER_VERSION) != VENTUS_AS_VERSION_V2)
      return fail(result, VENTUS_VTAS_V2_BAD_VERSION, 0, 0);
   if (load_u16(data, VENTUS_AS_HEADER_HEADER_SIZE) != VENTUS_AS_HEADER_SIZE)
      return fail(result, VENTUS_VTAS_V2_BAD_HEADER, 0,
                  VENTUS_AS_HEADER_HEADER_SIZE);

   const uint32_t as_type = load_u32(data, VENTUS_AS_HEADER_TYPE);
   if (as_type != VENTUS_AS_TYPE_BLAS && as_type != VENTUS_AS_TYPE_TLAS)
      return fail(result, VENTUS_VTAS_V2_BAD_HEADER, 0,
                  VENTUS_AS_HEADER_TYPE);
   if (load_u32(data, VENTUS_AS_HEADER_NODE_REGION_OFFSET) !=
       VENTUS_AS_HEADER_SIZE)
      return fail(result, VENTUS_VTAS_V2_BAD_HEADER, 0,
                  VENTUS_AS_HEADER_NODE_REGION_OFFSET);

   const uint32_t region_size =
      load_u32(data, VENTUS_AS_HEADER_NODE_REGION_SIZE);
   if (region_size > image_size - VENTUS_AS_HEADER_SIZE)
      return fail(result, VENTUS_VTAS_V2_TRUNCATED, 0, region_size);
   const size_t used_size = VENTUS_AS_HEADER_SIZE + size_t(region_size);
   const uint32_t expected_count =
      load_u32(data, VENTUS_AS_HEADER_NODE_COUNT);
   const uint32_t root_ref =
      load_u32(data, VENTUS_AS_HEADER_ROOT_NODE_REF);
   if (root_ref == VENTUS_AS_INVALID_NODE) {
      if (expected_count != 0)
         return fail(result, VENTUS_VTAS_V2_NODE_COUNT_MISMATCH, root_ref,
                     expected_count);
      return true;
   }
   if (expected_count == 0)
      return fail(result, VENTUS_VTAS_V2_BAD_ROOT, root_ref, 0);

   std::vector<node_work> work;
   work.push_back({root_ref, VENTUS_AS_INVALID_NODE, 0xff, true});
   std::unordered_set<uint32_t> visited;
   uint32_t reachable_count = 0;

   while (!work.empty()) {
      const node_work item = work.back();
      work.pop_back();
      if (!visited.insert(item.ref).second)
         return fail(result, VENTUS_VTAS_V2_DUPLICATE_OR_CYCLE, item.ref, 0);

      const uint32_t type = ventus_bvh_node_ref_type(item.ref);
      const uint32_t offset = ventus_bvh_node_ref_offset(item.ref);
      const size_t size = node_size(type);
      if (!size)
         return fail(result, VENTUS_VTAS_V2_BAD_NODE_TYPE, item.ref, type);
      if (offset < VENTUS_AS_HEADER_SIZE || offset > used_size ||
          size > used_size - offset)
         return fail(result, VENTUS_VTAS_V2_BAD_NODE_REF, item.ref, offset);
      if (type == VENTUS_BVH_NODE_BOX4 &&
          (offset & (VENTUS_AS_BOX4_NODE_SIZE - 1)) != 0)
         return fail(result, VENTUS_VTAS_V2_BAD_NODE_REF, item.ref, offset);

      const uint8_t *node = data + offset;
      reachable_count++;
      if (type == VENTUS_BVH_NODE_BOX4) {
         const uint32_t parent_ref =
            load_u32(node, VENTUS_BOX4_V2_PARENT_NODE_REF);
         const uint8_t parent_child =
            node[VENTUS_BOX4_V2_PARENT_CHILD_INDEX];
         const uint8_t flags = node[VENTUS_BOX4_V2_NODE_FLAGS];
         if (flags & ~VENTUS_BOX4_V2_NODE_FLAG_ROOT)
            return fail(result, VENTUS_VTAS_V2_NONZERO_RESERVED, item.ref,
                        flags);
         if (item.root) {
            if (parent_ref != VENTUS_AS_INVALID_NODE || parent_child != 0xff ||
                flags != VENTUS_BOX4_V2_NODE_FLAG_ROOT)
               return fail(result, VENTUS_VTAS_V2_BAD_ROOT, item.ref, 0);
         } else if (parent_ref != item.parent_ref ||
                    parent_child != item.parent_child_index || flags != 0) {
            return fail(result, VENTUS_VTAS_V2_BAD_PARENT, item.ref,
                        parent_ref);
         }
         for (uint32_t i = 0; i < 4; i++) {
            if (node[VENTUS_BOX4_V2_CHILD_FLAGS + i] != 0)
               return fail(result, VENTUS_VTAS_V2_NONZERO_RESERVED, item.ref,
                           VENTUS_BOX4_V2_CHILD_FLAGS + i);
         }
         for (uint32_t i = 0; i < 6; i++) {
            if (node[VENTUS_BOX4_V2_RESERVED + i] != 0)
               return fail(result, VENTUS_VTAS_V2_NONZERO_RESERVED, item.ref,
                           VENTUS_BOX4_V2_RESERVED + i);
         }
         for (uint32_t i = 0; i < 4; i++) {
            const uint32_t child =
               load_u32(node, VENTUS_BOX4_CHILD_REF + i * 4);
            if (child != VENTUS_AS_INVALID_NODE)
               work.push_back({child, item.ref, uint8_t(i), false});
         }
         continue;
      }

      if (type == VENTUS_BVH_NODE_INSTANCE) {
         if (as_type != VENTUS_AS_TYPE_TLAS)
            return fail(result, VENTUS_VTAS_V2_BAD_NODE_TYPE, item.ref, type);
         const uint32_t edge =
            load_u32(node, VENTUS_INSTANCE_V2_PARENT_EDGE);
         if (item.root) {
            if (edge != VENTUS_INSTANCE_V2_ROOT_PARENT_EDGE)
               return fail(result, VENTUS_VTAS_V2_BAD_ROOT, item.ref, edge);
         } else {
            const uint32_t parent_offset =
               ventus_bvh_node_ref_offset(item.parent_ref);
            if ((edge & VENTUS_INSTANCE_V2_PARENT_RESERVED_MASK) != 0 ||
                (edge & VENTUS_INSTANCE_V2_PARENT_OFFSET_MASK) !=
                   parent_offset ||
                (edge & VENTUS_INSTANCE_V2_PARENT_CHILD_INDEX_MASK) !=
                   item.parent_child_index)
               return fail(result, VENTUS_VTAS_V2_BAD_PARENT, item.ref, edge);
         }
         continue;
      }

      if (item.root && as_type == VENTUS_AS_TYPE_TLAS)
         return fail(result, VENTUS_VTAS_V2_BAD_ROOT, item.ref, type);
      if (!item.root) {
         const uint32_t parent_offset =
            ventus_bvh_node_ref_offset(item.parent_ref);
         const uint8_t *parent = data + parent_offset;
         if (load_u32(parent, VENTUS_BOX4_CHILD_REF +
                                  item.parent_child_index * 4) != item.ref)
            return fail(result, VENTUS_VTAS_V2_BAD_BACKLINK, item.ref,
                        item.parent_ref);
      }
      if (as_type != VENTUS_AS_TYPE_BLAS)
         return fail(result, VENTUS_VTAS_V2_BAD_NODE_TYPE, item.ref, type);
   }

   if (reachable_count != expected_count)
      return fail(result, VENTUS_VTAS_V2_NODE_COUNT_MISMATCH, root_ref,
                  reachable_count);
   return true;
}

extern "C" const char *
ventus_vtas_v2_validation_error_string(
   enum ventus_vtas_v2_validation_error error)
{
   switch (error) {
   case VENTUS_VTAS_V2_VALID: return "valid";
   case VENTUS_VTAS_V2_TRUNCATED: return "truncated";
   case VENTUS_VTAS_V2_BAD_MAGIC: return "bad magic";
   case VENTUS_VTAS_V2_BAD_VERSION: return "bad version";
   case VENTUS_VTAS_V2_BAD_HEADER: return "bad header";
   case VENTUS_VTAS_V2_BAD_ROOT: return "bad root";
   case VENTUS_VTAS_V2_BAD_NODE_REF: return "bad node ref";
   case VENTUS_VTAS_V2_BAD_NODE_TYPE: return "bad node type";
   case VENTUS_VTAS_V2_BAD_PARENT: return "bad parent";
   case VENTUS_VTAS_V2_BAD_BACKLINK: return "bad backlink";
   case VENTUS_VTAS_V2_DUPLICATE_OR_CYCLE: return "duplicate or cycle";
   case VENTUS_VTAS_V2_NONZERO_RESERVED: return "nonzero reserved";
   case VENTUS_VTAS_V2_NODE_COUNT_MISMATCH: return "node count mismatch";
   }
   return "unknown";
}
