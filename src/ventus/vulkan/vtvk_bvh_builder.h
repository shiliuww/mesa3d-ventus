#pragma once

#include <cstdint>
#include <string>
#include <vector>

/* Host-only topology. Leaves retain their caller-owned serialized reference. */
constexpr uint32_t VENTUS_BVH_TRAVERSAL_STACK_BUDGET = 64;
struct vtvk_aabb {
   float min[3];
   float max[3];
};

struct vtvk_build_node {
   uint32_t ref;
   vtvk_aabb bounds;
   /* Includes the root entry; instance leaves inherit their BLAS requirement. */
   uint32_t stack_required = 1;
};

struct vtvk_bvh_node {
   vtvk_aabb bounds;
   uint32_t children[4] = {};
   uint32_t child_count = 0;
   uint32_t leaf = 0;
   uint32_t stack_required = 1;
   uint32_t depth = 0;
};

struct vtvk_bvh_topology {
   /* Postorder: all child indices precede their parent, root is last. */
   std::vector<vtvk_bvh_node> nodes;
   uint32_t fanout[5] = {};
   uint32_t branching = 4;
   bool stack_repacked = false;
};

bool vtvk_embree_available();
const char *vtvk_embree_version();
bool vtvk_build_embree(const std::vector<vtvk_build_node> &leaves,
                      vtvk_bvh_topology *tree, std::string *error,
                      uint32_t stack_budget = VENTUS_BVH_TRAVERSAL_STACK_BUDGET);
