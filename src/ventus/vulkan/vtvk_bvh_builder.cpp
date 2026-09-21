#include "vtvk_bvh_builder.h"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef HAVE_VENTUS_EMBREE
#include <embree3/rtcore.h>
#include <embree3/rtcore_builder.h>
#include <atomic>
#include <new>

static_assert(RTC_VERSION_MAJOR == 3, "Ventus requires the Embree 3 builder ABI");

namespace {
struct temporary_node {
   temporary_node *children[4] = {};
   unsigned count = 0;
   unsigned leaf = 0;
};

struct build_context {
   std::atomic<bool> invalid{false};
};

void *create_node(RTCThreadLocalAllocator allocator, unsigned count, void *user)
{
   auto *ctx = static_cast<build_context *>(user);
   if (count < 2 || count > 4)
      ctx->invalid = true;
   return new (rtcThreadLocalAlloc(allocator, sizeof(temporary_node),
                                   alignof(temporary_node))) temporary_node;
}

void set_children(void *ptr, void **children, unsigned count, void *user)
{
   auto *node = static_cast<temporary_node *>(ptr);
   if (count < 2 || count > 4) {
      static_cast<build_context *>(user)->invalid = true;
      return;
   }
   node->count = count;
   for (unsigned i = 0; i < count; ++i)
      node->children[i] = static_cast<temporary_node *>(children[i]);
}

/* Recompute exact unions from the original bounds during postorder export. */
void set_bounds(void *, const RTCBounds **, unsigned, void *) {}

void *create_leaf(RTCThreadLocalAllocator allocator,
                  const RTCBuildPrimitive *primitives, size_t count, void *user)
{
   if (count != 1)
      static_cast<build_context *>(user)->invalid = true;
   auto *node = new (rtcThreadLocalAlloc(allocator, sizeof(temporary_node),
                                        alignof(temporary_node))) temporary_node;
   node->leaf = count ? primitives[0].primID : 0;
   return node;
}

bool export_node(const temporary_node *src,
                 const std::vector<vtvk_build_node> &leaves,
                 std::vector<bool> &seen, vtvk_bvh_topology *tree,
                 unsigned level)
{
   if (!src || level > 32 || tree->nodes.size() >= 2 * leaves.size() - 1)
      return false;
   vtvk_bvh_node node;
   node.child_count = src->count;
   if (!src->count) {
      if (src->leaf >= leaves.size() || seen[src->leaf])
         return false;
      seen[src->leaf] = true;
      node.leaf = src->leaf;
      node.bounds = leaves[src->leaf].bounds;
      node.stack_required = leaves[src->leaf].stack_required;
   } else {
      if (src->count < 2 || src->count > 4)
         return false;
      for (unsigned i = 0; i < src->count; ++i) {
         if (!export_node(src->children[i], leaves, seen, tree, level + 1))
            return false;
         node.children[i] = tree->nodes.size() - 1;
         const auto &child = tree->nodes.back();
         if (!i)
            node.bounds = child.bounds;
         for (unsigned axis = 0; axis < 3; ++axis) {
            node.bounds.min[axis] = std::min(node.bounds.min[axis], child.bounds.min[axis]);
            node.bounds.max[axis] = std::max(node.bounds.max[axis], child.bounds.max[axis]);
         }
         node.depth = std::max(node.depth, child.depth + 1);
         node.stack_required = std::max(node.stack_required,
                                        src->count - 1 + child.stack_required);
      }
      tree->fanout[src->count]++;
   }
   tree->nodes.push_back(node);
   return true;
}

} // namespace
#endif

bool vtvk_embree_available()
{
#ifdef HAVE_VENTUS_EMBREE
   return true;
#else
   return false;
#endif
}

const char *vtvk_embree_version()
{
#ifdef HAVE_VENTUS_EMBREE
   return RTC_VERSION_STRING;
#else
   return "unavailable";
#endif
}

bool vtvk_build_embree(const std::vector<vtvk_build_node> &leaves,
                      vtvk_bvh_topology *tree, std::string *error,
                      uint32_t stack_budget)
{
   *tree = {};
   error->clear();
#ifndef HAVE_VENTUS_EMBREE
   *error = "Embree 3 support was not compiled in";
   return false;
#else
   if (leaves.empty() || leaves.size() > UINT32_MAX / 2 ||
       !stack_budget || stack_budget > VENTUS_BVH_TRAVERSAL_STACK_BUDGET) {
      *error = "unsupported primitive count";
      return false;
   }
   std::vector<RTCBuildPrimitive> primitives(leaves.size());
   for (size_t i = 0; i < leaves.size(); ++i) {
      const auto &b = leaves[i].bounds;
      for (unsigned axis = 0; axis < 3; ++axis) {
         if (!std::isfinite(b.min[axis]) || !std::isfinite(b.max[axis]) ||
             b.min[axis] > b.max[axis]) {
            *error = "non-finite or inverted primitive bounds";
            return false;
         }
      }
      if (!leaves[i].stack_required || leaves[i].stack_required > stack_budget) {
         *error = "leaf/BLAS exceeds the traversal stack budget";
         return false;
      }
      primitives[i] = {b.min[0], b.min[1], b.min[2], 0,
                       b.max[0], b.max[1], b.max[2], unsigned(i)};
   }
   /* A fixed worker count also avoids callback scheduling affecting layout. */
   RTCDevice device = rtcNewDevice("threads=1,ignore_config_files=1");
   if (!device) {
      *error = "rtcNewDevice failed";
      return false;
   }
   bool success = false;
   /* Keep the Box4 build policy; overflow is an explicit build failure. */
   do {
      const unsigned branching = 4;
      *tree = {};
      tree->branching = branching;
      RTCBVH bvh = rtcNewBVH(device);
      if (!bvh) {
         *error = "rtcNewBVH failed";
         break;
      }
      build_context ctx;
      RTCBuildArguments args = rtcDefaultBuildArguments();
      args.buildQuality = RTC_BUILD_QUALITY_MEDIUM;
      args.maxBranchingFactor = branching;
      args.maxDepth = 32;
      args.minLeafSize = args.maxLeafSize = 1;
      args.bvh = bvh;
      args.primitives = primitives.data();
      args.primitiveCount = args.primitiveArrayCapacity = primitives.size();
      args.createNode = create_node;
      args.setNodeChildren = set_children;
      args.setNodeBounds = set_bounds;
      args.createLeaf = create_leaf;
      args.userPtr = &ctx;
      auto *root = static_cast<temporary_node *>(rtcBuildBVH(&args));
      const RTCError rtc_error = rtcGetDeviceError(device);
      std::vector<bool> seen(leaves.size(), false);
      bool valid = rtc_error == RTC_ERROR_NONE && !ctx.invalid &&
                   export_node(root, leaves, seen, tree, 0) &&
                   std::all_of(seen.begin(), seen.end(), [](bool v) { return v; });
      rtcReleaseBVH(bvh);
      if (!valid) {
         *error = "Embree failed or produced invalid topology";
         break;
      }
      if (tree->nodes.back().stack_required <= stack_budget) {
         success = true;
         break;
      }
      *error = "four-way BVH requires " +
         std::to_string(tree->nodes.back().stack_required) +
         " stack entries, budget is " + std::to_string(stack_budget);
   } while (false);
   rtcReleaseDevice(device);
   if (!success)
      *tree = {};
   else
      error->clear();
   return success;
#endif
}
