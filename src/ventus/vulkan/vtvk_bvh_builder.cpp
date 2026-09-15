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

/* Recover wide nodes from the safe binary fallback. Expand the largest-area
 * internal child first, but reserve enough stack for every frontier subtree.
 * Recursion receives the residual budget, so local widening cannot overflow
 * an ancestor. This changes grouping only, never primitive membership. */
void widen_node(const vtvk_bvh_topology &binary, unsigned index,
                unsigned budget, vtvk_bvh_topology *wide)
{
   const auto &src = binary.nodes[index];
   if (!src.child_count) {
      wide->nodes.push_back(src);
      return;
   }
   std::vector<unsigned> frontier(src.children, src.children + src.child_count);
   while (frontier.size() < 4) {
      int best = -1;
      double best_area = -1;
      for (size_t i = 0; i < frontier.size(); ++i) {
         const auto &candidate = binary.nodes[frontier[i]];
         if (!candidate.child_count)
            continue;
         unsigned max_stack = 0;
         for (size_t j = 0; j < frontier.size(); ++j) {
            if (i == j) {
               for (unsigned c = 0; c < candidate.child_count; ++c)
                  max_stack = std::max(max_stack,
                     binary.nodes[candidate.children[c]].stack_required);
            } else {
               max_stack = std::max(max_stack, binary.nodes[frontier[j]].stack_required);
            }
         }
         const unsigned new_count = frontier.size() - 1 + candidate.child_count;
         if (new_count > 4 || new_count - 1 + max_stack > budget)
            continue;
         const auto &b = candidate.bounds;
         double x = double(b.max[0]) - b.min[0];
         double y = double(b.max[1]) - b.min[1];
         double z = double(b.max[2]) - b.min[2];
         double area = x*y + y*z + z*x;
         if (area > best_area) {
            best = i;
            best_area = area;
         }
      }
      if (best < 0)
         break;
      const auto &candidate = binary.nodes[frontier[best]];
      frontier.erase(frontier.begin() + best);
      frontier.insert(frontier.begin() + best, candidate.children,
                      candidate.children + candidate.child_count);
   }
   vtvk_bvh_node node;
   node.bounds = src.bounds;
   node.child_count = frontier.size();
   for (unsigned i = 0; i < node.child_count; ++i) {
      widen_node(binary, frontier[i], budget - node.child_count + 1, wide);
      node.children[i] = wide->nodes.size() - 1;
      node.depth = std::max(node.depth, wide->nodes.back().depth + 1);
      node.stack_required = std::max(node.stack_required,
         node.child_count - 1 + wide->nodes.back().stack_required);
   }
   wide->fanout[node.child_count]++;
   wide->nodes.push_back(node);
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
       !stack_budget || stack_budget > 32) {
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
   /* Try a narrower SAH tree if deferred siblings exceed the hardware stack. */
   for (unsigned branching : {4u, 2u}) {
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
         if (branching == 2) {
            vtvk_bvh_topology wide;
            wide.branching = 2;
            wide.stack_repacked = true;
            widen_node(*tree, tree->nodes.size() - 1, stack_budget, &wide);
            *tree = std::move(wide);
         }
         success = true;
         break;
      }
      *error = "traversal stack bound exceeds budget, including binary fallback";
   }
   rtcReleaseDevice(device);
   if (!success)
      *tree = {};
   else
      error->clear();
   return success;
#endif
}
