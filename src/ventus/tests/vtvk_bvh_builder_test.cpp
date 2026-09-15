#include "../vulkan/vtvk_bvh_builder.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define CHECK(x) do { if (!(x)) { \
   std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); \
   std::abort(); } } while (0)

static void check_tree(const std::vector<vtvk_build_node> &leaves,
                       const vtvk_bvh_topology &tree)
{
   CHECK(!tree.nodes.empty());
   CHECK(tree.nodes.size() <= 2 * leaves.size() - 1);
   std::vector<unsigned> seen(leaves.size());
   for (size_t i = 0; i < tree.nodes.size(); ++i) {
      const auto &n = tree.nodes[i];
      if (!n.child_count) {
         CHECK(n.leaf < leaves.size());
         CHECK(++seen[n.leaf] == 1);
         CHECK(n.stack_required == leaves[n.leaf].stack_required);
      } else {
         CHECK(n.child_count >= 2 && n.child_count <= 4);
         for (unsigned c = 0; c < n.child_count; ++c) {
            CHECK(n.children[c] < i);
            const auto &child = tree.nodes[n.children[c]];
            CHECK(n.stack_required >= n.child_count - 1 + child.stack_required);
            for (unsigned a = 0; a < 3; ++a) {
               CHECK(n.bounds.min[a] <= child.bounds.min[a]);
               CHECK(n.bounds.max[a] >= child.bounds.max[a]);
            }
         }
      }
   }
   for (unsigned count : seen)
      CHECK(count == 1);
   CHECK(tree.nodes.back().stack_required <= 32);
}

static bool intersects(const vtvk_aabb &b, const float origin[3], const float dir[3])
{
   double near = 0, far = 1e30;
   for (unsigned a = 0; a < 3; ++a) {
      if (dir[a] == 0) {
         if (origin[a] < b.min[a] || origin[a] > b.max[a])
            return false;
         continue;
      }
      double lo = (double(b.min[a]) - origin[a]) / dir[a];
      double hi = (double(b.max[a]) - origin[a]) / dir[a];
      near = std::max(near, std::min(lo, hi));
      far = std::min(far, std::max(lo, hi));
   }
   return near <= far;
}

static void replay_bounds(const std::vector<vtvk_build_node> &leaves,
                          const vtvk_bvh_topology &tree)
{
   for (unsigned r = 0; r < 512; ++r) {
      const auto &target = leaves[(r * 97) % leaves.size()].bounds;
      float origin[3] = {(target.min[0] + target.max[0]) * 0.5f,
                         target.min[1], -1};
      float dir[3] = {r % 2 ? 0.0f : 0.1f, 0, 1};
      std::vector<bool> expected(leaves.size()), found(leaves.size());
      for (size_t i = 0; i < leaves.size(); ++i)
         expected[i] = intersects(leaves[i].bounds, origin, dir);
      std::vector<unsigned> stack{unsigned(tree.nodes.size() - 1)};
      while (!stack.empty()) {
         const auto &n = tree.nodes[stack.back()];
         stack.pop_back();
         if (!intersects(n.bounds, origin, dir))
            continue;
         if (!n.child_count)
            found[n.leaf] = true;
         else
            stack.insert(stack.end(), n.children, n.children + n.child_count);
      }
      CHECK(found == expected);
   }
}

int main()
{
   std::string error;
   vtvk_bvh_topology tree;
   std::vector<vtvk_build_node> leaves;
   if (!vtvk_embree_available()) {
      CHECK(!vtvk_build_embree(leaves, &tree, &error));
      CHECK(!error.empty());
      std::puts("PASS: unavailable builder rejects explicit requests");
      return 0;
   }
   CHECK(!vtvk_build_embree(leaves, &tree, &error));
   for (unsigned count : {1u, 2u, 3u, 4u, 5u, 17u, 1024u, 78666u}) {
      leaves.clear();
      for (unsigned i = 0; i < count; ++i) {
         float x = float((i * 7919u) % 104729u);
         leaves.push_back({i, {{x, float(i % 13), 0},
                              {x + 1, float(i % 13) + 1, 0}}});
      }
      CHECK(vtvk_build_embree(leaves, &tree, &error));
      check_tree(leaves, tree);
      if (count == 1024)
         replay_bounds(leaves, tree);
      vtvk_bvh_topology second;
      CHECK(vtvk_build_embree(leaves, &second, &error));
      CHECK(second.nodes.size() == tree.nodes.size());
      for (size_t i = 0; i < tree.nodes.size(); ++i) {
         CHECK(second.nodes[i].child_count == tree.nodes[i].child_count);
         CHECK(second.nodes[i].leaf == tree.nodes[i].leaf);
         for (unsigned c = 0; c < 4; ++c)
            CHECK(second.nodes[i].children[c] == tree.nodes[i].children[c]);
      }
      CHECK(vtvk_build_embree(leaves, &tree, &error, 24));
      CHECK(tree.nodes.back().stack_required <= 24);
      check_tree(leaves, tree);
      if (count == 1024)
         replay_bounds(leaves, tree);
   }
   /* Coincident, zero-area bounds and conservative TLAS+BLAS stack budget. */
   leaves.assign(1024, {0, {{0, 0, 0}, {0, 0, 0}}});
   CHECK(vtvk_build_embree(leaves, &tree, &error));
   check_tree(leaves, tree);
   leaves.resize(4);
   for (auto &leaf : leaves)
      leaf.stack_required = 30;
   CHECK(vtvk_build_embree(leaves, &tree, &error));
   CHECK(tree.branching == 2);
   CHECK(tree.nodes.back().stack_required == 32);
   for (auto &leaf : leaves)
      leaf.stack_required = 32;
   CHECK(!vtvk_build_embree(leaves, &tree, &error));
   leaves.resize(1);
   CHECK(vtvk_build_embree(leaves, &tree, &error));
   CHECK(tree.nodes.back().stack_required == 32);
   leaves[0].bounds.min[0] = std::numeric_limits<float>::quiet_NaN();
   CHECK(!vtvk_build_embree(leaves, &tree, &error));
   leaves[0].bounds.min[0] = 1;
   CHECK(!vtvk_build_embree(leaves, &tree, &error));
   std::puts("PASS: Embree topology, preservation, degeneracy, determinism, stack limits");
}
