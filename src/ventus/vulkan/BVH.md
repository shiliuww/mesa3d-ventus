# Ventus host BVH construction

The builder changes host-side topology only. VTAS version, Box4/Triangle/AABB/
Instance layouts, primitive IDs, instance IDs, masks and SBT data are unchanged.
Application BLAS/TLAS partitioning is not rewritten. Existing geometry support
limits still apply; the topology builder does not itself implement update/refit
or change geometry support.

## Build and select

Meson `ventus_embree=auto|enabled|disabled` controls the optional Embree 3
dependency. A system SDK is discovered through its CMake package. Intel's binary
SDK can instead be selected with `ventus_embree_prefix=/absolute/sdk/path`, which
uses its `include/embree3` and shared `lib/libembree3.so` directly; its bundled TBB
runtime must remain available. The prefix is a build setting, never hard-coded
in the driver. Use the canonical `mesa/build-ventus` build for full applications.

Runtime `VENTUS_VK_BVH_BUILDER=embree4|morton4` selects the builder. With Embree
compiled in the default is `embree4`; otherwise it is `morton4`. Explicitly
requesting an unavailable or unknown builder fails, rather than silently changing
the experiment. `morton4` retains the original Morton sort and bottom-up grouping.

The first Embree policy uses MEDIUM quality, maximum four children, one primitive
per leaf, depth limit 32, SAH block size 1 and traversal/intersection costs 1/1.
No spatial splitting or primitive duplication is enabled. One host worker and
ignored external Embree configuration files make the experiment reproducible.
This policy currently does not select a different builder from Vulkan build
preference flags; fast-build and cost-weight tuning are future work.

## Stack safety and storage

Each leaf has a conservative stack requirement of one. Instance leaves inherit
the referenced BLAS requirement. An internal node with k children requires
`k - 1 + max(child requirements)` entries, including the subtree root. This
assumes arbitrary ray-dependent visitation order and all children intersecting;
it is an upper bound, not a measured ray stack high-water mark.

BLAS and TLAS construction use the shared 64-entry traversal stack budget,
matching a lane's 32 base and up to 32 overflow entries. Hardware has only
four shared overflow stacks per physical session, not one per lane; deeper
lanes must wait for an available overflow stack. TLAS instance leaves carry the
referenced BLAS requirement, so the composed tree is checked against the same
budget; no fixed TLAS quota is reserved in each BLAS. This is conservative and
may reject scenes that particular rays would traverse safely; it does not
guarantee arbitrary scene sizes fit. Candidate spill/resume is a separate
hardware lifecycle mechanism, not a change to host BVH construction.

If four-way Embree output exceeds the budget, build a binary SAH tree. If that
fits, widen it selectively: expand the largest-area internal child while the
frontier has at most four children and every subtree fits the residual budget.
Pass that residual budget down recursively. Otherwise report a build error.
The result is still serialized exclusively into the existing Box4 format, with
invalid references in unused slots. The widening heuristic is not claimed to
globally minimize SAH.

Construction verifies finite ordered bounds, exactly one occurrence of every
input primitive, bounded depth and child counts. Parent bounds are recomputed
from original child bounds. The host temporary tree is released after export;
callbacks allocate through Embree's thread-local allocator, not shared vectors.

AS size queries reserve up to N-1 internal nodes, plus leaf bytes and alignment
slack. This is necessary for variable fanout; the old balanced-four-way estimate
was insufficient. Build admission checks allocation size and the 32-bit offset
range before writing leaves. Actual used bytes remain in the VTAS header; the
Vulkan allocation and existing snapshot can be larger. The conservative size
query applies to both builders to allow switching without mismatched allocations.

## Evidence and comparison

Each AS build emits a bounded line identifying builder, Embree version, leaf/node
counts, initial branching factor, whether fallback widening was used, Embree-path
fanout counts, depth, conservative combined stack bound, actual bytes and host
topology time. `topology_ms` excludes input extraction, leaf serialization and
upload; it is not simulated GPU cycles. Legacy fanout counters are not collected
(their displayed zero values are not a measurement).

Build `src/ventus/vulkan/vtvk_bvh_builder_test` with Ninja, then run that binary.
Tests cover primitive preservation, containment, postorder topology, determinism,
coincident/planar bounds, invalid inputs, budget limits, fallback widening and
ray/AABB replay against brute force. Full image correctness is checked separately
against lavapipe through `tools/rtcore.py regression` (1x1, then 16x9).

Compare builders using the same executable, seed, scene, ray settings and PMU
mode. Record the environment in the regression artifacts. The allocation size
bound changed, so even `morton4` is not an address-identical replay of an older
binary: cache effects can differ. BVH quality, node format, memory footprint and
hardware scheduling are distinct contributors when comparing Vulkan-Sim.
