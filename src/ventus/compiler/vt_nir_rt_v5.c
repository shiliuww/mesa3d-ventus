/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include <float.h>

#include "vt_nir.h"
#include "ventus/common/ventus_rt_bvh.h"
#include "ventus/common/ventus_rt_v5.h"

#define VENTUS_RT_V5_SUPPORTED_RAY_FLAGS 0x000003ffu

static nir_def *
add_u64_offset(nir_builder *b, nir_def *base, uint32_t offset)
{
   return nir_iadd_imm(b, nir_u2u64(b, base), offset);
}

static nir_def *
load_global_u32(nir_builder *b, nir_def *base, uint32_t offset)
{
   return nir_load_global(b, 1, 32, add_u64_offset(b, base, offset),
                          .align_mul = 4);
}

static void
store_pds_u32(nir_builder *b, nir_def *base, uint32_t offset, nir_def *value)
{
   nir_ventus_rt_store_slot(b, nir_u2u32(b, value),
                            nir_iadd_imm(b, base, offset));
}

static void
store_pds_u64(nir_builder *b, nir_def *base, uint32_t offset, nir_def *value)
{
   nir_def *wide = nir_u2u64(b, value);
   store_pds_u32(b, base, offset, nir_unpack_64_2x32_split_x(b, wide));
   store_pds_u32(b, base, offset + 4, nir_unpack_64_2x32_split_y(b, wide));
}

static void
store_shared_u32(nir_builder *b, nir_def *base, uint32_t offset,
                 nir_def *value)
{
   nir_store_shared(b, nir_u2u32(b, value), nir_iadd_imm(b, base, offset),
                    .write_mask = 0x1, .align_mul = 4);
}

static void
store_shared_u64(nir_builder *b, nir_def *base, uint32_t offset,
                 nir_def *value)
{
   nir_def *wide = nir_u2u64(b, value);
   store_shared_u32(b, base, offset, nir_unpack_64_2x32_split_x(b, wide));
   store_shared_u32(b, base, offset + 4,
                    nir_unpack_64_2x32_split_y(b, wide));
}

static nir_def *
all_finite_vec(nir_builder *b, nir_def *value)
{
   nir_def *finite = nir_imm_true(b);
   for (unsigned i = 0; i < value->num_components; i++)
      finite = nir_iand(b, finite, nir_fisfinite(b, nir_channel(b, value, i)));
   return finite;
}

static nir_def *
inverse_direction_component(nir_builder *b, nir_def *direction)
{
   nir_def *reciprocal = nir_frcp(b, direction);
   nir_def *saturated = nir_bcsel(
      b, nir_flt_imm(b, direction, 0.0f), nir_imm_float(b, -FLT_MAX),
      nir_imm_float(b, FLT_MAX));
   return nir_bcsel(b, nir_fisfinite(b, reciprocal), reciprocal, saturated);
}

nir_def *
vt_nir_rt_v5_private_address(nir_builder *b, nir_def *cta_shared_base,
                             nir_def *private_region_offset,
                             nir_def *warp_index_in_cta, nir_def *lane_id)
{
   nir_def *address = nir_iadd(b, nir_u2u32(b, cta_shared_base),
                              nir_u2u32(b, private_region_offset));
   address = nir_iadd(b, address,
                      nir_imul_imm(b, nir_u2u32(b, warp_index_in_cta),
                                   VENTUS_RT_V5_PRIVATE_WARP_STRIDE_BYTES));
   return nir_iadd(b, address,
                   nir_imul_imm(b, nir_u2u32(b, lane_id),
                                VENTUS_RT_V5_PRIVATE_LANE_STRIDE_BYTES));
}

static nir_def *
pack_trace_selector(nir_builder *b,
                    const struct vt_nir_rt_trace_params *params)
{
   nir_def *selector = nir_iand_imm(b, params->cull_mask, 0xff);
   selector = nir_ior(
      b, selector,
      nir_ishl_imm(b, nir_iand_imm(b, params->sbt_offset, 0xf), 8));
   selector = nir_ior(
      b, selector,
      nir_ishl_imm(b, nir_iand_imm(b, params->sbt_stride, 0xf), 12));
   return nir_ior(
      b, selector,
      nir_ishl_imm(b, nir_iand_imm(b, params->miss_index, 0xffff), 16));
}

static nir_def *
valid_trace_params(nir_builder *b,
                   const struct vt_nir_rt_trace_params *params)
{
   nir_def *valid = nir_imm_true(b);
   valid = nir_iand(
      b, valid,
      nir_ieq_imm(b, nir_iand_imm(b, params->flags,
                                  ~VENTUS_RT_V5_SUPPORTED_RAY_FLAGS), 0));
   valid = nir_iand(b, valid, nir_ult_imm(b, params->cull_mask, 0x100));
   valid = nir_iand(b, valid, nir_ult_imm(b, params->sbt_offset, 0x10));
   valid = nir_iand(b, valid, nir_ult_imm(b, params->sbt_stride, 0x10));
   valid = nir_iand(b, valid, nir_ult_imm(b, params->miss_index, 0x10000));
   valid = nir_iand(b, valid, nir_fisfinite(b, params->tmin));
   valid = nir_iand(b, valid, nir_fisfinite(b, params->tmax));
   valid = nir_iand(b, valid, nir_fge(b, params->tmax, params->tmin));
   valid = nir_iand(b, valid, all_finite_vec(b, params->origin));
   return nir_iand(b, valid, all_finite_vec(b, params->direction));
}

static void
write_v5_pds_input(nir_builder *b,
                   const struct vt_nir_rt_trace_params *params,
                   nir_def *pds_address)
{
   store_pds_u64(b, pds_address, VENTUS_RT_V5_PDS_TRAVERSABLE_REFERENCE,
                 params->accel_struct);
   store_pds_u32(b, pds_address, VENTUS_RT_V5_PDS_RAY_FLAGS, params->flags);
   store_pds_u32(b, pds_address, VENTUS_RT_V5_PDS_TRACE_SELECTOR_CONTROL,
                 pack_trace_selector(b, params));
   for (uint32_t offset = 0x10; offset < 0x20; offset += 4)
      store_pds_u32(b, pds_address, offset, nir_imm_int(b, 0));
   for (unsigned i = 0; i < 3; i++)
      store_pds_u32(b, pds_address,
                    VENTUS_RT_V5_PDS_WORLD_ORIGIN + i * 4,
                    nir_channel(b, params->origin, i));
   store_pds_u32(b, pds_address, VENTUS_RT_V5_PDS_RAY_TMIN, params->tmin);
   for (unsigned i = 0; i < 3; i++)
      store_pds_u32(b, pds_address,
                    VENTUS_RT_V5_PDS_WORLD_DIRECTION + i * 4,
                    nir_channel(b, params->direction, i));
   store_pds_u32(b, pds_address, VENTUS_RT_V5_PDS_RAY_TMAX, params->tmax);

   for (uint32_t offset = 0; offset < 0x20; offset += 4)
      store_pds_u32(b, pds_address,
                    VENTUS_RT_V5_PDS_COMMITTED_BASE + offset,
                    offset == 0 ? params->tmax : nir_imm_int(b, 0));
}

static void
write_v5_private_input(nir_builder *b,
                       const struct vt_nir_rt_trace_params *params,
                       nir_def *private_state_address, nir_def *root_ref)
{
   store_shared_u64(b, private_state_address,
                    VENTUS_RT_V5_PRIVATE_FRONTIER_0,
                    nir_u2u64(b, root_ref));
   store_shared_u32(b, private_state_address,
                    VENTUS_RT_V5_PRIVATE_RAY_TMIN, params->tmin);
   store_shared_u32(b, private_state_address,
                    VENTUS_RT_V5_PRIVATE_RAY_TMAX, params->tmax);
   store_shared_u32(b, private_state_address,
                    VENTUS_RT_V5_PRIVATE_RAY_FLAGS, params->flags);

   nir_def *hot_control = nir_iand_imm(b, params->cull_mask, 0xff);
   for (unsigned i = 0; i < 3; i++) {
      nir_def *parallel =
         nir_feq_imm(b, nir_channel(b, params->direction, i), 0.0f);
      hot_control = nir_ior(
         b, hot_control,
         nir_ishl_imm(b, nir_b2i32(b, parallel), 14 + i));
   }
   store_shared_u32(b, private_state_address,
                    VENTUS_RT_V5_PRIVATE_HOT_CONTROL, hot_control);

   for (unsigned i = 0; i < 3; i++) {
      nir_def *direction = nir_channel(b, params->direction, i);
      store_shared_u32(b, private_state_address,
                       VENTUS_RT_V5_PRIVATE_ORIGIN + i * 4,
                       nir_channel(b, params->origin, i));
      store_shared_u32(b, private_state_address,
                       VENTUS_RT_V5_PRIVATE_INV_DIRECTION + i * 4,
                       inverse_direction_component(b, direction));
      store_shared_u32(b, private_state_address,
                       VENTUS_RT_V5_PRIVATE_DIRECTION + i * 4, direction);
   }
   store_shared_u64(b, private_state_address,
                    VENTUS_RT_V5_PRIVATE_ACTIVE_AS_BASE,
                    params->accel_struct);
}

struct vt_nir_rt_v5_submit_result
vt_nir_build_traversal_v5_submit(nir_builder *b,
                                 const struct vt_nir_rt_trace_params *params,
                                 nir_def *pds_address,
                                 nir_def *private_state_address)
{
   nir_def *as_address_valid = nir_iand(
      b, nir_ine_imm(b, params->accel_struct, 0),
      nir_ieq_imm(b, nir_iand_imm(b, params->accel_struct, 0x7), 0));
   nir_push_if(b, as_address_valid);
   nir_def *magic = load_global_u32(
      b, params->accel_struct, VENTUS_AS_HEADER_MAGIC);
   nir_def *version_and_size = load_global_u32(
      b, params->accel_struct, VENTUS_AS_HEADER_VERSION);
   nir_def *loaded_root_ref = load_global_u32(
      b, params->accel_struct, VENTUS_AS_HEADER_ROOT_NODE_REF);
   nir_def *loaded_header_valid = nir_iand(
      b, nir_ieq_imm(b, magic, VENTUS_AS_MAGIC),
      nir_iand(b, nir_ieq_imm(b, nir_iand_imm(b, version_and_size, 0xffff), 2),
               nir_ieq_imm(b, nir_ushr_imm(b, version_and_size, 16),
                           VENTUS_AS_HEADER_SIZE)));
   nir_push_else(b, NULL);
   nir_def *invalid_root_ref = nir_imm_int(b, VENTUS_AS_INVALID_NODE);
   nir_def *invalid_header = nir_imm_false(b);
   nir_pop_if(b, NULL);
   nir_def *root_ref = nir_if_phi(b, loaded_root_ref, invalid_root_ref);
   nir_def *valid_header = nir_if_phi(b, loaded_header_valid, invalid_header);
   nir_def *params_valid = nir_iand(
      b, as_address_valid, valid_trace_params(b, params));
   nir_def *empty_as = nir_iand(
      b, valid_header, nir_ieq_imm(b, root_ref, VENTUS_AS_INVALID_NODE));
   nir_def *admitted = nir_iand(
      b, nir_iand(b, params_valid, valid_header), nir_inot(b, empty_as));

   nir_def *bypass = nir_bcsel(
      b, nir_inot(b, params_valid),
      nir_imm_int(b,
                  ventus_rt_v5_compact_result(VENTUS_RT_V5_RETURN_FAULT)),
      nir_bcsel(
         b, nir_inot(b, valid_header),
         nir_imm_int(b, ventus_rt_v5_compact_result(
                           VENTUS_RT_V5_RETURN_UNSUPPORTED)),
         nir_imm_int(b,
                     ventus_rt_v5_compact_result(VENTUS_RT_V5_RETURN_MISS))));

   nir_push_if(b, admitted);
   write_v5_pds_input(b, params, pds_address);
   write_v5_private_input(b, params, private_state_address, root_ref);
   nir_scoped_memory_barrier(b, SCOPE_DEVICE, NIR_MEMORY_RELEASE,
                             nir_var_mem_shared | nir_var_mem_global);
   nir_def *traversed =
      nir_ventus_rt_traverse_v5(b, pds_address, private_state_address);
   nir_push_else(b, NULL);
   nir_def *bypassed = bypass;
   nir_pop_if(b, NULL);

   return (struct vt_nir_rt_v5_submit_result) {
      .compact_result = nir_if_phi(b, traversed, bypassed),
      .private_state_address = private_state_address,
      .traverse_admitted = admitted,
   };
}
