/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <array>

#include "ventus/common/ventus_rt_bvh.h"
#include "ventus/common/ventus_rt_bvh_v2.h"

static void
store_u16(uint8_t *data, uint32_t offset, uint16_t value)
{
   data[offset + 0] = uint8_t(value);
   data[offset + 1] = uint8_t(value >> 8);
}

static void
store_u32(uint8_t *data, uint32_t offset, uint32_t value)
{
   data[offset + 0] = uint8_t(value);
   data[offset + 1] = uint8_t(value >> 8);
   data[offset + 2] = uint8_t(value >> 16);
   data[offset + 3] = uint8_t(value >> 24);
}

static std::array<uint8_t, 512>
make_two_instance_tlas()
{
   constexpr uint32_t instance0_offset = 80;
   constexpr uint32_t instance1_offset = 208;
   constexpr uint32_t root_offset = 384;
   constexpr uint32_t root_ref = root_offset | VENTUS_BVH_NODE_BOX4;

   std::array<uint8_t, 512> image = {};
   store_u32(image.data(), VENTUS_AS_HEADER_MAGIC, VENTUS_AS_MAGIC);
   store_u16(image.data(), VENTUS_AS_HEADER_VERSION, VENTUS_AS_VERSION_V2);
   store_u16(image.data(), VENTUS_AS_HEADER_HEADER_SIZE,
             VENTUS_AS_HEADER_SIZE);
   store_u32(image.data(), VENTUS_AS_HEADER_TYPE, VENTUS_AS_TYPE_TLAS);
   store_u32(image.data(), VENTUS_AS_HEADER_ROOT_NODE_REF, root_ref);
   store_u32(image.data(), VENTUS_AS_HEADER_NODE_COUNT, 3);
   store_u32(image.data(), VENTUS_AS_HEADER_NODE_REGION_OFFSET,
             VENTUS_AS_HEADER_SIZE);
   store_u32(image.data(), VENTUS_AS_HEADER_NODE_REGION_SIZE,
             image.size() - VENTUS_AS_HEADER_SIZE);
   store_u32(image.data(), VENTUS_AS_HEADER_INSTANCE_REGION_OFFSET,
             VENTUS_AS_HEADER_SIZE);
   store_u32(image.data(), VENTUS_AS_HEADER_INSTANCE_COUNT, 2);

   store_u32(image.data() + instance0_offset,
             VENTUS_INSTANCE_V2_PARENT_EDGE, root_offset | 0);
   store_u32(image.data() + instance1_offset,
             VENTUS_INSTANCE_V2_PARENT_EDGE, root_offset | 1);

   uint8_t *root = image.data() + root_offset;
   for (uint32_t i = 0; i < 4; i++)
      store_u32(root, VENTUS_BOX4_CHILD_REF + i * 4,
                VENTUS_AS_INVALID_NODE);
   store_u32(root, VENTUS_BOX4_CHILD_REF + 0 * 4,
             instance0_offset | VENTUS_BVH_NODE_INSTANCE);
   store_u32(root, VENTUS_BOX4_CHILD_REF + 1 * 4,
             instance1_offset | VENTUS_BVH_NODE_INSTANCE);
   store_u32(root, VENTUS_BOX4_V2_PARENT_NODE_REF,
             VENTUS_AS_INVALID_NODE);
   root[VENTUS_BOX4_V2_PARENT_CHILD_INDEX] = 0xff;
   root[VENTUS_BOX4_V2_NODE_FLAGS] = VENTUS_BOX4_V2_NODE_FLAG_ROOT;
   return image;
}

static enum ventus_vtas_v2_validation_error
validate(const uint8_t *data, size_t size)
{
   struct ventus_vtas_v2_validation_result result = {};
   if (ventus_vtas_v2_validate(data, size, &result))
      return VENTUS_VTAS_V2_VALID;
   return result.error;
}

int
main()
{
   const auto valid = make_two_instance_tlas();
   assert(validate(valid.data(), valid.size()) == VENTUS_VTAS_V2_VALID);

   auto bad = valid;
   store_u16(bad.data(), VENTUS_AS_HEADER_VERSION, VENTUS_AS_VERSION_V1);
   assert(validate(bad.data(), bad.size()) == VENTUS_VTAS_V2_BAD_VERSION);

   bad = valid;
   bad[384 + VENTUS_BOX4_V2_NODE_FLAGS] = 0;
   assert(validate(bad.data(), bad.size()) == VENTUS_VTAS_V2_BAD_ROOT);

   bad = valid;
   store_u32(bad.data() + 80, VENTUS_INSTANCE_V2_PARENT_EDGE, 384 | 1);
   assert(validate(bad.data(), bad.size()) == VENTUS_VTAS_V2_BAD_PARENT);

   bad = valid;
   bad[384 + VENTUS_BOX4_V2_RESERVED] = 1;
   assert(validate(bad.data(), bad.size()) ==
          VENTUS_VTAS_V2_NONZERO_RESERVED);

   bad = valid;
   store_u32(bad.data() + 384, VENTUS_BOX4_CHILD_REF,
             384 | VENTUS_BVH_NODE_BOX4);
   assert(validate(bad.data(), bad.size()) ==
          VENTUS_VTAS_V2_DUPLICATE_OR_CYCLE);

   bad = valid;
   store_u32(bad.data(), VENTUS_AS_HEADER_NODE_COUNT, 4);
   assert(validate(bad.data(), bad.size()) ==
          VENTUS_VTAS_V2_NODE_COUNT_MISMATCH);

   assert(validate(valid.data(), valid.size() - 1) ==
          VENTUS_VTAS_V2_TRUNCATED);
   return 0;
}
