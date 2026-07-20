/*
 * Copyright 2026
 * SPDX-License-Identifier: MIT
 */

#include "vtvk_driver_bridge.h"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <inttypes.h>
#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "util/log.h"
#include "ventus/common/ventus_rt_bvh.h"
#include "ventus/compiler/vt_nir.h"

static_assert(VENTUS_RT_RESIDENT_WARPS > 0,
              "Ventus RT requires at least one resident warp");

static constexpr uint64_t vtvk_rt_resident_warps =
   VENTUS_RT_RESIDENT_WARPS;
static constexpr uint64_t vtvk_rt_lanes_per_warp = 32;

static_assert(sizeof(struct vtvk_rt_descriptor_record) ==
              VT_NIR_RT_DESCRIPTOR_RECORD_SIZE,
              "RT descriptor ABI size mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, abi_tag) ==
              VT_NIR_RT_DISPATCH_ARG_ABI_TAG,
              "RT ABI tag dispatch argument mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, continuation_arena_addr) ==
              VT_NIR_RT_DISPATCH_ARG_CONTINUATION_ARENA_ADDR,
              "RT CPS arena argument ABI mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, continuation_frame_stride) ==
              VT_NIR_RT_DISPATCH_ARG_CONTINUATION_FRAME_STRIDE,
              "RT CPS frame stride argument ABI mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, continuation_context_count) ==
              VT_NIR_RT_DISPATCH_ARG_CONTINUATION_CONTEXT_COUNT,
              "RT CPS context count argument ABI mismatch");
static_assert(sizeof(struct vtvk_rt_dispatch_args) ==
              VT_NIR_RT_DISPATCH_ARG_SIZE,
              "RT dispatch argument ABI size mismatch");

struct vtvk_driver_kernel_metadata {
   uint64_t kernel_id;
   uint64_t kernel_size[3];
   uint64_t wf_size;
   uint64_t wg_size;
   uint64_t metaDataBaseAddr;
   uint64_t ldsSize;
   uint64_t pdsSize;
   uint64_t sgprUsage;
   uint64_t vgprUsage;
   uint64_t pdsBaseAddr;
   uint64_t num_thread_global[3];
   uint64_t num_thread_local[3];
   uint64_t threadIdxOffset[3];
   const char *kernel_name;
   uint64_t pdsResidentWgCount;
};

struct vtvk_driver_api {
   void *lib;
   void *device;

   int (*vt_dev_open)(void **device);
   int (*vt_dev_close)(void *device);
   int (*vt_buf_alloc)(void *device, uint64_t size, uint64_t *vaddr,
                       int buf_type, uint64_t task_id, uint64_t kernel_id);
   int (*vt_buf_free)(void *device, uint64_t size, uint64_t *vaddr,
                      uint64_t task_id, uint64_t kernel_id);
   int (*vt_copy_to_dev)(void *device, uint64_t dev_vaddr, const void *src,
                         uint64_t size, uint64_t task_id, uint64_t kernel_id);
   int (*vt_copy_from_dev)(void *device, uint64_t dev_vaddr, void *dst,
                           uint64_t size, uint64_t task_id, uint64_t kernel_id);
   int (*vt_upload_kernel_file)(void *device, const char *filename,
                                int kernel_id);
   int (*vt_start)(void *device,
                   struct vtvk_driver_kernel_metadata *metadata,
                   uint64_t task_id);
   int (*vt_ready_wait)(void *device, uint64_t timeout_ms);
};

struct vtvk_driver_upload {
   uint64_t dev_addr;
   uint64_t size;
};

enum vtvk_driver_knl_offset {
   VTVK_KNL_ENTRY = 0,
   VTVK_KNL_ARG_BASE = 4,
   VTVK_KNL_WORK_DIM = 8,
   VTVK_KNL_GL_SIZE_X = 12,
   VTVK_KNL_GL_SIZE_Y = 16,
   VTVK_KNL_GL_SIZE_Z = 20,
   VTVK_KNL_LC_SIZE_X = 24,
   VTVK_KNL_LC_SIZE_Y = 28,
   VTVK_KNL_LC_SIZE_Z = 32,
   VTVK_KNL_GL_OFFSET_X = 36,
   VTVK_KNL_GL_OFFSET_Y = 40,
   VTVK_KNL_GL_OFFSET_Z = 44,
   VTVK_KNL_MAX_METADATA_SIZE = 64,
};

static bool
vtvk_bridge_env_enabled(const char *name)
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

static uint32_t
vtvk_bridge_load_u32(const uint8_t *base, uint32_t offset)
{
   uint32_t value;
   memcpy(&value, base + offset, sizeof(value));
   return value;
}

static uint64_t
vtvk_bridge_load_u64(const uint8_t *base, uint32_t offset)
{
   uint64_t value;
   memcpy(&value, base + offset, sizeof(value));
   return value;
}

static void
vtvk_bridge_store_u64(uint8_t *base, uint32_t offset, uint64_t value)
{
   memcpy(base + offset, &value, sizeof(value));
}

static uint64_t
vtvk_bridge_align_u64(uint64_t value, uint64_t alignment)
{
   return (value + alignment - 1) & ~(alignment - 1);
}

static uint32_t
vtvk_bridge_format_bytes_per_pixel(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
      return 4;
   case VK_FORMAT_R16G16B16A16_UNORM:
   case VK_FORMAT_R16G16B16A16_SNORM:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
      return 8;
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32G32B32A32_SFLOAT:
      return 16;
   case VK_FORMAT_R16G16_UNORM:
   case VK_FORMAT_R16G16_SNORM:
   case VK_FORMAT_R16G16_UINT:
      return 4;
   default:
      break;
   }

   mesa_logw("ventus-vk: unsupported storage image format 0x%x, using 4 bytes/pixel",
             (unsigned)format);
   return 4;
}

static const char *
vtvk_bridge_backend_library(void)
{
   const char *override_path = getenv("VENTUS_VK_DRIVER_LIB");
   if (override_path && override_path[0])
      return override_path;

   const char *backend = getenv("VENTUS_VK_DRIVER_BACKEND");
   if (!backend || !backend[0])
      backend = "spike";

   if (strcmp(backend, "rtlsim") == 0)
      return "librtlsim_driver.so";
   if (strcmp(backend, "cyclesim") == 0)
      return "libcyclesim_driver.so";
   if (strcmp(backend, "auto") == 0)
      return "libauto_select_driver.so";

   return "libspike_driver.so";
}

static bool
vtvk_driver_resolve(void *lib, const char *name, void **out)
{
   *out = dlsym(lib, name);
   if (!*out) {
      mesa_loge("ventus-vk: driver bridge missing symbol %s", name);
      return false;
   }
   return true;
}

static bool
vtvk_driver_open(struct vtvk_driver_api *api)
{
   memset(api, 0, sizeof(*api));

   const char *path = vtvk_bridge_backend_library();
   api->lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
   if (!api->lib) {
      mesa_loge("ventus-vk: driver bridge dlopen failed: %s", dlerror());
      return false;
   }

#define VTVK_RESOLVE(name)                                                     \
   if (!vtvk_driver_resolve(api->lib, #name, (void **)&api->name))             \
      return false
   VTVK_RESOLVE(vt_dev_open);
   VTVK_RESOLVE(vt_dev_close);
   VTVK_RESOLVE(vt_buf_alloc);
   VTVK_RESOLVE(vt_buf_free);
   VTVK_RESOLVE(vt_copy_to_dev);
   VTVK_RESOLVE(vt_copy_from_dev);
   VTVK_RESOLVE(vt_upload_kernel_file);
   VTVK_RESOLVE(vt_start);
   VTVK_RESOLVE(vt_ready_wait);
#undef VTVK_RESOLVE

   if (api->vt_dev_open(&api->device) != 0 || !api->device) {
      mesa_loge("ventus-vk: driver bridge vt_dev_open failed");
      return false;
   }

   return true;
}

static void
vtvk_driver_close(struct vtvk_driver_api *api)
{
   if (api->device && api->vt_dev_close)
      api->vt_dev_close(api->device);
   if (api->lib)
      dlclose(api->lib);
   memset(api, 0, sizeof(*api));
}

static bool
vtvk_driver_alloc_copy(struct vtvk_driver_api *api, const void *src,
                       uint64_t size, uint64_t *dev_addr)
{
   if (!size)
      size = 1;
   if (api->vt_buf_alloc(api->device, size, dev_addr, 0, 0, 0) != 0)
      return false;
   return !src ||
          api->vt_copy_to_dev(api->device, *dev_addr, src, size, 0, 0) == 0;
}

static bool
vtvk_driver_dump_image_ppm(const char *path, const void *data,
                           uint32_t width, uint32_t height, VkFormat format)
{
   if (!path || !path[0] || !data || !width || !height)
      return true;

   FILE *file = fopen(path, "wb");
   if (!file) {
      mesa_loge("ventus-vk: failed to open image dump %s", path);
      return false;
   }

   fprintf(file, "P6\n%u %u\n255\n", width, height);
   const uint8_t *pixels = (const uint8_t *)data;
   for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
         const uint8_t *p = pixels + ((uint64_t)y * width + x) * 4;
         uint8_t rgb[3];
         switch (format) {
         case VK_FORMAT_B8G8R8A8_UNORM:
            rgb[0] = p[2];
            rgb[1] = p[1];
            rgb[2] = p[0];
            break;
         default:
            rgb[0] = p[0];
            rgb[1] = p[1];
            rgb[2] = p[2];
            break;
         }
         fwrite(rgb, 1, sizeof(rgb), file);
      }
   }

   bool ok = fclose(file) == 0;
   if (ok)
      mesa_logi("ventus-vk: dumped Spike storage image to %s", path);
   else
      mesa_loge("ventus-vk: failed to write image dump %s", path);
   return ok;
}

static void
vtvk_driver_log_u32_words(const char *name, const void *data, uint64_t size,
                          uint32_t max_words)
{
   if (!vtvk_bridge_env_enabled("VENTUS_VK_PROBE_LOG") || !data || !size)
      return;

   const uint32_t *words = (const uint32_t *)data;
   uint32_t word_count =
      std::min<uint64_t>(max_words, size / sizeof(uint32_t));
   for (uint32_t i = 0; i < word_count; i += 4) {
      uint32_t w0 = words[i];
      uint32_t w1 = i + 1 < word_count ? words[i + 1] : 0;
      uint32_t w2 = i + 2 < word_count ? words[i + 2] : 0;
      uint32_t w3 = i + 3 < word_count ? words[i + 3] : 0;
      mesa_logi("ventus-vk: driver bridge %s[%03u..%03u] "
                "%08x %08x %08x %08x",
                name, i, i + 3, w0, w1, w2, w3);
   }
}

static bool
vtvk_driver_patch_tlas_node(
   struct vtvk_driver_api *api, uint8_t *image, uint64_t image_size,
   uint32_t node_ref,
   std::unordered_map<struct vtvk_acceleration_structure *,
                      struct vtvk_driver_upload> *uploads);

static bool
vtvk_driver_upload_as(
   struct vtvk_driver_api *api, struct vtvk_acceleration_structure *as,
   std::unordered_map<struct vtvk_acceleration_structure *,
                      struct vtvk_driver_upload> *uploads)
{
   auto existing = uploads->find(as);
   if (existing != uploads->end())
      return true;

   const void *as_data = as->snapshot ? as->snapshot : as->data;
   uint64_t as_size = as->snapshot ? as->snapshot_size : as->size;

   if (!as_data)
      as->data = vtvk_buffer_host_ptr(as->buffer, as->offset, as->size);
   as_data = as->snapshot ? as->snapshot : as->data;
   as_size = as->snapshot ? as->snapshot_size : as->size;
   if (!as_data || !as_size) {
      mesa_loge("ventus-vk: driver bridge AS has no host data");
      return false;
   }

   std::vector<uint8_t> image(as_size);
   memcpy(image.data(), as_data, as_size);

   if (vtvk_bridge_load_u32(image.data(), VENTUS_AS_HEADER_MAGIC) != VENTUS_AS_MAGIC) {
      mesa_loge("ventus-vk: driver bridge AS has invalid VTAS magic");
      return false;
   }

   if (as->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
      uint32_t root_ref =
         vtvk_bridge_load_u32(image.data(), VENTUS_AS_HEADER_ROOT_NODE_REF);
      if (!vtvk_driver_patch_tlas_node(api, image.data(), as_size, root_ref,
                                       uploads)) {
         mesa_loge("ventus-vk: driver bridge failed to patch TLAS root ref=0x%x",
                   root_ref);
         return false;
      }
   }

   struct vtvk_driver_upload upload = {};
   upload.size = as_size;
   if (!vtvk_driver_alloc_copy(api, image.data(), upload.size,
                               &upload.dev_addr)) {
      mesa_loge("ventus-vk: driver bridge failed to allocate/copy AS");
      return false;
   }

   (*uploads)[as] = upload;
   return true;
}

static bool
vtvk_driver_patch_tlas_node(
   struct vtvk_driver_api *api, uint8_t *image, uint64_t image_size,
   uint32_t node_ref,
   std::unordered_map<struct vtvk_acceleration_structure *,
                      struct vtvk_driver_upload> *uploads)
{
   if (node_ref == VENTUS_AS_INVALID_NODE)
      return true;

   uint32_t type = ventus_bvh_node_ref_type(node_ref);
   uint32_t offset = ventus_bvh_node_ref_offset(node_ref);
   if (offset >= image_size) {
      mesa_loge("ventus-vk: driver bridge TLAS node offset out of range "
                "ref=0x%x size=%" PRIu64, node_ref, image_size);
      return false;
   }

   if (type == VENTUS_BVH_NODE_BOX4) {
      if ((uint64_t)offset + VENTUS_AS_BOX4_NODE_SIZE > image_size) {
         mesa_loge("ventus-vk: driver bridge TLAS box4 node out of range");
         return false;
      }
      uint8_t *node = image + offset;
      for (uint32_t i = 0; i < 4; i++) {
         uint32_t child_ref =
            vtvk_bridge_load_u32(node, VENTUS_BOX4_CHILD_REF + i * 4);
         if (!vtvk_driver_patch_tlas_node(api, image, image_size, child_ref,
                                          uploads))
            return false;
      }
      return true;
   }

   if (type == VENTUS_BVH_NODE_INSTANCE) {
      if ((uint64_t)offset + VENTUS_AS_INSTANCE_NODE_SIZE > image_size) {
         mesa_loge("ventus-vk: driver bridge TLAS instance node out of range");
         return false;
      }
      uint8_t *node = image + offset;
      uint64_t host_blas_addr =
         vtvk_bridge_load_u64(node, VENTUS_INSTANCE_BLAS_ADDR_LO);
      struct vtvk_acceleration_structure *blas =
         vtvk_find_as_by_address(host_blas_addr);
      if (!blas) {
         mesa_loge("ventus-vk: driver bridge cannot find BLAS host address "
                   "0x%" PRIx64, host_blas_addr);
         return false;
      }
      if (!vtvk_driver_upload_as(api, blas, uploads))
         return false;
      vtvk_bridge_store_u64(node, VENTUS_INSTANCE_BLAS_ADDR_LO,
                            (*uploads)[blas].dev_addr);
      return true;
   }

   mesa_loge("ventus-vk: driver bridge unsupported TLAS node type %u", type);
   return false;
}

static bool
vtvk_driver_upload_sbt_region(struct vtvk_driver_api *api,
                              const VkStridedDeviceAddressRegionKHR *src,
                              struct vtvk_rt_sbt_region *dst,
                              const char *name)
{
   memset(dst, 0, sizeof(*dst));
   if (!src || !src->deviceAddress || !src->size)
      return true;

   void *host = vtvk_buffer_host_ptr_from_addr(src->deviceAddress, src->size);
   if (!host) {
      mesa_loge("ventus-vk: driver bridge cannot map %s SBT "
                "addr=0x%" PRIx64 " size=%" PRIu64,
                name, (uint64_t)src->deviceAddress, (uint64_t)src->size);
      return false;
   }

   if (!vtvk_driver_alloc_copy(api, host, src->size, &dst->addr)) {
      mesa_loge("ventus-vk: driver bridge failed to upload %s SBT", name);
      return false;
   }

   dst->size = src->size;
   dst->stride = src->stride;
   if (vtvk_bridge_env_enabled("VENTUS_VK_PROBE_LOG")) {
      mesa_logi("ventus-vk: driver bridge %s SBT host=0x%" PRIx64
                " dev=0x%" PRIx64 " size=%" PRIu64 " stride=%" PRIu64,
                name, (uint64_t)src->deviceAddress, dst->addr,
                (uint64_t)src->size, (uint64_t)src->stride);
      vtvk_driver_log_u32_words(name, host, src->size, 16);
   }
   return true;
}

static uint32_t
vtvk_driver_descriptor_table_slots(const struct vtvk_driver_trace_rays_info *info)
{
   uint32_t slots = 5;
   for (uint32_t i = 0; i < info->descriptor_count; i++)
      slots = std::max(slots, info->descriptors[i].binding + 1);
   return slots;
}

static bool
vtvk_driver_upload_descriptor(
   struct vtvk_driver_api *api,
   const struct vtvk_rt_descriptor_source *src,
   struct vtvk_rt_descriptor_record *record,
   std::unordered_map<struct vtvk_acceleration_structure *,
                      struct vtvk_driver_upload> *as_uploads)
{
   memset(record, 0, sizeof(*record));
   record->type = src->type;
   record->set = src->set;
   record->binding = src->binding;

   switch (src->type) {
   case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
      struct vtvk_acceleration_structure *as =
         (struct vtvk_acceleration_structure *)
            (uintptr_t)(uint64_t)src->acceleration_structure;
      if (!as || !vtvk_driver_upload_as(api, as, as_uploads)) {
         mesa_loge("ventus-vk: failed to upload AS descriptor binding=%u",
                   src->binding);
         return false;
      }
      record->device_addr = (*as_uploads)[as].dev_addr;
      record->byte_size = (*as_uploads)[as].size;
      break;
   }

   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
      VkExtent3D extent = {};
      VkFormat format = VK_FORMAT_UNDEFINED;
      void *host = vtvk_image_host_ptr(src->image, &extent, &format);
      uint32_t bpp = vtvk_bridge_format_bytes_per_pixel(format);
      uint32_t stride = extent.width * bpp;
      uint64_t size = (uint64_t)stride * extent.height *
                      (extent.depth ? extent.depth : 1);
      if (!host || !vtvk_driver_alloc_copy(api, host, size, &record->device_addr)) {
         mesa_loge("ventus-vk: failed to upload storage image descriptor binding=%u",
                   src->binding);
         return false;
      }
      record->byte_size = size;
      record->format = format;
      record->stride = stride;
      record->aux = extent.width;
      break;
   }

   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
      void *host = vtvk_buffer_host_ptr(src->buffer, src->offset, src->range);
      if (!host || !vtvk_driver_alloc_copy(api, host, src->range,
                                           &record->device_addr)) {
         mesa_loge("ventus-vk: failed to upload buffer descriptor binding=%u "
                   "range=%" PRIu64,
                   src->binding, (uint64_t)src->range);
         return false;
      }
      record->byte_size = src->range;
      break;
   }

   default:
      mesa_loge("ventus-vk: unsupported RT descriptor binding=%u type=%u",
                src->binding, (unsigned)src->type);
      return false;
   }

   if (vtvk_bridge_env_enabled("VENTUS_VK_PROBE_LOG")) {
      mesa_logi("ventus-vk: descriptor set=%u binding=%u type=%u "
                "addr=0x%" PRIx64 " size=%" PRIu64 " stride=%u format=%u aux=%u",
                record->set, record->binding, record->type,
                record->device_addr, record->byte_size,
                record->stride, record->format, record->aux);
   }

   return true;
}

extern "C" bool
vtvk_driver_trace_rays(const struct vtvk_driver_trace_rays_info *info)
{
   if (!vtvk_bridge_env_enabled("VENTUS_VK_DRIVER_BRIDGE") &&
       !getenv("VENTUS_VK_RAYGEN_ELF"))
      return true;

   static bool bridge_failed;
   static bool staged_without_kernel;
   const char *kernel_path = getenv("VENTUS_VK_RAYGEN_ELF");
   if ((!kernel_path || !kernel_path[0]) && info)
      kernel_path = info->kernel_path;
   if (bridge_failed)
      return false;
   if ((!kernel_path || !kernel_path[0]) && staged_without_kernel)
      return true;

   struct vtvk_driver_api driver = {};
   if (!info || !vtvk_driver_open(&driver))
      return false;

   bool ok = false;
   std::unordered_map<struct vtvk_acceleration_structure *,
                      struct vtvk_driver_upload>
      uploads;
   uint64_t tlas_dev_addr = 0;
   uint64_t image_dev_addr = 0;
   uint64_t camera_dev_addr = 0;
   uint64_t descriptor_table_dev_addr = 0;
   uint64_t args_dev_addr = 0;
   uint64_t metadata_dev_addr = 0;
   uint64_t pds_dev_addr = 0;
   uint64_t continuation_dev_addr = 0;
   uint32_t image_width = info->image_extent.width ? info->image_extent.width : info->width;
   uint32_t image_height = info->image_extent.height ? info->image_extent.height : info->height;
   uint32_t image_depth = info->image_extent.depth ? info->image_extent.depth : 1;
   uint32_t image_bpp = vtvk_bridge_format_bytes_per_pixel(info->image_format);
   uint64_t image_bytes =
      (uint64_t)image_width * image_height * image_depth * image_bpp;
   struct vtvk_rt_dispatch_args args = {};
   std::vector<struct vtvk_rt_descriptor_record> descriptor_table;
   std::vector<uint8_t> pds_image;
   uint32_t metadata[VTVK_KNL_MAX_METADATA_SIZE / 4] = {};
   uint64_t raygen_wg_x = DIV_ROUND_UP(info->width, 32);
   uint64_t raygen_wg_y = MAX2(info->height, 1u);
   uint64_t raygen_wg_z = MAX2(info->depth, 1u);
   uint64_t pds_resident_lanes =
      vtvk_rt_resident_warps * vtvk_rt_lanes_per_warp;
   uint64_t pds_size =
      vtvk_bridge_align_u64(MAX2((uint64_t)info->per_lane_scratch_size *
                                    pds_resident_lanes,
                                 (uint64_t)64 * 1024),
                            4096);
   uint64_t continuation_size =
      (uint64_t)info->continuation_frame_size * pds_resident_lanes;
   struct vtvk_driver_kernel_metadata kernel = {};
   const char *dump_image = getenv("VENTUS_VK_DUMP_IMAGE");

   if (info->descriptor_count) {
      uint32_t slots = vtvk_driver_descriptor_table_slots(info);
      descriptor_table.resize(slots);
      for (uint32_t binding = 0; binding < slots; binding++) {
         descriptor_table[binding].set = 0;
         descriptor_table[binding].binding = binding;
      }

      for (uint32_t i = 0; i < info->descriptor_count; i++) {
         const struct vtvk_rt_descriptor_source *src = &info->descriptors[i];
         if (src->set != 0 || src->binding >= slots) {
            mesa_loge("ventus-vk: unsupported RT descriptor set=%u binding=%u",
                      src->set, src->binding);
            goto out;
         }
         if (!vtvk_driver_upload_descriptor(&driver, src,
                                            &descriptor_table[src->binding],
                                            &uploads))
            goto out;
      }

      if (slots > 0)
         tlas_dev_addr = descriptor_table[0].device_addr;
      if (slots > 1)
         image_dev_addr = descriptor_table[1].device_addr;
      if (slots > 2)
         camera_dev_addr = descriptor_table[2].device_addr;

      if (!vtvk_driver_alloc_copy(&driver, descriptor_table.data(),
                                  descriptor_table.size() *
                                     sizeof(descriptor_table[0]),
                                  &descriptor_table_dev_addr)) {
         mesa_loge("ventus-vk: driver bridge failed to upload descriptor table");
         goto out;
      }
   } else {
      if (!vtvk_driver_upload_as(&driver, info->tlas, &uploads)) {
         mesa_loge("ventus-vk: driver bridge failed to upload TLAS");
         goto out;
      }

      tlas_dev_addr = uploads[info->tlas].dev_addr;
      if (!vtvk_driver_alloc_copy(&driver, info->storage_image_data, image_bytes,
                                  &image_dev_addr)) {
         mesa_loge("ventus-vk: driver bridge failed to upload storage image");
         goto out;
      }

      if (!vtvk_driver_alloc_copy(&driver, info->camera_data, info->camera_range,
                                  &camera_dev_addr)) {
         mesa_loge("ventus-vk: driver bridge failed to upload camera UBO");
         goto out;
      }
   }

   vtvk_driver_log_u32_words("camera", info->camera_data, info->camera_range,
                             32);

   if (!vtvk_driver_upload_sbt_region(&driver, info->raygen_sbt,
                                      &args.raygen_sbt, "raygen") ||
       !vtvk_driver_upload_sbt_region(&driver, info->miss_sbt,
                                      &args.miss_sbt, "miss") ||
       !vtvk_driver_upload_sbt_region(&driver, info->hit_sbt,
                                      &args.hit_sbt, "hit") ||
       !vtvk_driver_upload_sbt_region(&driver, info->callable_sbt,
                                      &args.callable_sbt, "callable"))
      goto out;

   args.tlas_addr = tlas_dev_addr;
   args.image_addr = image_dev_addr;
   args.camera_addr = camera_dev_addr;
   args.width = info->width;
   args.height = info->height;
   args.depth = info->depth;
   args.image_format = info->image_format;
   args.image_stride = image_width * image_bpp;
   args.camera_size = info->camera_range;
   args.scratch_stride = info->per_lane_scratch_size;
   args.abi_tag = VT_RT_ABI_TAG;
   args.continuation_frame_stride = info->continuation_frame_size;
   args.continuation_context_count = pds_resident_lanes;
   args.descriptor_table_addr = descriptor_table_dev_addr;
   args.descriptor_count = descriptor_table.size();
   args.descriptor_record_stride = sizeof(struct vtvk_rt_descriptor_record);
   vtvk_driver_log_u32_words("dispatch_args", &args, sizeof(args), 40);

   if (!vtvk_driver_alloc_copy(&driver, &args, sizeof(args), &args_dev_addr)) {
      mesa_loge("ventus-vk: driver bridge failed to upload dispatch args");
      goto out;
   }

   metadata[VTVK_KNL_ENTRY / 4] =
      info->kernel_entry ? (uint32_t)info->kernel_entry : 0x80000000u;
   metadata[VTVK_KNL_ARG_BASE / 4] = (uint32_t)args_dev_addr;
   metadata[VTVK_KNL_WORK_DIM / 4] = info->depth > 1 ? 3 : info->height > 1 ? 2 : 1;
   metadata[VTVK_KNL_GL_SIZE_X / 4] = info->width;
   metadata[VTVK_KNL_GL_SIZE_Y / 4] = info->height;
   metadata[VTVK_KNL_GL_SIZE_Z / 4] = info->depth;
   metadata[VTVK_KNL_LC_SIZE_X / 4] = 32;
   metadata[VTVK_KNL_LC_SIZE_Y / 4] = 1;
   metadata[VTVK_KNL_LC_SIZE_Z / 4] = 1;
   metadata[VTVK_KNL_GL_OFFSET_X / 4] = 0;
   metadata[VTVK_KNL_GL_OFFSET_Y / 4] = 0;
   metadata[VTVK_KNL_GL_OFFSET_Z / 4] = 0;

   if (!vtvk_driver_alloc_copy(&driver, metadata, sizeof(metadata),
                               &metadata_dev_addr)) {
      mesa_loge("ventus-vk: driver bridge failed to upload kernel metadata");
      goto out;
   }

   if (continuation_size &&
       !vtvk_driver_alloc_copy(&driver, NULL, continuation_size,
                               &continuation_dev_addr)) {
      mesa_loge("ventus-vk: driver bridge failed to allocate CPS arena");
      goto out;
   }
   args.continuation_arena_addr = continuation_dev_addr;

   pds_image.assign(pds_size, 0);
   if (continuation_dev_addr) {
      const uint64_t header_offset = VT_RT_ABI_PDS_BASE_BYTES +
         VT_RT_ABI_CPS_HEADER_BASE_BYTES +
         VT_RT_CPS_FRAME_BASE * VT_RT_ABI_WORD_BYTES;
      const uint64_t aligned = header_offset & ~3ull;
      for (uint64_t context = 0; context < pds_resident_lanes; context++) {
         const uint64_t byte_offset = aligned * pds_resident_lanes + context * 4;
         if (byte_offset + 4 > pds_image.size()) {
            mesa_loge("ventus-vk: CPS frame pointer exceeds PDS allocation");
            goto out;
         }
         const uint32_t frame_base = (uint32_t)(continuation_dev_addr +
            context * info->continuation_frame_size);
         memcpy(pds_image.data() + byte_offset, &frame_base, sizeof(frame_base));
      }
   }
   if (!vtvk_driver_alloc_copy(&driver, pds_image.data(), pds_image.size(),
                               &pds_dev_addr)) {
      mesa_loge("ventus-vk: driver bridge failed to allocate PDS");
      goto out;
   }
   if (driver.vt_copy_to_dev(driver.device, args_dev_addr, &args, sizeof(args),
                             0, 0) != 0) {
      mesa_loge("ventus-vk: driver bridge failed to update dispatch args");
      goto out;
   }

   if (!kernel_path || !kernel_path[0]) {
      mesa_logi("ventus-vk: driver bridge staged device memory "
                "tlas=0x%" PRIx64 " image=0x%" PRIx64
                " camera=0x%" PRIx64 " descriptors=0x%" PRIx64
                " args=0x%" PRIx64
                " pds=0x%" PRIx64 "; VENTUS_VK_RAYGEN_ELF is not set",
                tlas_dev_addr, image_dev_addr, camera_dev_addr,
                descriptor_table_dev_addr, args_dev_addr, pds_dev_addr);
      ok = true;
      staged_without_kernel = true;
      goto out;
   }

   if (driver.vt_upload_kernel_file(driver.device, kernel_path, 0) != 0) {
      mesa_loge("ventus-vk: driver bridge vt_upload_kernel_file failed");
      goto out;
   }

   kernel.kernel_id = 0;
   kernel.kernel_size[0] = raygen_wg_x;
   kernel.kernel_size[1] = raygen_wg_y;
   kernel.kernel_size[2] = raygen_wg_z;
   kernel.wf_size = vtvk_rt_lanes_per_warp;
   kernel.wg_size = 1;
   kernel.metaDataBaseAddr = metadata_dev_addr;
   kernel.ldsSize = 0;
   kernel.pdsSize = info->per_lane_scratch_size;
   kernel.pdsBaseAddr = pds_dev_addr;
   kernel.num_thread_global[0] = info->width;
   kernel.num_thread_global[1] = info->height;
   kernel.num_thread_global[2] = info->depth;
   kernel.num_thread_local[0] = vtvk_rt_lanes_per_warp;
   kernel.num_thread_local[1] = 1;
   kernel.num_thread_local[2] = 1;
   kernel.kernel_name = "ventus_vk_raygen";
   kernel.pdsResidentWgCount = vtvk_rt_resident_warps;

   if (driver.vt_start(driver.device, &kernel, 0) != 0 ||
       driver.vt_ready_wait(driver.device, 60 * 60 * 1000) != 0) {
      mesa_loge("ventus-vk: driver bridge execution failed");
      goto out;
   }

   if (driver.vt_copy_from_dev(driver.device, image_dev_addr,
                               info->storage_image_data, image_bytes, 0, 0) != 0) {
      mesa_loge("ventus-vk: driver bridge failed to copy storage image back");
      goto out;
   }

   if (vtvk_bridge_env_enabled("VENTUS_VK_PROBE_LOG") && image_bytes >= 16) {
      uint32_t pixels[4];
      memcpy(pixels, info->storage_image_data, sizeof(pixels));
      mesa_logi("ventus-vk: driver bridge image probe pixels "
                "%08x %08x %08x %08x",
                pixels[0], pixels[1], pixels[2], pixels[3]);
   }

   if (!vtvk_driver_dump_image_ppm(dump_image, info->storage_image_data,
                                   image_width, image_height,
                                   info->image_format))
      goto out;

   mesa_logi("ventus-vk: driver bridge executed raygen kernel %s",
             kernel_path);
   ok = true;

out:
   vtvk_driver_close(&driver);
   if (!ok)
      bridge_failed = true;
   return ok;
}
