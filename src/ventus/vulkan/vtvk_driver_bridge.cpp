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
#include "ventus/common/ventus_rt_address.h"
#include "ventus/common/ventus_rt_bvh.h"
#include "ventus/common/ventus_kernel_metadata.h"
#include "ventus/compiler/vt_nir.h"

static_assert(VENTUS_RT_RESIDENT_WARPS > 0,
              "Ventus RT requires at least one resident warp");

static constexpr uint64_t vtvk_rt_resident_warps =
   VENTUS_RT_RESIDENT_WARPS;
static constexpr uint64_t vtvk_rt_lanes_per_warp = 32;

static_assert(sizeof(struct vtvk_rt_descriptor_record) ==
              VT_NIR_RT_DESCRIPTOR_RECORD_SIZE,
              "RT descriptor ABI size mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, continuation_arena_addr) ==
              VT_NIR_RT_DISPATCH_ARG_CONTINUATION_ARENA_ADDR,
              "RT CPS arena argument ABI mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, continuation_frame_stride) ==
              VT_NIR_RT_DISPATCH_ARG_CONTINUATION_FRAME_STRIDE,
              "RT CPS frame stride argument ABI mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, continuation_context_count) ==
              VT_NIR_RT_DISPATCH_ARG_CONTINUATION_CONTEXT_COUNT,
              "RT CPS context count argument ABI mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, global_queue_base) ==
              VT_NIR_RT_DISPATCH_ARG_GLOBAL_QUEUE_BASE,
              "RT global queue base argument ABI mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, global_cps_arena_addr) ==
              VT_NIR_RT_DISPATCH_ARG_GLOBAL_CPS_ARENA_ADDR,
              "RT global CPS arena argument ABI mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, global_completion_base) ==
              VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_BASE,
              "RT global completion base argument ABI mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, global_completion_capacity) ==
              VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_CAPACITY,
              "RT global completion capacity argument ABI mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, global_ray_ref_list_base) ==
              VT_NIR_RT_DISPATCH_ARG_GLOBAL_RAY_REF_LIST_BASE,
              "RT global ray-ref list argument ABI mismatch");
static_assert(offsetof(struct vtvk_rt_dispatch_args, global_completion_field_stride) ==
              VT_NIR_RT_DISPATCH_ARG_GLOBAL_COMPLETION_FIELD_STRIDE,
              "RT global completion stride argument ABI mismatch");
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

/* Keep this private mirror byte-for-byte compatible with the optional Spike
 * driver C ABI.  It is resolved only for the global-wavefront profile. */
struct vtvk_driver_rt_global_consume_info {
   uint64_t queue_base;
   uint64_t completion_base;
   uint64_t hit_attribute_base;
   uint64_t miss_sbt_base;
   uint64_t hit_sbt_base;
   uint32_t capacity_rays;
   uint32_t hit_attribute_stride_bytes;
   uint32_t max_batch_rays;
   uint64_t miss_sbt_stride_bytes;
   uint64_t hit_sbt_stride_bytes;
};

struct vtvk_driver_rt_resume_request {
   uint64_t payload_address;
   uint32_t ray_ref;
   uint32_t completed_stage;
   uint32_t callback_group;
   uint32_t cps_frame;
   uint32_t cps_stack_size;
   uint32_t continuation_id;
   uint32_t launch_id_x;
   uint32_t launch_id_y;
   uint32_t launch_id_z;
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
   int (*vt_rt_consume_global)(
      void *device, const struct vtvk_driver_rt_global_consume_info *info,
      uint32_t *out_request_count);
   int (*vt_rt_get_resume_requests)(
      void *device, struct vtvk_driver_rt_resume_request *requests,
      uint32_t capacity, uint32_t *out_request_count);
};

struct vtvk_driver_upload {
   uint64_t dev_addr;
   uint64_t size;
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

static bool
vtvk_bridge_env_u32(const char *name, uint32_t *out_value)
{
   const char *value = getenv(name);
   if (!value || !value[0] || !out_value)
      return false;

   char *end = NULL;
   const unsigned long parsed = strtoul(value, &end, 0);
   if (*end || parsed > UINT32_MAX)
      return false;
   *out_value = (uint32_t)parsed;
   return true;
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

static bool
vtvk_bridge_require_rv32_range(const char *label, uint64_t addr, uint64_t size)
{
   if (ventus_rt_device_range_fits_u32(addr, size))
      return true;

   mesa_loge("ventus-vk: RV32 execution rejects %s range base=0x%" PRIx64
             " size=0x%" PRIx64,
             label, addr, size);
   return false;
}

static bool
vtvk_bridge_checked_u32(const char *label, uint64_t value, uint32_t *out)
{
   if (!vtvk_bridge_require_rv32_range(label, value, 1))
      return false;
   *out = (uint32_t)value;
   return true;
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
      return "/home/guanys/ventus-env/driver/build/driver/rtlsim_device/librtlsim_driver.so";
   if (strcmp(backend, "cyclesim") == 0)
      return "/home/guanys/ventus-env/driver/build/driver/cyclesim_device/libcyclesim_driver.so";
   if (strcmp(backend, "auto") == 0)
      return "/home/guanys/ventus-env/driver/build/driver/auto_select/libauto_select_driver.so";

   return "/home/guanys/ventus-env/driver/build/driver/spike_device/libspike_driver.so";
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
   if (!api->lib && !getenv("VENTUS_VK_DRIVER_LIB"))
      api->lib = dlopen("/home/guanys/ventus-env/install/lib/libspike_driver.so",
                        RTLD_NOW | RTLD_LOCAL);
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
vtvk_global_callback_kernel(const struct vtvk_driver_trace_rays_info *info,
                            uint32_t stage, uint32_t group,
                            uint32_t *out_index, const char **out_path,
                            uint64_t *out_entry)
{
   uint32_t found = UINT32_MAX;
   for (uint32_t i = 0; i < info->callback_kernel_count; i++) {
      if (info->callback_kernel_stages[i] != stage ||
          info->callback_kernel_groups[i] != group)
         continue;
      if (found != UINT32_MAX) {
         mesa_loge("ventus-vk: global RT has duplicate callback kernels for "
                   "stage %u SBT group %u", stage, group);
         return false;
      }
      found = i;
   }
   if (found == UINT32_MAX || !info->callback_kernel_paths[found]) {
      mesa_loge("ventus-vk: global RT has no emitted terminal callback kernel "
                "for stage %u SBT group %u", stage, group);
      return false;
   }
   *out_index = found;
   *out_path = info->callback_kernel_paths[found];
   *out_entry = info->callback_kernel_entries[found];
   if (vtvk_bridge_env_enabled("VENTUS_VK_RT_DEBUG_CALLBACK_DISPATCH"))
      mesa_logi("ventus-vk: global RT callback stage=%u group=%u index=%u entry=0x%" PRIx64
                " path=%s", stage, group, found, *out_entry, *out_path);
   return true;
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
   uint64_t mailbox_dev_addr = 0;
   uint64_t global_queue_dev_addr = 0;
   uint64_t global_cps_dev_addr = 0;
   uint64_t global_completion_dev_addr = 0;
   uint64_t global_hit_attribute_dev_addr = 0;
   uint32_t image_width = info->image_extent.width ? info->image_extent.width : info->width;
   uint32_t image_height = info->image_extent.height ? info->image_extent.height : info->height;
   uint32_t image_depth = info->image_extent.depth ? info->image_extent.depth : 1;
   uint32_t image_bpp = vtvk_bridge_format_bytes_per_pixel(info->image_format);
   uint64_t image_bytes =
      (uint64_t)image_width * image_height * image_depth * image_bpp;
   struct vtvk_rt_dispatch_args args = {};
   std::vector<struct vtvk_rt_descriptor_record> descriptor_table;
   std::vector<uint8_t> pds_image;
   uint64_t raygen_wg_x = DIV_ROUND_UP(info->width, 32);
   uint64_t raygen_wg_y = MAX2(info->height, 1u);
   uint64_t raygen_wg_z = MAX2(info->depth, 1u);
   uint64_t pds_resident_lanes =
      vtvk_rt_resident_warps * vtvk_rt_lanes_per_warp;
   const bool wavefront_mirror = vtvk_bridge_env_enabled("VENTUS_VK_RT_WAVEFRONT_MIRROR");
   const bool wavefront_global = vtvk_bridge_env_enabled("VENTUS_VK_RT_WAVEFRONT_GLOBAL");
   const bool wavefront_profile = wavefront_mirror || wavefront_global;
   uint64_t pds_size =
      wavefront_global ? 0 :
      vtvk_bridge_align_u64(MAX2((uint64_t)info->per_lane_scratch_size *
                                    pds_resident_lanes,
                                 (uint64_t)64 * 1024),
                            4096);
   uint64_t continuation_size = wavefront_global ? 0 :
      (uint64_t)info->continuation_frame_size * pds_resident_lanes;
   const bool rv64 = vtvk_bridge_env_enabled("VENTUS_VK_RT_RV64");
   std::vector<uint8_t> metadata(rv64 ? (size_t)VENTUS_KNL64_MAX_SIZE :
                                       (size_t)VENTUS_KNL32_MAX_SIZE, 0);
   uint64_t kernel_entry = 0;
   uint32_t work_dim = 0;
   static_assert(VT_RT_SUBMISSION_MAILBOX_SIZE_BYTES <= UINT32_MAX,
                 "RT mailbox stride must fit the dispatch ABI");
   static_assert(VT_RT_SUBMISSION_MAILBOX_SIZE_BYTES % sizeof(uint32_t) == 0,
                 "RT mailbox fields require a u32-aligned stride");
   /* This is an array of wire records, not a field-major queue plane.  Every
    * field is a u32, so use the exact ABI size (140 B); the old 128-B stride
    * aliased the next record's header. */
   const uint64_t mailbox_stride = VT_RT_SUBMISSION_MAILBOX_SIZE_BYTES;
   uint64_t mailbox_count = 0;
   uint32_t global_queue_capacity = 0;
   struct vt_rt_global_queue_layout global_queue_layout = {};
   struct vt_rt_global_completion_layout global_completion_layout = {};
   struct vtvk_driver_kernel_metadata kernel = {};
   const char *dump_image = getenv("VENTUS_VK_DUMP_IMAGE");

   if (wavefront_global &&
       (!vtvk_driver_resolve(driver.lib, "vt_rt_consume_global",
                             (void **)&driver.vt_rt_consume_global) ||
        !vtvk_driver_resolve(driver.lib, "vt_rt_get_resume_requests",
                             (void **)&driver.vt_rt_get_resume_requests)))
      goto out;

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
   args.scratch_stride = wavefront_global ? 0 : info->per_lane_scratch_size;
   args.continuation_frame_stride = info->continuation_frame_size;
   args.continuation_context_count = wavefront_global ? 0 : pds_resident_lanes;
   args.submission_mailbox_stride = wavefront_profile ? mailbox_stride : 0;
   args.submission_mailbox_generation = wavefront_profile ? 1 : 0;
   args.descriptor_table_addr = descriptor_table_dev_addr;
   args.descriptor_count = descriptor_table.size();
   args.descriptor_record_stride = sizeof(struct vtvk_rt_descriptor_record);
   vtvk_driver_log_u32_words("dispatch_args", &args, sizeof(args), 40);

   if (!vtvk_driver_alloc_copy(&driver, &args, sizeof(args), &args_dev_addr)) {
      mesa_loge("ventus-vk: driver bridge failed to upload dispatch args");
      goto out;
   }

   kernel_entry = info->kernel_entry ? info->kernel_entry : 0x80000000ull;
   work_dim = info->depth > 1 ? 3 : info->height > 1 ? 2 : 1;
   if (!rv64) {
      if (!vtvk_bridge_require_rv32_range("TLAS", tlas_dev_addr, 1) ||
          !vtvk_bridge_require_rv32_range("storage image", image_dev_addr,
                                          image_bytes) ||
          !vtvk_bridge_require_rv32_range("camera", camera_dev_addr,
                                          info->camera_range) ||
          !vtvk_bridge_require_rv32_range("descriptor table",
                                          descriptor_table_dev_addr,
                                          descriptor_table.size() *
                                             sizeof(descriptor_table[0])) ||
          !vtvk_bridge_require_rv32_range("dispatch args", args_dev_addr,
                                          sizeof(args)) ||
          !vtvk_bridge_require_rv32_range("raygen SBT", args.raygen_sbt.addr,
                                          args.raygen_sbt.size) ||
          !vtvk_bridge_require_rv32_range("miss SBT", args.miss_sbt.addr,
                                          args.miss_sbt.size) ||
          !vtvk_bridge_require_rv32_range("hit SBT", args.hit_sbt.addr,
                                          args.hit_sbt.size) ||
          !vtvk_bridge_require_rv32_range("callable SBT",
                                          args.callable_sbt.addr,
                                          args.callable_sbt.size) ||
          !vtvk_bridge_require_rv32_range("kernel entry", kernel_entry, 1))
         goto out;

      for (uint32_t i = 0; i < descriptor_table.size(); i++) {
         if (!vtvk_bridge_require_rv32_range("descriptor resource",
                                             descriptor_table[i].device_addr,
                                             descriptor_table[i].byte_size))
            goto out;
      }
   }
   if (rv64) {
      memcpy(metadata.data() + VENTUS_KNL64_ENTRY, &kernel_entry,
             sizeof(kernel_entry));
      memcpy(metadata.data() + VENTUS_KNL64_ARG_BASE, &args_dev_addr,
             sizeof(args_dev_addr));
      memcpy(metadata.data() + VENTUS_KNL64_WORK_DIM, &work_dim,
             sizeof(work_dim));
      memcpy(metadata.data() + VENTUS_KNL64_GL_SIZE_X, &info->width,
             sizeof(info->width));
      memcpy(metadata.data() + VENTUS_KNL64_GL_SIZE_Y, &info->height,
             sizeof(info->height));
      memcpy(metadata.data() + VENTUS_KNL64_GL_SIZE_Z, &info->depth,
             sizeof(info->depth));
      const uint32_t local_size_x = 32;
      memcpy(metadata.data() + VENTUS_KNL64_LC_SIZE_X, &local_size_x,
             sizeof(local_size_x));
   } else {
      uint32_t entry32 = 0;
      uint32_t args32 = 0;
      if (!vtvk_bridge_checked_u32("kernel entry", kernel_entry, &entry32) ||
          !vtvk_bridge_checked_u32("dispatch args", args_dev_addr, &args32))
         goto out;
      const uint32_t local_size_x = 32;
      memcpy(metadata.data() + VENTUS_KNL32_ENTRY, &entry32, sizeof(entry32));
      memcpy(metadata.data() + VENTUS_KNL32_ARG_BASE, &args32, sizeof(args32));
      memcpy(metadata.data() + VENTUS_KNL32_WORK_DIM, &work_dim, sizeof(work_dim));
      memcpy(metadata.data() + VENTUS_KNL32_GL_SIZE_X, &info->width,
             sizeof(info->width));
      memcpy(metadata.data() + VENTUS_KNL32_GL_SIZE_Y, &info->height,
             sizeof(info->height));
      memcpy(metadata.data() + VENTUS_KNL32_GL_SIZE_Z, &info->depth,
             sizeof(info->depth));
      memcpy(metadata.data() + VENTUS_KNL32_LC_SIZE_X, &local_size_x,
             sizeof(local_size_x));
   }

   if (!vtvk_driver_alloc_copy(&driver, metadata.data(), metadata.size(),
                               &metadata_dev_addr)) {
      mesa_loge("ventus-vk: driver bridge failed to upload kernel metadata");
      goto out;
   }
   if (!rv64 &&
       !vtvk_bridge_require_rv32_range("kernel metadata", metadata_dev_addr,
                                       metadata.size()))
      goto out;

   if (!wavefront_global && continuation_size &&
       !vtvk_driver_alloc_copy(&driver, NULL, continuation_size,
                               &continuation_dev_addr)) {
      mesa_loge("ventus-vk: driver bridge failed to allocate CPS arena");
      goto out;
   }
   args.continuation_arena_addr = continuation_dev_addr;
   if (!rv64 &&
       !vtvk_bridge_require_rv32_range("legacy CPS arena", continuation_dev_addr,
                                       continuation_size))
      goto out;

   /* Both the mirror and PDS-free producer use the same global queue ABI.
    * Only the latter must not allocate a legacy slot/CPS backing store. */
   if (wavefront_profile) {
      if (info->depth != 1 ||
          !vt_nir_rt_global_queue_capacity(info->width, info->height,
                                            info->max_ray_recursion_depth,
                                            3, &global_queue_capacity)) {
         mesa_loge("ventus-vk: unsupported wavefront queue dimensions "
                   "%ux%ux%u max_depth=%u",
                   info->width, info->height, info->depth,
                   info->max_ray_recursion_depth);
         goto out;
      }
      global_queue_layout = vt_nir_rt_global_queue_layout(global_queue_capacity);
      global_completion_layout =
         vt_nir_rt_global_completion_layout(global_queue_capacity);
      if (!global_queue_layout.total_size_bytes ||
          !global_completion_layout.total_size_bytes) {
         mesa_loge("ventus-vk: wavefront queue layout exceeds the ABI limit");
         goto out;
      }

      std::vector<uint8_t> global_queue(global_queue_layout.total_size_bytes, 0);
      const uint32_t generation = 1;
      memcpy(global_queue.data() +
                VT_RT_GLOBAL_QUEUE_CAPACITY_RAYS * sizeof(uint32_t),
             &global_queue_capacity, sizeof(global_queue_capacity));
      memcpy(global_queue.data() +
                VT_RT_GLOBAL_QUEUE_GENERATION * sizeof(uint32_t),
             &generation, sizeof(generation));
      if (!vtvk_driver_alloc_copy(&driver, global_queue.data(),
                                  global_queue.size(),
                                  &global_queue_dev_addr)) {
         mesa_loge("ventus-vk: driver bridge failed to allocate global ray queue");
         goto out;
      }
      if (!rv64 &&
          !vtvk_bridge_require_rv32_range("global ray queue", global_queue_dev_addr,
                                          global_queue.size()))
         goto out;

      std::vector<uint8_t> global_completion(
         global_completion_layout.total_size_bytes, 0);
      if (!vtvk_driver_alloc_copy(&driver, global_completion.data(),
                                  global_completion.size(),
                                  &global_completion_dev_addr)) {
         mesa_loge("ventus-vk: driver bridge failed to allocate global completion plane");
         goto out;
      }
      if (!rv64 &&
          !vtvk_bridge_require_rv32_range("global completion plane",
                                          global_completion_dev_addr,
                                          global_completion.size()))
         goto out;

      const uint64_t global_hit_attribute_size =
         (uint64_t)global_queue_capacity * 2 * sizeof(uint32_t);
      if (!vtvk_driver_alloc_copy(&driver, NULL, global_hit_attribute_size,
                                  &global_hit_attribute_dev_addr)) {
         mesa_loge("ventus-vk: driver bridge failed to allocate global hit attributes");
         goto out;
      }
      if (!rv64 &&
          !vtvk_bridge_require_rv32_range("global hit attributes",
                                          global_hit_attribute_dev_addr,
                                          global_hit_attribute_size))
         goto out;

      const uint64_t global_cps_size =
         (uint64_t)info->continuation_frame_size * global_queue_capacity;
      if (global_cps_size &&
          !vtvk_driver_alloc_copy(&driver, NULL, global_cps_size,
                                  &global_cps_dev_addr)) {
         mesa_loge("ventus-vk: driver bridge failed to allocate global CPS arena");
         goto out;
      }
      if (!rv64 &&
          !vtvk_bridge_require_rv32_range("global CPS arena", global_cps_dev_addr,
                                          global_cps_size))
         goto out;
      args.global_queue_base = global_queue_dev_addr;
      args.global_queue_capacity = global_queue_capacity;
      args.global_queue_generation = generation;
      args.global_cps_arena_addr = global_cps_dev_addr;
      args.global_cps_frame_stride = info->continuation_frame_size;
      args.max_ray_recursion_depth = info->max_ray_recursion_depth;
      args.global_completion_base = global_completion_dev_addr;
      args.global_completion_capacity = global_queue_capacity;
      args.global_completion_generation = generation;
      if (global_completion_layout.field_stride_bytes > UINT32_MAX) {
         mesa_loge("ventus-vk: global completion field stride exceeds u32 ABI");
         goto out;
      }
      args.global_completion_field_stride =
         (uint32_t)global_completion_layout.field_stride_bytes;

      if (wavefront_global &&
          !vt_nir_rt_submission_mailbox_capacity(
             vtvk_rt_resident_warps, vtvk_rt_lanes_per_warp,
             &mailbox_count)) {
         mesa_loge("ventus-vk: wavefront mailbox layout exceeds the ABI limit");
         goto out;
      }
   }

   /* The compatibility mirror publishes dense primary records.  The PDS-free
    * profile has one reusable PRIMARY/shadow/reflection/refraction mailbox
    * per resident physical lane. */
   if (wavefront_mirror)
      mailbox_count = (uint64_t)info->width * info->height *
                      MAX2(info->depth, 1u);

   if (wavefront_profile && mailbox_count > UINT64_MAX / mailbox_stride) {
      mesa_loge("ventus-vk: wavefront mailbox allocation size overflow");
      goto out;
   }
   if (wavefront_profile &&
       !vtvk_driver_alloc_copy(&driver, NULL, mailbox_count * mailbox_stride,
                               &mailbox_dev_addr)) {
      mesa_loge("ventus-vk: driver bridge failed to allocate mailbox pool");
      goto out;
   }
   args.submission_mailbox_base = mailbox_dev_addr;
   if (!rv64 &&
       !vtvk_bridge_require_rv32_range("submission mailbox pool", mailbox_dev_addr,
                                       wavefront_profile ?
                                          mailbox_count * mailbox_stride : 0))
      goto out;

   if (!wavefront_global) {
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
            uint32_t frame_base = 0;
            if (!vtvk_bridge_checked_u32("legacy CPS frame",
                                        continuation_dev_addr +
                                           context * info->continuation_frame_size,
                                        &frame_base))
               goto out;
            memcpy(pds_image.data() + byte_offset, &frame_base, sizeof(frame_base));
         }
      }
      if (!vtvk_driver_alloc_copy(&driver, pds_image.data(), pds_image.size(),
                                  &pds_dev_addr)) {
         mesa_loge("ventus-vk: driver bridge failed to allocate PDS");
         goto out;
      }
      if (!rv64 && !vtvk_bridge_require_rv32_range("legacy PDS", pds_dev_addr,
                                                    pds_image.size()))
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
   kernel.pdsSize = wavefront_global ? 0 : info->per_lane_scratch_size;
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

   if (wavefront_global) {
      if (vtvk_bridge_env_enabled("VENTUS_VK_RT_DEBUG_GLOBAL")) {
         uint32_t queue_header[VT_RT_GLOBAL_QUEUE_HEADER_WORD_COUNT] = {};
         if (driver.vt_copy_from_dev(driver.device, global_queue_dev_addr,
                                     queue_header, sizeof(queue_header),
                                     0, 0) != 0) {
            mesa_loge("ventus-vk: failed to inspect global RT queue header");
            goto out;
         }
         mesa_logi("ventus-vk: global RT queue capacity=%u tail=%u head=%u "
                   "overflow=%u generation=%u",
                   queue_header[VT_RT_GLOBAL_QUEUE_CAPACITY_RAYS],
                   queue_header[VT_RT_GLOBAL_QUEUE_RESERVE_TAIL],
                   queue_header[VT_RT_GLOBAL_QUEUE_CONSUME_HEAD],
                   queue_header[VT_RT_GLOBAL_QUEUE_OVERFLOW_FLAG],
                   queue_header[VT_RT_GLOBAL_QUEUE_GENERATION]);
      }
      const struct vtvk_driver_rt_global_consume_info consume = {
         .queue_base = global_queue_dev_addr,
         .completion_base = global_completion_dev_addr,
         .hit_attribute_base = global_hit_attribute_dev_addr,
         .miss_sbt_base = args.miss_sbt.addr,
         .hit_sbt_base = args.hit_sbt.addr,
         .capacity_rays = global_queue_capacity,
         .hit_attribute_stride_bytes = 2 * sizeof(uint32_t),
         .max_batch_rays = vtvk_rt_lanes_per_warp,
         .miss_sbt_stride_bytes = args.miss_sbt.stride,
         .hit_sbt_stride_bytes = args.hit_sbt.stride,
      };
      auto dispatch_compact = [&](const char *path, uint64_t entry,
                                  const std::vector<uint32_t> &ray_refs,
                                  const char *name,
                                  uint32_t dispatch_generation) -> bool {
         if (ray_refs.empty())
            return true;

         uint64_t list_dev_addr = 0;
         const uint64_t list_size = ray_refs.size() * sizeof(ray_refs[0]);
         if (!vtvk_driver_alloc_copy(&driver, ray_refs.data(), list_size,
                                     &list_dev_addr) ||
             (!rv64 && !vtvk_bridge_require_rv32_range(
                         "global RT ray-ref list", list_dev_addr, list_size)))
            return false;

         args.global_ray_ref_list_base = list_dev_addr;
         args.global_ray_ref_count = ray_refs.size();
         args.global_ray_ref_generation = dispatch_generation;
         if (driver.vt_copy_to_dev(driver.device, args_dev_addr, &args,
                                   sizeof(args), 0, 0) != 0)
            return false;

         if (rv64) {
            memcpy(metadata.data() + VENTUS_KNL64_ENTRY, &entry, sizeof(entry));
            memcpy(metadata.data() + VENTUS_KNL64_GL_SIZE_X,
                   &args.global_ray_ref_count, sizeof(args.global_ray_ref_count));
            const uint32_t one = 1;
            memcpy(metadata.data() + VENTUS_KNL64_GL_SIZE_Y, &one, sizeof(one));
            memcpy(metadata.data() + VENTUS_KNL64_GL_SIZE_Z, &one, sizeof(one));
         } else {
            uint32_t entry32 = 0;
            if (!vtvk_bridge_checked_u32("global RT kernel entry", entry,
                                         &entry32))
               return false;
            memcpy(metadata.data() + VENTUS_KNL32_ENTRY, &entry32, sizeof(entry32));
            memcpy(metadata.data() + VENTUS_KNL32_GL_SIZE_X,
                   &args.global_ray_ref_count, sizeof(args.global_ray_ref_count));
            const uint32_t one = 1;
            memcpy(metadata.data() + VENTUS_KNL32_GL_SIZE_Y, &one, sizeof(one));
            memcpy(metadata.data() + VENTUS_KNL32_GL_SIZE_Z, &one, sizeof(one));
         }
         if (driver.vt_copy_to_dev(driver.device, metadata_dev_addr, metadata.data(),
                                   metadata.size(), 0, 0) != 0 ||
             driver.vt_upload_kernel_file(driver.device, path, 0) != 0)
            return false;

         kernel.kernel_size[0] =
            DIV_ROUND_UP((uint64_t)args.global_ray_ref_count,
                         vtvk_rt_lanes_per_warp);
         kernel.kernel_size[1] = 1;
         kernel.kernel_size[2] = 1;
         kernel.num_thread_global[0] = args.global_ray_ref_count;
         kernel.num_thread_global[1] = 1;
         kernel.num_thread_global[2] = 1;
         kernel.kernel_name = name;
         return driver.vt_start(driver.device, &kernel, 0) == 0 &&
                driver.vt_ready_wait(driver.device, 60 * 60 * 1000) == 0;
      };

      /* Root continuations execute terminal SBT code in the resume ELF.  Read
       * this optional probe only after that resume has finished. */
      const bool debug_payload_rgb =
         vtvk_bridge_env_enabled("VENTUS_VK_RT_DEBUG_PAYLOAD_RGB");
      uint32_t debug_payload_ray_ref = 0;
      const bool debug_payload_ray_ref_enabled =
         vtvk_bridge_env_u32("VENTUS_VK_RT_DEBUG_PAYLOAD_RAY_REF",
                             &debug_payload_ray_ref);
      uint32_t debug_payload_launch_x = 0;
      uint32_t debug_payload_launch_y = 0;
      const bool debug_payload_launch_enabled =
         vtvk_bridge_env_u32("VENTUS_VK_RT_DEBUG_PAYLOAD_LAUNCH_X",
                             &debug_payload_launch_x) &&
         vtvk_bridge_env_u32("VENTUS_VK_RT_DEBUG_PAYLOAD_LAUNCH_Y",
                             &debug_payload_launch_y);
      const bool debug_trace_inputs =
         vtvk_bridge_env_enabled("VENTUS_VK_RT_DEBUG_TRACE_INPUTS");
      const auto log_trace_input = [&](uint32_t ray_ref) {
         uint32_t words[7] = {};
         const uint32_t fields[] = {
            VT_RT_GLOBAL_TRACE_ORIGIN_X, VT_RT_GLOBAL_TRACE_ORIGIN_Y,
            VT_RT_GLOBAL_TRACE_ORIGIN_Z, VT_RT_GLOBAL_TRACE_DIRECTION_X,
            VT_RT_GLOBAL_TRACE_DIRECTION_Y, VT_RT_GLOBAL_TRACE_DIRECTION_Z,
            VT_RT_GLOBAL_TRACE_TMAX,
         };
         for (uint32_t i = 0; i < ARRAY_SIZE(fields); ++i) {
            const uint64_t address = global_queue_dev_addr +
               VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES +
               (uint64_t)fields[i] * global_queue_layout.field_stride_bytes +
               (uint64_t)ray_ref * sizeof(uint32_t);
            if (driver.vt_copy_from_dev(driver.device, address, &words[i],
                                        sizeof(words[i]), 0, 0) != 0) {
               mesa_loge("ventus-vk: global RT trace-input probe read failed "
                         "ray_ref=%u field=%u", ray_ref, fields[i]);
               return;
            }
         }
         float values[7];
         memcpy(values, words, sizeof(values));
         mesa_logi("ventus-vk: global RT trace-input ray_ref=%u "
                   "origin=(%.6f,%.6f,%.6f) direction=(%.6f,%.6f,%.6f) "
                   "tmax=%.6f",
                   ray_ref, values[0], values[1], values[2], values[3],
                   values[4], values[5], values[6]);
      };
      const auto log_payload_rgb =
         [&](const std::vector<struct vtvk_driver_rt_resume_request> &requests,
             uint32_t phase) {
            if (!debug_payload_rgb)
               return;
            uint32_t stage_samples[2] = {};
            for (uint32_t i = 0; i < requests.size(); ++i) {
               const auto &request = requests[i];
               if (debug_payload_ray_ref_enabled &&
                   request.ray_ref != debug_payload_ray_ref)
                  continue;
               if (debug_payload_launch_enabled &&
                   (request.launch_id_x != debug_payload_launch_x ||
                    request.launch_id_y != debug_payload_launch_y))
                  continue;
               if (request.completed_stage > 1 ||
                   stage_samples[request.completed_stage]++ >= 8)
                  continue;
               /* The reflections payload is eight words (color, distance,
                * normal, reflector).  Keep the complete small payload in
                * this opt-in probe: RGB alone cannot distinguish a callback
                * that wrote a black material from one that did not publish
                * its hit distance/normal at all. */
               uint32_t bits[8] = {};
               if (driver.vt_copy_from_dev(driver.device, request.payload_address,
                                           bits, sizeof(bits), 0, 0) != 0) {
                  mesa_loge("ventus-vk: global RT payload probe read failed "
                            "phase=%u ray_ref=%u", phase, request.ray_ref);
                  continue;
               }
               float values[8];
               memcpy(values, bits, sizeof(values));
               uint32_t primitive_id = 0;
               float hit_t = -1.0f;
               uint64_t hit_attribute_address = 0;
               float hit_attributes[2] = {};
               if (request.completed_stage == 1) {
                  const uint32_t fields[] = {
                     VT_RT_GLOBAL_COMPLETION_COMMITTED_HIT_BASE +
                        VT_RT_HIT_RECORD_HIT_T,
                     VT_RT_GLOBAL_COMPLETION_COMMITTED_HIT_BASE +
                        VT_RT_HIT_RECORD_PRIMITIVE_ID,
                     VT_RT_GLOBAL_COMPLETION_HIT_ATTRIBUTE_ADDR_LO,
                     VT_RT_GLOBAL_COMPLETION_HIT_ATTRIBUTE_ADDR_HI,
                  };
                  uint32_t hit_words[4] = {};
                  for (uint32_t field = 0; field < ARRAY_SIZE(fields); ++field) {
                     const uint64_t address = global_completion_dev_addr +
                        VT_RT_GLOBAL_QUEUE_HEADER_SIZE_BYTES +
                        (uint64_t)fields[field] *
                           global_completion_layout.field_stride_bytes +
                        (uint64_t)request.ray_ref * sizeof(uint32_t);
                     if (driver.vt_copy_from_dev(driver.device, address,
                                                 &hit_words[field],
                                                 sizeof(hit_words[field]), 0, 0) != 0)
                        continue;
                  }
                  memcpy(&hit_t, &hit_words[0], sizeof(hit_t));
                  primitive_id = hit_words[1];
                  hit_attribute_address = (uint64_t)hit_words[2] |
                                          ((uint64_t)hit_words[3] << 32);
                  if (hit_attribute_address &&
                      driver.vt_copy_from_dev(driver.device,
                                              hit_attribute_address,
                                              hit_attributes,
                                              sizeof(hit_attributes), 0, 0) != 0) {
                     mesa_loge("ventus-vk: global RT hit-attribute probe read failed "
                               "ray_ref=%u", request.ray_ref);
                  }
               }
               mesa_logi("ventus-vk: global RT payload post-callback "
                         "phase=%u ray_ref=%u stage=%u addr=%" PRIx64
                         " payload=(%.6f,%.6f,%.6f; d=%.6f; n=%.6f,%.6f,%.6f; r=%.6f) "
                         "bits=(%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x) "
                         "frame=%u stack=%u continuation=%u launch=(%u,%u,%u)",
                         phase, request.ray_ref, request.completed_stage,
                         request.payload_address, values[0], values[1], values[2],
                         values[3], values[4], values[5], values[6], values[7],
                         bits[0], bits[1], bits[2], bits[3], bits[4], bits[5],
                         bits[6], bits[7], request.cps_frame,
                         request.cps_stack_size, request.continuation_id,
                         request.launch_id_x, request.launch_id_y,
                         request.launch_id_z);
               if (request.completed_stage == 1)
                  mesa_logi("ventus-vk: global RT committed hit ray_ref=%u "
                            "primitive=%u t=%.6f attr_addr=%" PRIx64
                            " attr=(%.6f,%.6f)", request.ray_ref,
                            primitive_id, hit_t, hit_attribute_address,
                            hit_attributes[0], hit_attributes[1]);
            }
         };

      bool complete = false;
      const uint32_t max_phases = MAX2(info->max_ray_recursion_depth, 1u);
      for (uint32_t phase = 0; phase <= max_phases; phase++) {
         uint32_t request_count = 0;
         if (driver.vt_rt_consume_global(driver.device, &consume,
                                         &request_count) != 0) {
            mesa_loge("ventus-vk: global RT queue consumer failed");
            goto out;
         }
         if (!request_count) {
            if (phase == 0) {
               mesa_loge("ventus-vk: global RT producer published no traversal "
                         "requests; refusing to copy an unwritten image");
               goto out;
            }
            complete = true;
            break;
         }

         std::vector<struct vtvk_driver_rt_resume_request> requests(request_count);
         uint32_t copied_count = 0;
         if (driver.vt_rt_get_resume_requests(
                driver.device, requests.data(), request_count, &copied_count) != 0 ||
             copied_count != request_count) {
            mesa_loge("ventus-vk: global RT queue consumer request export failed");
            goto out;
         }
         if (debug_trace_inputs && phase == 0) {
            for (const auto &request : requests)
               if (request.ray_ref < 16)
                  log_trace_input(request.ray_ref);
         }
         std::vector<uint32_t> miss_refs;
         std::vector<uint32_t> hit_refs;
         std::vector<std::vector<uint32_t>> callback_refs(
            info->callback_kernel_count);
         std::vector<std::vector<uint32_t>> miss_resume_refs(
            info->resume_kernel_count);
         std::vector<std::vector<uint32_t>> hit_resume_refs(
            info->resume_kernel_count);
         for (const auto &request : requests) {
            uint32_t callback_stage = 0;
            if (request.completed_stage == 0) {
               miss_refs.push_back(request.ray_ref);
               callback_stage = MESA_SHADER_MISS;
               if (request.continuation_id < miss_resume_refs.size())
                  miss_resume_refs[request.continuation_id].push_back(
                     request.ray_ref);
            } else if (request.completed_stage == 1) {
               hit_refs.push_back(request.ray_ref);
               callback_stage = MESA_SHADER_CLOSEST_HIT;
               if (request.continuation_id < hit_resume_refs.size())
                  hit_resume_refs[request.continuation_id].push_back(
                     request.ray_ref);
            } else {
               mesa_loge("ventus-vk: global RT produced unsupported candidate stage %u",
                         request.completed_stage);
               goto out;
            }
            if (request.continuation_id >= miss_resume_refs.size()) {
               mesa_loge("ventus-vk: global RT continuation %u has no resume ELF",
                         request.continuation_id);
               goto out;
            }
            if (info->callback_kernel_count) {
               uint32_t callback_index = UINT32_MAX;
               const char *callback_path = NULL;
               uint64_t callback_entry = 0;
               if (!vtvk_global_callback_kernel(info, callback_stage,
                                                request.callback_group,
                                                &callback_index, &callback_path,
                                                &callback_entry))
                  goto out;
               callback_refs[callback_index].push_back(request.ray_ref);
            }
         }

         if (!hit_refs.empty() && debug_payload_rgb) {
            /* Poison the complete hit payload before the callback.  This
             * opt-in diagnostic separates "callback wrote zero" from
             * "callback never reached its global stores" without changing
             * normal global-wavefront execution. */
            const uint32_t poison[8] = {
               0x7fc01234u, 0x7fc01235u, 0x7fc01236u, 0x7fc01237u,
               0x7fc01238u, 0x7fc01239u, 0x7fc0123au, 0x7fc0123bu,
            };
            for (const auto &request : requests) {
               if (request.completed_stage != 1)
                  continue;
               if (driver.vt_copy_to_dev(driver.device, request.payload_address,
                                         poison, sizeof(poison), 0, 0) != 0) {
                  mesa_loge("ventus-vk: global RT hit payload poison failed "
                            "ray_ref=%u", request.ray_ref);
                  goto out;
               }
            }
         }
         for (uint32_t callback = 0; callback < callback_refs.size(); callback++) {
            const std::vector<uint32_t> &refs = callback_refs[callback];
            if (refs.empty())
               continue;
            const uint32_t stage = info->callback_kernel_stages[callback];
            const char *name = stage == MESA_SHADER_MISS ?
               "ventus_vk_miss" : "ventus_vk_closest_hit";
            if (!dispatch_compact(info->callback_kernel_paths[callback],
                                  info->callback_kernel_entries[callback], refs,
                                  name, phase + 1)) {
               mesa_loge("ventus-vk: global RT callback dispatch failed "
                         "stage=%u group=%u", stage,
                         info->callback_kernel_groups[callback]);
               goto out;
            }
         }
         /* Completion stage is uniform within each terminal callback.  Keep
          * it uniform for its resume batch as well: mixing miss and hit
          * records here immediately re-diverges the raygen post-trace path,
          * even though those records have no data dependency on one another.
          */
         for (uint32_t continuation = 0;
              continuation < miss_resume_refs.size(); continuation++) {
            if (!dispatch_compact(info->resume_kernel_paths[continuation],
                                  info->resume_kernel_entries[continuation],
                                  miss_resume_refs[continuation],
                                  "ventus_vk_resume_miss", phase + 1)) {
               mesa_loge("ventus-vk: global RT miss resume %u dispatch failed",
                         continuation);
               goto out;
            }
         }
         for (uint32_t continuation = 0;
              continuation < hit_resume_refs.size(); continuation++) {
            if (!dispatch_compact(info->resume_kernel_paths[continuation],
                                  info->resume_kernel_entries[continuation],
                                  hit_resume_refs[continuation],
                                  "ventus_vk_resume_hit",
                                  phase + 1)) {
               mesa_loge("ventus-vk: global RT hit resume %u dispatch failed",
                         continuation);
               goto out;
            }
         }
         log_payload_rgb(requests, phase + 1);
      }
      if (!complete) {
         mesa_loge("ventus-vk: global RT exceeded recursion-phase dispatch bound");
         goto out;
      }
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
