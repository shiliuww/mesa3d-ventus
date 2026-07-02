#include "vtvk_private.h"

#include "util/u_debug.h"
#include "vk_common_entrypoints.h"
#include "vtvk_entrypoints.h"
#include "wsi_common.h"
#include "wsi_common_entrypoints.h"

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_EnumerateInstanceVersion(uint32_t *pApiVersion)
{
   uint32_t version_override = vk_get_version_override();
   *pApiVersion = version_override ? version_override :
      VK_MAKE_API_VERSION(0, 1, 3, VK_HEADER_VERSION);

   return VK_SUCCESS;
}

static const struct vk_instance_extension_table vtvk_instance_extensions = {
   .KHR_device_group_creation = true,
   .KHR_external_fence_capabilities = true,
   .KHR_external_memory_capabilities = true,
   .KHR_external_semaphore_capabilities = true,
   .KHR_get_physical_device_properties2 = true,
   .KHR_surface = true,
   .EXT_debug_report = true,
   .EXT_debug_utils = true,
#ifndef VK_USE_PLATFORM_WIN32_KHR
   .EXT_headless_surface = true,
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
   .KHR_wayland_surface = true,
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
   .KHR_xcb_surface = true,
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
   .KHR_xlib_surface = true,
#endif
};

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_EnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                      VkLayerProperties *pProperties)
{
   *pPropertyCount = 0;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_EnumerateInstanceExtensionProperties(const char *pLayerName,
                                          uint32_t *pPropertyCount,
                                          VkExtensionProperties *pProperties)
{
   if (pLayerName)
      return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);

   return vk_enumerate_instance_extension_properties(
      &vtvk_instance_extensions, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkInstance *pInstance)
{
   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);

   if (!pAllocator)
      pAllocator = vk_default_allocator();

   struct vtvk_instance *instance =
      vk_zalloc(pAllocator, sizeof(*instance), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!instance)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_instance_dispatch_table dispatch_table;
   vk_instance_dispatch_table_from_entrypoints(
      &dispatch_table, &vtvk_instance_entrypoints, true);
   vk_instance_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_instance_entrypoints, false);

   VkResult result =
      vk_instance_init(&instance->vk, &vtvk_instance_extensions,
                       &dispatch_table, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, instance);
      return vk_error(NULL, result);
   }

   instance->vk.physical_devices.enumerate = vtvk_enumerate_physical_devices;
   instance->vk.physical_devices.destroy = vtvk_destroy_physical_device;

   *pInstance = vtvk_instance_to_handle(instance);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyInstance(VkInstance _instance,
                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_instance, instance, _instance);

   if (!instance)
      return;

   vk_instance_finish(&instance->vk);
   vk_free(&instance->vk.alloc, instance);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateHeadlessSurfaceEXT(VkInstance instance,
                              const VkHeadlessSurfaceCreateInfoEXT *pCreateInfo,
                              const VkAllocationCallbacks *pAllocator,
                              VkSurfaceKHR *pSurface)
{
   return wsi_CreateHeadlessSurfaceEXT(instance, pCreateInfo, pAllocator,
                                       pSurface);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vtvk_GetInstanceProcAddr(VkInstance _instance, const char *pName)
{
   VK_FROM_HANDLE(vtvk_instance, instance, _instance);
   return vk_instance_get_proc_addr(instance ? &instance->vk : NULL,
                                    &vtvk_instance_entrypoints,
                                    pName);
}

PUBLIC
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName)
{
   return vtvk_GetInstanceProcAddr(instance, pName);
}
