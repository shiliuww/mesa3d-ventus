#include "vtvk_private.h"

#include "util/list.h"
#include "vk_format.h"
#include "vk_util.h"
#include "vtvk_entrypoints.h"

static void
vtvk_init_properties(struct vtvk_physical_device *pdev)
{
   struct vk_properties *p = &pdev->vk.properties;

   p->apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);
   p->driverVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
   p->vendorID = VTVK_VENDOR_ID;
   p->deviceID = VTVK_DEVICE_ID;
   p->deviceType = VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
   snprintf(p->deviceName, sizeof(p->deviceName), "Ventus RT Spike");

   static const uint8_t uuid[VK_UUID_SIZE] = {
      'V', 'e', 'n', 't', 'u', 's', '-', 'R',
      'T', '-', 'S', 'p', 'i', 'k', 'e', 0,
   };
   memcpy(p->pipelineCacheUUID, uuid, VK_UUID_SIZE);
   memcpy(p->deviceUUID, uuid, VK_UUID_SIZE);
   memcpy(p->driverUUID, uuid, VK_UUID_SIZE);

   p->maxImageDimension1D = 16384;
   p->maxImageDimension2D = 16384;
   p->maxImageDimension3D = 2048;
   p->maxImageDimensionCube = 16384;
   p->maxImageArrayLayers = 2048;
   p->maxTexelBufferElements = 1 << 27;
   p->maxUniformBufferRange = 1 << 27;
   p->maxStorageBufferRange = 1 << 30;
   p->maxPushConstantsSize = 256;
   p->maxMemoryAllocationCount = 4096;
   p->maxSamplerAllocationCount = 4096;
   p->bufferImageGranularity = 64;
   p->maxBoundDescriptorSets = 8;
   p->maxPerStageDescriptorSamplers = 1024;
   p->maxPerStageDescriptorUniformBuffers = 64;
   p->maxPerStageDescriptorStorageBuffers = 1024;
   p->maxPerStageDescriptorSampledImages = 1024;
   p->maxPerStageDescriptorStorageImages = 1024;
   p->maxPerStageResources = 4096;
   p->maxDescriptorSetSamplers = 1024;
   p->maxDescriptorSetUniformBuffers = 1024;
   p->maxDescriptorSetStorageBuffers = 1024;
   p->maxDescriptorSetSampledImages = 1024;
   p->maxDescriptorSetStorageImages = 1024;
   p->maxVertexInputAttributes = 32;
   p->maxVertexInputBindings = 32;
   p->maxVertexInputAttributeOffset = 2047;
   p->maxVertexInputBindingStride = 2048;
   p->maxVertexOutputComponents = 128;
   p->maxFragmentInputComponents = 128;
   p->maxFragmentOutputAttachments = 8;
   p->maxFragmentCombinedOutputResources = 8;
   p->maxComputeSharedMemorySize = 32768;
   p->maxComputeWorkGroupCount[0] = 65535;
   p->maxComputeWorkGroupCount[1] = 65535;
   p->maxComputeWorkGroupCount[2] = 65535;
   p->maxComputeWorkGroupInvocations = 1024;
   p->maxComputeWorkGroupSize[0] = 1024;
   p->maxComputeWorkGroupSize[1] = 1024;
   p->maxComputeWorkGroupSize[2] = 64;
   p->maxDrawIndexedIndexValue = UINT32_MAX;
   p->maxDrawIndirectCount = UINT32_MAX;
   p->maxViewports = 1;
   p->maxViewportDimensions[0] = 16384;
   p->maxViewportDimensions[1] = 16384;
   p->viewportBoundsRange[0] = -32768.0f;
   p->viewportBoundsRange[1] = 32767.0f;
   p->minMemoryMapAlignment = 64;
   p->minTexelBufferOffsetAlignment = 16;
   p->minUniformBufferOffsetAlignment = 16;
   p->minStorageBufferOffsetAlignment = 16;
   p->maxFramebufferWidth = 16384;
   p->maxFramebufferHeight = 16384;
   p->maxFramebufferLayers = 2048;
   p->framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->framebufferStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->framebufferNoAttachmentsSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->maxColorAttachments = 8;
   p->sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->sampledImageDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->sampledImageStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT;
   p->maxSampleMaskWords = 1;
   p->timestampPeriod = 1.0f;
   p->discreteQueuePriorities = 1;
   p->pointSizeRange[0] = 1.0f;
   p->pointSizeRange[1] = 1.0f;
   p->lineWidthRange[0] = 1.0f;
   p->lineWidthRange[1] = 1.0f;
   p->pointSizeGranularity = 1.0f;
   p->lineWidthGranularity = 1.0f;
   p->standardSampleLocations = true;
   p->optimalBufferCopyOffsetAlignment = 4;
   p->optimalBufferCopyRowPitchAlignment = 4;
   p->nonCoherentAtomSize = 64;

   p->maxBufferSize = 1ull << 32;
   p->maxMemoryAllocationSize = 1ull << 32;
   p->driverID = VK_DRIVER_ID_MESA_VENUS;
   snprintf(p->driverName, sizeof(p->driverName), "ventus");
   snprintf(p->driverInfo, sizeof(p->driverInfo),
            "Ventus Vulkan ICD over Spike");
   p->conformanceVersion = (VkConformanceVersion){0, 0, 0, 0};

   p->subgroupSize = 32;
   p->subgroupSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT |
                                VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                VK_SHADER_STAGE_MISS_BIT_KHR |
                                VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
   p->subgroupSupportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT;
   p->denormBehaviorIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL;
   p->roundingModeIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL;
   p->shaderSignedZeroInfNanPreserveFloat32 = true;
   p->shaderDenormPreserveFloat32 = true;
   p->shaderRoundingModeRTEFloat32 = true;
   p->shaderRoundingModeRTZFloat32 = true;
   p->shaderUniformBufferArrayNonUniformIndexingNative = true;
   p->shaderSampledImageArrayNonUniformIndexingNative = true;
   p->shaderStorageBufferArrayNonUniformIndexingNative = true;
   p->shaderStorageImageArrayNonUniformIndexingNative = true;

   p->maxGeometryCount = 1 << 20;
   p->maxInstanceCount = 1 << 20;
   p->maxPrimitiveCount = 1 << 20;
   p->maxPerStageDescriptorAccelerationStructures = 64;
   p->maxDescriptorSetAccelerationStructures = 64;
   p->minAccelerationStructureScratchOffsetAlignment = 128;
   p->shaderGroupHandleSize = 32;
   /* The global wavefront profile carries recursion in CPS frames and drains
    * one traversal generation at a time.  Advertise the depth the bridge can
    * actually retain, rather than forcing reflection shaders to a stale
    * one-trace limit. */
   p->maxRayRecursionDepth = 4;
   p->maxShaderGroupStride = 4096;
   p->shaderGroupBaseAlignment = 64;
   p->shaderGroupHandleCaptureReplaySize = 32;
   p->maxRayDispatchInvocationCount = 1 << 24;
   p->shaderGroupHandleAlignment = 32;
   p->maxRayHitAttributeSize = 32;
   p->maxTriangleCount = 1 << 20;
   p->accelerationStructureDescriptorSize = 8;
   p->storageImageDescriptorSize = 32;
   p->storageBufferDescriptorSize = 16;
   p->uniformBufferDescriptorSize = 16;
   p->combinedImageSamplerDescriptorSize = 32;
   p->samplerDescriptorSize = 16;
}

static void
vtvk_init_features(struct vtvk_physical_device *pdev)
{
   struct vk_features *f = &pdev->vk.supported_features;

   f->shaderInt16 = true;
   f->shaderInt64 = true;
   f->shaderStorageImageWriteWithoutFormat = true;
   f->shaderStorageImageReadWithoutFormat = true;
   f->variablePointers = true;
   f->variablePointersStorageBuffer = true;
   f->maintenance4 = true;
   f->maintenance5 = true;
   f->bufferDeviceAddress = true;
   f->vulkanMemoryModel = true;
   f->descriptorIndexing = true;
   f->shaderUniformBufferArrayNonUniformIndexing = true;
   f->shaderSampledImageArrayNonUniformIndexing = true;
   f->shaderStorageBufferArrayNonUniformIndexing = true;
   f->shaderStorageImageArrayNonUniformIndexing = true;
   f->shaderUniformTexelBufferArrayNonUniformIndexing = true;
   f->shaderStorageTexelBufferArrayNonUniformIndexing = true;
   f->descriptorBindingUniformBufferUpdateAfterBind = true;
   f->descriptorBindingSampledImageUpdateAfterBind = true;
   f->descriptorBindingStorageImageUpdateAfterBind = true;
   f->descriptorBindingStorageBufferUpdateAfterBind = true;
   f->descriptorBindingUniformTexelBufferUpdateAfterBind = true;
   f->descriptorBindingStorageTexelBufferUpdateAfterBind = true;
   f->descriptorBindingUpdateUnusedWhilePending = true;
   f->descriptorBindingPartiallyBound = true;
   f->descriptorBindingVariableDescriptorCount = true;
   f->runtimeDescriptorArray = true;
   f->accelerationStructure = true;
   f->accelerationStructureHostCommands = true;
   f->rayTracingPipeline = true;
   f->descriptorBindingAccelerationStructureUpdateAfterBind = true;
}

static void
vtvk_init_extensions(struct vtvk_physical_device *pdev)
{
   struct vk_device_extension_table *e = &pdev->vk.supported_extensions;

   e->KHR_acceleration_structure = true;
   e->KHR_buffer_device_address = true;
   e->KHR_deferred_host_operations = true;
   e->KHR_pipeline_library = true;
   e->KHR_ray_tracing_pipeline = true;
   e->KHR_shader_float_controls = true;
   e->KHR_spirv_1_4 = true;
   e->KHR_swapchain = true;
   e->EXT_descriptor_indexing = true;
};

VkResult
vtvk_enumerate_physical_devices(struct vk_instance *vk_instance)
{
   struct vtvk_instance *instance =
      container_of(vk_instance, struct vtvk_instance, vk);

   struct vtvk_physical_device *pdev =
      vk_zalloc(&instance->vk.alloc, sizeof(*pdev), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!pdev)
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_physical_device_dispatch_table dispatch_table;
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &vtvk_physical_device_entrypoints, true);

   VkResult result = vk_physical_device_init(&pdev->vk, &instance->vk,
                                             NULL, NULL, NULL,
                                             &dispatch_table);
   if (result != VK_SUCCESS) {
      vk_free(&instance->vk.alloc, pdev);
      return result;
   }

   vtvk_init_extensions(pdev);
   vtvk_init_features(pdev);
   vtvk_init_properties(pdev);

   pdev->memory_heaps[0] = (VkMemoryHeap){
      .size = 2ull * 1024 * 1024 * 1024,
      .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
   };
   pdev->memory_types[0] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      .heapIndex = 0,
   };
   pdev->sync_types[0] = NULL;
   pdev->vk.supported_sync_types = pdev->sync_types;

   list_addtail(&pdev->vk.link, &instance->vk.physical_devices.list);
   return VK_SUCCESS;
}

void
vtvk_destroy_physical_device(struct vk_physical_device *vk_pdev)
{
   struct vtvk_physical_device *pdev =
      container_of(vk_pdev, struct vtvk_physical_device, vk);

   vk_physical_device_finish(&pdev->vk);
   vk_free(&pdev->vk.instance->alloc, pdev);
}

VKAPI_ATTR void VKAPI_CALL
vtvk_GetPhysicalDeviceQueueFamilyProperties2(
   VkPhysicalDevice physicalDevice,
   uint32_t *pQueueFamilyPropertyCount,
   VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out,
                          pQueueFamilyProperties, pQueueFamilyPropertyCount);

   vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p) {
      p->queueFamilyProperties = (VkQueueFamilyProperties){
         .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                       VK_QUEUE_TRANSFER_BIT,
         .queueCount = 1,
         .timestampValidBits = 0,
         .minImageTransferGranularity = {1, 1, 1},
      };
   }
}

VKAPI_ATTR void VKAPI_CALL
vtvk_GetPhysicalDeviceMemoryProperties2(
   VkPhysicalDevice physicalDevice,
   VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   VK_FROM_HANDLE(vtvk_physical_device, pdev, physicalDevice);

   pMemoryProperties->memoryProperties.memoryHeapCount = 1;
   pMemoryProperties->memoryProperties.memoryHeaps[0] = pdev->memory_heaps[0];
   pMemoryProperties->memoryProperties.memoryTypeCount = 1;
   pMemoryProperties->memoryProperties.memoryTypes[0] = pdev->memory_types[0];
}

VKAPI_ATTR void VKAPI_CALL
vtvk_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice,
                                        VkFormat format,
                                        VkFormatProperties2 *pFormatProperties)
{
   memset(&pFormatProperties->formatProperties, 0,
          sizeof(pFormatProperties->formatProperties));

   if (format == VK_FORMAT_R8G8B8A8_UNORM ||
       format == VK_FORMAT_B8G8R8A8_UNORM) {
      pFormatProperties->formatProperties.linearTilingFeatures =
         VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
         VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT |
         VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
         VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
      pFormatProperties->formatProperties.optimalTilingFeatures =
         pFormatProperties->formatProperties.linearTilingFeatures;
      pFormatProperties->formatProperties.bufferFeatures =
         VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT;
   } else if (format == VK_FORMAT_D16_UNORM ||
              format == VK_FORMAT_D32_SFLOAT ||
              format == VK_FORMAT_D24_UNORM_S8_UINT ||
              format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
      pFormatProperties->formatProperties.optimalTilingFeatures =
         VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT |
         VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
         VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetPhysicalDeviceImageFormatProperties2(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
   VkImageFormatProperties2 *pImageFormatProperties)
{
   if (pImageFormatInfo->format != VK_FORMAT_R8G8B8A8_UNORM &&
       pImageFormatInfo->format != VK_FORMAT_B8G8R8A8_UNORM &&
       pImageFormatInfo->format != VK_FORMAT_D16_UNORM &&
       pImageFormatInfo->format != VK_FORMAT_D32_SFLOAT &&
       pImageFormatInfo->format != VK_FORMAT_D24_UNORM_S8_UINT &&
       pImageFormatInfo->format != VK_FORMAT_D32_SFLOAT_S8_UINT)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties){
      .maxExtent = {16384, 16384, 1},
      .maxMipLevels = 1,
      .maxArrayLayers = 1,
      .sampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .maxResourceSize = 1ull << 32,
   };
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_GetPhysicalDeviceSparseImageFormatProperties2(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceSparseImageFormatInfo2 *pFormatInfo,
   uint32_t *pPropertyCount,
   VkSparseImageFormatProperties2 *pProperties)
{
   *pPropertyCount = 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice physicalDevice,
                                        uint32_t queueFamilyIndex,
                                        VkSurfaceKHR surface,
                                        VkBool32 *pSupported)
{
   (void)physicalDevice;
   (void)surface;
   *pSupported = queueFamilyIndex == VTVK_QUEUE_FAMILY;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetPhysicalDeviceSurfaceCapabilitiesKHR(
   VkPhysicalDevice physicalDevice,
   VkSurfaceKHR surface,
   VkSurfaceCapabilitiesKHR *pSurfaceCapabilities)
{
   (void)physicalDevice;
   (void)surface;
   *pSurfaceCapabilities = (VkSurfaceCapabilitiesKHR){
      .minImageCount = 2,
      .maxImageCount = 3,
      .currentExtent = { UINT32_MAX, UINT32_MAX },
      .minImageExtent = { 1, 1 },
      .maxImageExtent = { 16384, 16384 },
      .maxImageArrayLayers = 1,
      .supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                             VK_IMAGE_USAGE_STORAGE_BIT,
   };
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice physicalDevice,
                                        VkSurfaceKHR surface,
                                        uint32_t *pSurfaceFormatCount,
                                        VkSurfaceFormatKHR *pSurfaceFormats)
{
   (void)physicalDevice;
   (void)surface;
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormatKHR, out, pSurfaceFormats,
                          pSurfaceFormatCount);

   vk_outarray_append_typed(VkSurfaceFormatKHR, &out, f) {
      *f = (VkSurfaceFormatKHR){
         .format = VK_FORMAT_B8G8R8A8_UNORM,
         .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
      };
   }
   vk_outarray_append_typed(VkSurfaceFormatKHR, &out, f) {
      *f = (VkSurfaceFormatKHR){
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
      };
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice physicalDevice,
                                             VkSurfaceKHR surface,
                                             uint32_t *pPresentModeCount,
                                             VkPresentModeKHR *pPresentModes)
{
   (void)physicalDevice;
   (void)surface;
   VK_OUTARRAY_MAKE_TYPED(VkPresentModeKHR, out, pPresentModes,
                          pPresentModeCount);

   vk_outarray_append_typed(VkPresentModeKHR, &out, mode) {
      *mode = VK_PRESENT_MODE_FIFO_KHR;
   }
   vk_outarray_append_typed(VkPresentModeKHR, &out, mode) {
      *mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
   }
   return vk_outarray_status(&out);
}
