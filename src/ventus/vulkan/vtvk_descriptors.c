#include "vtvk_private.h"
#include "vtvk_memory.h"

#include <string.h>

#include "vk_util.h"
#include "vtvk_entrypoints.h"

struct vtvk_descriptor_set_layout {
   uint32_t binding_count;
};

struct vtvk_pipeline_layout {
   uint32_t set_layout_count;
};

struct vtvk_descriptor_pool {
   uint32_t max_sets;
};

struct vtvk_descriptor_set {
   VkDescriptorSetLayout layout;
   struct {
      VkDescriptorType type;
      VkAccelerationStructureKHR acceleration_structure;
      VkDescriptorImageInfo image;
      VkDescriptorBufferInfo buffer;
   } bindings[VTVK_MAX_DESCRIPTOR_BINDINGS];
};
static void *
vtvk_handle_to_ptr(uint64_t handle)
{
   return (void *)(uintptr_t)handle;
}

static uint64_t
vtvk_ptr_to_handle(void *ptr)
{
   return (uint64_t)(uintptr_t)ptr;
}
VkAccelerationStructureKHR
vtvk_descriptor_set_acceleration_structure(VkDescriptorSet descriptorSet,
                                           uint32_t binding)
{
   struct vtvk_descriptor_set *set =
      vtvk_handle_to_ptr((uint64_t)descriptorSet);
   if (!set || binding >= VTVK_MAX_DESCRIPTOR_BINDINGS)
      return VK_NULL_HANDLE;

   return set->bindings[binding].acceleration_structure;
}

VkImage
vtvk_descriptor_set_image(VkDescriptorSet descriptorSet, uint32_t binding)
{
   struct vtvk_descriptor_set *set =
      vtvk_handle_to_ptr((uint64_t)descriptorSet);
   if (!set || binding >= VTVK_MAX_DESCRIPTOR_BINDINGS)
      return VK_NULL_HANDLE;

   struct vtvk_image_view *view =
      vtvk_handle_to_ptr((uint64_t)set->bindings[binding].image.imageView);
   return view ? view->image : VK_NULL_HANDLE;
}

void *
vtvk_descriptor_set_buffer_ptr(VkDescriptorSet descriptorSet, uint32_t binding,
                               VkDeviceSize *range_out)
{
   struct vtvk_descriptor_set *set =
      vtvk_handle_to_ptr((uint64_t)descriptorSet);
   if (!set || binding >= VTVK_MAX_DESCRIPTOR_BINDINGS)
      return NULL;

   VkDescriptorBufferInfo *info = &set->bindings[binding].buffer;
   VkDeviceSize range = info->range;
   if (range == VK_WHOLE_SIZE) {
      struct vtvk_buffer *buffer = vtvk_handle_to_ptr((uint64_t)info->buffer);
      range = buffer && info->offset <= buffer->size ?
                 buffer->size - info->offset : 0;
   }
   if (range_out)
      *range_out = range;
   return vtvk_buffer_host_ptr(info->buffer, info->offset, range);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateDescriptorSetLayout(VkDevice _device,
                               const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                               const VkAllocationCallbacks *pAllocator,
                               VkDescriptorSetLayout *pSetLayout)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_descriptor_set_layout *layout =
      vk_zalloc(alloc, sizeof(*layout), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!layout)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   layout->binding_count = pCreateInfo->bindingCount;
   *pSetLayout = (VkDescriptorSetLayout)vtvk_ptr_to_handle(layout);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyDescriptorSetLayout(VkDevice _device,
                                VkDescriptorSetLayout descriptorSetLayout,
                                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)descriptorSetLayout));
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreatePipelineLayout(VkDevice _device,
                          const VkPipelineLayoutCreateInfo *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkPipelineLayout *pPipelineLayout)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_pipeline_layout *layout =
      vk_zalloc(alloc, sizeof(*layout), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!layout)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   layout->set_layout_count = pCreateInfo->setLayoutCount;
   *pPipelineLayout = (VkPipelineLayout)vtvk_ptr_to_handle(layout);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyPipelineLayout(VkDevice _device,
                           VkPipelineLayout pipelineLayout,
                           const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)pipelineLayout));
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateDescriptorPool(VkDevice _device,
                          const VkDescriptorPoolCreateInfo *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator,
                          VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   struct vtvk_descriptor_pool *pool =
      vk_zalloc(alloc, sizeof(*pool), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!pool)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   pool->max_sets = pCreateInfo->maxSets;
   *pDescriptorPool = (VkDescriptorPool)vtvk_ptr_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyDescriptorPool(VkDevice _device,
                           VkDescriptorPool descriptorPool,
                           const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   const VkAllocationCallbacks *alloc = pAllocator ? pAllocator : &device->vk.alloc;
   vk_free(alloc, vtvk_handle_to_ptr((uint64_t)descriptorPool));
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_AllocateDescriptorSets(VkDevice _device,
                            const VkDescriptorSetAllocateInfo *pAllocateInfo,
                            VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      struct vtvk_descriptor_set *set =
         vk_zalloc(&device->vk.alloc, sizeof(*set), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!set)
         return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

      set->layout = pAllocateInfo->pSetLayouts[i];
      pDescriptorSets[i] = (VkDescriptorSet)vtvk_ptr_to_handle(set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_FreeDescriptorSets(VkDevice _device,
                        VkDescriptorPool descriptorPool,
                        uint32_t descriptorSetCount,
                        const VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   (void)descriptorPool;
   for (uint32_t i = 0; i < descriptorSetCount; i++)
      vk_free(&device->vk.alloc, vtvk_handle_to_ptr((uint64_t)pDescriptorSets[i]));
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_UpdateDescriptorSets(VkDevice device,
                          uint32_t descriptorWriteCount,
                          const VkWriteDescriptorSet *pDescriptorWrites,
                          uint32_t descriptorCopyCount,
                          const VkCopyDescriptorSet *pDescriptorCopies)
{
   (void)device;
   for (uint32_t i = 0; i < descriptorWriteCount; i++) {
      const VkWriteDescriptorSet *write = &pDescriptorWrites[i];
      struct vtvk_descriptor_set *set =
         vtvk_handle_to_ptr((uint64_t)write->dstSet);
      if (!set || write->dstBinding >= VTVK_MAX_DESCRIPTOR_BINDINGS ||
          write->descriptorCount == 0)
         continue;

      set->bindings[write->dstBinding].type = write->descriptorType;
      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
         const VkWriteDescriptorSetAccelerationStructureKHR *as_info =
            vk_find_struct_const(write->pNext,
                                 WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR);
         if (as_info && as_info->accelerationStructureCount)
            set->bindings[write->dstBinding].acceleration_structure =
               as_info->pAccelerationStructures[0];
         break;
      }
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         if (write->pImageInfo)
            set->bindings[write->dstBinding].image = write->pImageInfo[0];
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         if (write->pBufferInfo)
            set->bindings[write->dstBinding].buffer = write->pBufferInfo[0];
         break;
      default:
         break;
      }
   }

   for (uint32_t i = 0; i < descriptorCopyCount; i++) {
      const VkCopyDescriptorSet *copy = &pDescriptorCopies[i];
      struct vtvk_descriptor_set *src =
         vtvk_handle_to_ptr((uint64_t)copy->srcSet);
      struct vtvk_descriptor_set *dst =
         vtvk_handle_to_ptr((uint64_t)copy->dstSet);
      if (!src || !dst ||
          copy->srcBinding >= VTVK_MAX_DESCRIPTOR_BINDINGS ||
          copy->dstBinding >= VTVK_MAX_DESCRIPTOR_BINDINGS ||
          copy->descriptorCount == 0)
         continue;
      dst->bindings[copy->dstBinding] = src->bindings[copy->srcBinding];
   }
}

