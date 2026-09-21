#include "vtvk_private.h"
#include "vtvk_memory.h"

#include <string.h>
#include <stdlib.h>

#include "vk_util.h"
#include "vtvk_entrypoints.h"

struct vtvk_descriptor_set_layout {
   struct vtvk_descriptor_layout data;
};

struct vtvk_pipeline_layout {
   struct vtvk_pipeline_layout_data data;
};

struct vtvk_descriptor_pool {
   uint32_t max_sets;
};

struct vtvk_descriptor_set {
   struct vtvk_descriptor_layout layout;
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
int
vtvk_descriptor_slot(const struct vtvk_descriptor_layout *layout, uint32_t binding)
{
   for (uint32_t i = 0; i < layout->count; i++)
      if (layout->bindings[i].binding == binding)
         return layout->slots[i];
   return -1;
}

bool
vtvk_get_pipeline_layout(VkPipelineLayout handle, struct vtvk_pipeline_layout_data *out)
{
   struct vtvk_pipeline_layout *layout = vtvk_handle_to_ptr((uint64_t)handle);
   if (!layout || !out)
      return false;
   *out = layout->data;
   return true;
}

bool
vtvk_descriptor_set_layout(VkDescriptorSet handle, struct vtvk_descriptor_layout *out)
{
   struct vtvk_descriptor_set *set = vtvk_handle_to_ptr((uint64_t)handle);
   if (!set || !out)
      return false;
   *out = set->layout;
   return true;
}

VkAccelerationStructureKHR
vtvk_descriptor_set_acceleration_structure(VkDescriptorSet descriptorSet,
                                           uint32_t binding)
{
   struct vtvk_descriptor_set *set =
      vtvk_handle_to_ptr((uint64_t)descriptorSet);
   int slot = set ? vtvk_descriptor_slot(&set->layout, binding) : -1;
   if (slot < 0)
      return VK_NULL_HANDLE;

   return set->bindings[slot].acceleration_structure;
}

VkImage
vtvk_descriptor_set_image(VkDescriptorSet descriptorSet, uint32_t binding)
{
   struct vtvk_descriptor_set *set =
      vtvk_handle_to_ptr((uint64_t)descriptorSet);
   int slot = set ? vtvk_descriptor_slot(&set->layout, binding) : -1;
   if (slot < 0)
      return VK_NULL_HANDLE;

   struct vtvk_image_view *view =
      vtvk_handle_to_ptr((uint64_t)set->bindings[slot].image.imageView);
   return view ? view->image : VK_NULL_HANDLE;
}

void *
vtvk_descriptor_set_buffer_ptr(VkDescriptorSet descriptorSet, uint32_t binding,
                               VkDeviceSize *range_out)
{
   struct vtvk_descriptor_set *set =
      vtvk_handle_to_ptr((uint64_t)descriptorSet);
   int slot = set ? vtvk_descriptor_slot(&set->layout, binding) : -1;
   if (slot < 0)
      return NULL;

   VkDescriptorBufferInfo *info = &set->bindings[slot].buffer;
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

bool
vtvk_descriptor_set_get_slot(VkDescriptorSet descriptorSet,
                                uint32_t binding,
                                struct vtvk_descriptor_binding_info *out)
{
   struct vtvk_descriptor_set *set =
      vtvk_handle_to_ptr((uint64_t)descriptorSet);
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));
   if (!set || binding >= VTVK_MAX_DESCRIPTOR_BINDINGS)
      return false;

   out->type = set->bindings[binding].type;
   out->acceleration_structure = set->bindings[binding].acceleration_structure;
   out->image_info = set->bindings[binding].image;
   out->buffer = set->bindings[binding].buffer;

   struct vtvk_image_view *view =
      vtvk_handle_to_ptr((uint64_t)out->image_info.imageView);
   out->image = view ? view->image : VK_NULL_HANDLE;

   if (out->buffer.range == VK_WHOLE_SIZE) {
      struct vtvk_buffer *buffer =
         vtvk_handle_to_ptr((uint64_t)out->buffer.buffer);
      out->buffer.range =
         buffer && out->buffer.offset <= buffer->size ?
            buffer->size - out->buffer.offset : 0;
   }

   switch (out->type) {
   case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
      out->valid = out->acceleration_structure != VK_NULL_HANDLE;
      break;
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      out->valid = out->image != VK_NULL_HANDLE;
      break;
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      out->valid = out->buffer.buffer != VK_NULL_HANDLE &&
                   out->buffer.range != 0;
      break;
   default:
      out->valid = false;
      break;
   }

   return out->valid;
}

bool
vtvk_descriptor_set_get_binding(VkDescriptorSet handle, uint32_t binding,
                                struct vtvk_descriptor_binding_info *out)
{
   struct vtvk_descriptor_set *set = vtvk_handle_to_ptr((uint64_t)handle);
   int slot = set ? vtvk_descriptor_slot(&set->layout, binding) : -1;
   return slot >= 0 && vtvk_descriptor_set_get_slot(handle, slot, out);
}

static int
compare_bindings(const void *a, const void *b)
{
   uint32_t x = ((const VkDescriptorSetLayoutBinding *)a)->binding;
   uint32_t y = ((const VkDescriptorSetLayoutBinding *)b)->binding;
   return (x > y) - (x < y);
}

static bool
vtvk_descriptor_layout_supported(const VkDescriptorSetLayoutCreateInfo *info)
{
   if (info->flags || info->bindingCount > VTVK_MAX_DESCRIPTOR_BINDINGS)
      return false;
   const VkDescriptorSetLayoutBindingFlagsCreateInfo *flags =
      vk_find_struct_const(info->pNext, DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
   if (flags)
      for (uint32_t i = 0; i < flags->bindingCount; i++)
         if (flags->pBindingFlags[i])
            return false;
   uint32_t count = 0;
   for (uint32_t i = 0; i < info->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *binding = &info->pBindings[i];
      if (binding->descriptorCount > VTVK_MAX_DESCRIPTOR_BINDINGS - count ||
          binding->pImmutableSamplers)
         return false;
      if (!binding->descriptorCount)
         continue;
      switch (binding->descriptorType) {
      case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         break;
      default:
         return false;
      }
      count += binding->descriptorCount;
   }
   return true;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_GetDescriptorSetLayoutSupport(VkDevice device,
                                    const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                    VkDescriptorSetLayoutSupport *pSupport)
{
   (void)device;
   pSupport->supported = vtvk_descriptor_layout_supported(pCreateInfo);
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

   if (!vtvk_descriptor_layout_supported(pCreateInfo)) {
      vk_free(alloc, layout);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   layout->data.count = pCreateInfo->bindingCount;
   memcpy(layout->data.bindings, pCreateInfo->pBindings,
          pCreateInfo->bindingCount * sizeof(*pCreateInfo->pBindings));
   qsort(layout->data.bindings, layout->data.count,
         sizeof(layout->data.bindings[0]), compare_bindings);
   uint32_t slot = 0;
   for (uint32_t i = 0; i < layout->data.count; i++) {
      const VkDescriptorSetLayoutBinding *binding = &layout->data.bindings[i];
      layout->data.slots[i] = slot;
      slot += binding->descriptorCount;
   }
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

   if (pCreateInfo->setLayoutCount > VTVK_MAX_BOUND_DESCRIPTOR_SETS) {
      vk_free(alloc, layout);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   layout->data.count = pCreateInfo->setLayoutCount;
   for (uint32_t i = 0; i < layout->data.count; i++) {
      struct vtvk_descriptor_set_layout *set =
         vtvk_handle_to_ptr((uint64_t)pCreateInfo->pSetLayouts[i]);
      if (set)
         layout->data.sets[i] = set->data;
   }
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

      struct vtvk_descriptor_set_layout *layout =
         vtvk_handle_to_ptr((uint64_t)pAllocateInfo->pSetLayouts[i]);
      set->layout = layout->data;
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
      int base = set ? vtvk_descriptor_slot(&set->layout, write->dstBinding) : -1;
      if (base < 0 || base + write->dstArrayElement + write->descriptorCount > VTVK_MAX_DESCRIPTOR_BINDINGS)
         continue;
      for (uint32_t element = 0; element < write->descriptorCount; element++) {
         uint32_t slot = base + write->dstArrayElement + element;

         set->bindings[slot].type = write->descriptorType;
         switch (write->descriptorType) {
         case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
            const VkWriteDescriptorSetAccelerationStructureKHR *as_info =
               vk_find_struct_const(write->pNext,
                                    WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR);
            if (as_info && as_info->accelerationStructureCount)
               set->bindings[slot].acceleration_structure =
                  as_info->pAccelerationStructures[element];
            break;
         }
         case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            if (write->pImageInfo)
               set->bindings[slot].image = write->pImageInfo[element];
            break;
         case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
         case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
         case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            if (write->pBufferInfo)
               set->bindings[slot].buffer = write->pBufferInfo[element];
            break;
         default:
            break;
         }
      }
   }

   for (uint32_t i = 0; i < descriptorCopyCount; i++) {
      const VkCopyDescriptorSet *copy = &pDescriptorCopies[i];
      struct vtvk_descriptor_set *src =
         vtvk_handle_to_ptr((uint64_t)copy->srcSet);
      struct vtvk_descriptor_set *dst =
         vtvk_handle_to_ptr((uint64_t)copy->dstSet);
      int src_slot = src ? vtvk_descriptor_slot(&src->layout, copy->srcBinding) : -1;
      int dst_slot = dst ? vtvk_descriptor_slot(&dst->layout, copy->dstBinding) : -1;
      if (src_slot < 0 || dst_slot < 0 ||
          src_slot + copy->srcArrayElement + copy->descriptorCount > VTVK_MAX_DESCRIPTOR_BINDINGS ||
          dst_slot + copy->dstArrayElement + copy->descriptorCount > VTVK_MAX_DESCRIPTOR_BINDINGS)
         continue;
      memmove(&dst->bindings[dst_slot + copy->dstArrayElement],
              &src->bindings[src_slot + copy->srcArrayElement],
              copy->descriptorCount * sizeof(dst->bindings[0]));
   }
}
