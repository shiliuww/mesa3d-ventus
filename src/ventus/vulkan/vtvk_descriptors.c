#include "vtvk_private.h"
#include "vtvk_memory.h"

#include <string.h>

#include "vk_util.h"
#include "vtvk_entrypoints.h"
#include "ventus/compiler/vt_rt_descriptor_abi.h"

struct vtvk_descriptor_binding_layout {
   VkDescriptorType type;
   uint32_t count;
   VkShaderStageFlags stages;
};

struct vtvk_descriptor_set_layout {
   struct vtvk_descriptor_binding_layout bindings[VTVK_MAX_DESCRIPTOR_BINDINGS];
};

struct vtvk_pipeline_layout {
   uint32_t set_layout_count;
};

struct vtvk_descriptor_element {
   VkAccelerationStructureKHR acceleration_structure;
   VkDescriptorImageInfo image;
   VkDescriptorBufferInfo buffer;
};

struct vtvk_descriptor_set;
struct vtvk_descriptor_pool {
   uint32_t max_sets, set_count;
   VkAllocationCallbacks alloc;
   struct vtvk_descriptor_set *sets;
};

struct vtvk_descriptor_set {
   struct vtvk_descriptor_pool *pool;
   struct vtvk_descriptor_set *next;
   struct {
      struct vtvk_descriptor_binding_layout layout;
      struct vtvk_descriptor_element *elements;
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
uint32_t
vtvk_descriptor_set_binding_count(VkDescriptorSet descriptorSet, uint32_t binding)
{
   struct vtvk_descriptor_set *set = vtvk_handle_to_ptr((uint64_t)descriptorSet);
   return set && binding < VTVK_MAX_DESCRIPTOR_BINDINGS ?
      set->bindings[binding].layout.count : 0;
}

bool
vtvk_descriptor_set_get_binding(VkDescriptorSet descriptorSet,
                                uint32_t binding, uint32_t element,
                                struct vtvk_descriptor_binding_info *out)
{
   struct vtvk_descriptor_set *set =
      vtvk_handle_to_ptr((uint64_t)descriptorSet);
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));
   if (!set || binding >= VTVK_MAX_DESCRIPTOR_BINDINGS ||
       element >= set->bindings[binding].layout.count)
      return false;

   out->type = set->bindings[binding].layout.type;
   out->array_element = element;
   out->descriptor_count = set->bindings[binding].layout.count;
   const struct vtvk_descriptor_element *src =
      &set->bindings[binding].elements[element];
   out->acceleration_structure = src->acceleration_structure;
   out->image_info = src->image;
   out->buffer = src->buffer;

   if (out->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
       out->type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
       out->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      struct vtvk_image_view *view =
         vtvk_handle_to_ptr((uint64_t)out->image_info.imageView);
      out->image = view ? view->image : VK_NULL_HANDLE;
   }

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
   case VK_DESCRIPTOR_TYPE_SAMPLER:
      out->valid = out->image_info.sampler != VK_NULL_HANDLE;
      break;
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      out->valid = out->image != VK_NULL_HANDLE;
      break;
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      out->valid = out->image != VK_NULL_HANDLE &&
                   out->image_info.sampler != VK_NULL_HANDLE;
      break;
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
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

   for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *b = &pCreateInfo->pBindings[i];
      if (b->binding >= VTVK_MAX_DESCRIPTOR_BINDINGS ||
          b->descriptorCount > (UINT32_MAX >> VT_RT_DESCRIPTOR_INDEX_SHIFT) ||
          ((b->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
            b->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) &&
           b->pImmutableSamplers)) {
         vk_free(alloc, layout);
         return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      }
      layout->bindings[b->binding] = (struct vtvk_descriptor_binding_layout) {
         b->descriptorType, b->descriptorCount, b->stageFlags
      };
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
   pool->alloc = *alloc;
   *pDescriptorPool = (VkDescriptorPool)vtvk_ptr_to_handle(pool);
   return VK_SUCCESS;
}

static void
vtvk_descriptor_set_destroy(struct vtvk_descriptor_pool *pool,
                             struct vtvk_descriptor_set *set)
{
   for (uint32_t b = 0; b < VTVK_MAX_DESCRIPTOR_BINDINGS; b++)
      vk_free(&pool->alloc, set->bindings[b].elements);
   vk_free(&pool->alloc, set);
}

static void
vtvk_descriptor_pool_clear(struct vtvk_descriptor_pool *pool)
{
   while (pool->sets) {
      struct vtvk_descriptor_set *set = pool->sets;
      pool->sets = set->next;
      vtvk_descriptor_set_destroy(pool, set);
   }
   pool->set_count = 0;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyDescriptorPool(VkDevice _device, VkDescriptorPool descriptorPool,
                           const VkAllocationCallbacks *pAllocator)
{
   (void)_device;
   (void)pAllocator;
   struct vtvk_descriptor_pool *pool =
      vtvk_handle_to_ptr((uint64_t)descriptorPool);
   if (!pool)
      return;
   vtvk_descriptor_pool_clear(pool);
   vk_free(&pool->alloc, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_ResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                         VkDescriptorPoolResetFlags flags)
{
   (void)device;
   (void)flags;
   vtvk_descriptor_pool_clear(vtvk_handle_to_ptr((uint64_t)descriptorPool));
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_AllocateDescriptorSets(VkDevice _device,
                            const VkDescriptorSetAllocateInfo *pAllocateInfo,
                            VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   struct vtvk_descriptor_pool *pool =
      vtvk_handle_to_ptr((uint64_t)pAllocateInfo->descriptorPool);
   for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; i++)
      pDescriptorSets[i] = VK_NULL_HANDLE;
   if (!pool || pAllocateInfo->descriptorSetCount > pool->max_sets - pool->set_count)
      return vk_error(device, VK_ERROR_OUT_OF_POOL_MEMORY);
   uint32_t allocated = 0;
   for (; allocated < pAllocateInfo->descriptorSetCount; allocated++) {
      struct vtvk_descriptor_set_layout *layout = vtvk_handle_to_ptr(
         (uint64_t)pAllocateInfo->pSetLayouts[allocated]);
      struct vtvk_descriptor_set *set = vk_zalloc(&pool->alloc, sizeof(*set), 8,
                                               VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!set)
         goto fail;
      set->pool = pool;
      for (uint32_t b = 0; b < VTVK_MAX_DESCRIPTOR_BINDINGS; b++) {
         set->bindings[b].layout = layout->bindings[b];
         if (!layout->bindings[b].count)
            continue;
         size_t size;
         if (__builtin_mul_overflow((size_t)layout->bindings[b].count,
                                    sizeof(struct vtvk_descriptor_element), &size)) {
            vtvk_descriptor_set_destroy(pool, set);
            goto fail;
         }
         set->bindings[b].elements = vk_zalloc(&pool->alloc, size, 8,
                                             VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
         if (!set->bindings[b].elements) {
            vtvk_descriptor_set_destroy(pool, set);
            goto fail;
         }
      }
      set->next = pool->sets;
      pool->sets = set;
      pool->set_count++;
      pDescriptorSets[allocated] = (VkDescriptorSet)vtvk_ptr_to_handle(set);
   }
   return VK_SUCCESS;
fail:
   while (allocated) {
      struct vtvk_descriptor_set *set = pool->sets;
      pool->sets = set->next;
      pool->set_count--;
      vtvk_descriptor_set_destroy(pool, set);
      pDescriptorSets[--allocated] = VK_NULL_HANDLE;
   }
   for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; i++)
      pDescriptorSets[i] = VK_NULL_HANDLE;
   return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_FreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool,
                        uint32_t descriptorSetCount,
                        const VkDescriptorSet *pDescriptorSets)
{
   (void)device;
   struct vtvk_descriptor_pool *pool = vtvk_handle_to_ptr((uint64_t)descriptorPool);
   for (uint32_t i = 0; i < descriptorSetCount; i++) {
      struct vtvk_descriptor_set *set = vtvk_handle_to_ptr((uint64_t)pDescriptorSets[i]);
      if (!set)
         continue;
      struct vtvk_descriptor_set **link = &pool->sets;
      while (*link && *link != set)
         link = &(*link)->next;
      if (*link) {
         *link = set->next;
         pool->set_count--;
         vtvk_descriptor_set_destroy(pool, set);
      }
   }
   return VK_SUCCESS;
}

/* Walk consecutive compatible bindings for legal multi-binding writes/copies.
 * Immutable samplers are rejected by layout creation in this profile. */
static struct vtvk_descriptor_element *
vtvk_descriptor_element_at(struct vtvk_descriptor_set *set, uint32_t *binding,
                           uint32_t *element,
                           const struct vtvk_descriptor_binding_layout *first)
{
   while (*binding < VTVK_MAX_DESCRIPTOR_BINDINGS) {
      const struct vtvk_descriptor_binding_layout *b = &set->bindings[*binding].layout;
      if (!b->count) {
         (*binding)++;
         continue;
      }
      if (b->type != first->type || b->stages != first->stages)
         return NULL;
      if (*element < b->count)
         return &set->bindings[*binding].elements[(*element)++];
      *element -= b->count;
      (*binding)++;
   }
   return NULL;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_UpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                          const VkWriteDescriptorSet *pDescriptorWrites,
                          uint32_t descriptorCopyCount,
                          const VkCopyDescriptorSet *pDescriptorCopies)
{
   (void)device;
   for (uint32_t i = 0; i < descriptorWriteCount; i++) {
      const VkWriteDescriptorSet *write = &pDescriptorWrites[i];
      struct vtvk_descriptor_set *set = vtvk_handle_to_ptr((uint64_t)write->dstSet);
      uint32_t binding = write->dstBinding, element = write->dstArrayElement;
      if (!set || binding >= VTVK_MAX_DESCRIPTOR_BINDINGS)
         continue;
      const struct vtvk_descriptor_binding_layout first = set->bindings[binding].layout;
      const VkWriteDescriptorSetAccelerationStructureKHR *as_info =
         vk_find_struct_const(write->pNext, WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR);
      if (first.type != write->descriptorType)
         continue;
      for (uint32_t j = 0; j < write->descriptorCount; j++) {
         struct vtvk_descriptor_element *dst =
            vtvk_descriptor_element_at(set, &binding, &element, &first);
         if (!dst)
            break;
         switch (write->descriptorType) {
         case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            if (as_info && j < as_info->accelerationStructureCount)
               dst->acceleration_structure = as_info->pAccelerationStructures[j];
            break;
         case VK_DESCRIPTOR_TYPE_SAMPLER:
         case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
         case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            if (write->pImageInfo)
               dst->image = write->pImageInfo[j];
            break;
         case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
         case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            if (write->pBufferInfo)
               dst->buffer = write->pBufferInfo[j];
            break;
         default:
            break;
         }
      }
   }
   for (uint32_t i = 0; i < descriptorCopyCount; i++) {
      const VkCopyDescriptorSet *copy = &pDescriptorCopies[i];
      struct vtvk_descriptor_set *src = vtvk_handle_to_ptr((uint64_t)copy->srcSet);
      struct vtvk_descriptor_set *dst = vtvk_handle_to_ptr((uint64_t)copy->dstSet);
      uint32_t sb = copy->srcBinding, se = copy->srcArrayElement;
      uint32_t db = copy->dstBinding, de = copy->dstArrayElement;
      if (!src || !dst || sb >= VTVK_MAX_DESCRIPTOR_BINDINGS ||
          db >= VTVK_MAX_DESCRIPTOR_BINDINGS)
         continue;
      const struct vtvk_descriptor_binding_layout sl = src->bindings[sb].layout;
      const struct vtvk_descriptor_binding_layout dl = dst->bindings[db].layout;
      if (sl.type != dl.type)
         continue;
      for (uint32_t j = 0; j < copy->descriptorCount; j++) {
         struct vtvk_descriptor_element *from = vtvk_descriptor_element_at(src, &sb, &se, &sl);
         struct vtvk_descriptor_element *to = vtvk_descriptor_element_at(dst, &db, &de, &dl);
         if (!from || !to)
            break;
         *to = *from;
      }
   }
}
