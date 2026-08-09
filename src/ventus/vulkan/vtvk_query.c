#include "vtvk_private.h"

#include <string.h>

#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vtvk_entrypoints.h"

static bool
vtvk_as_query_type_supported(VkQueryType type)
{
   return type ==
          VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_CreateQueryPool(VkDevice _device,
                     const VkQueryPoolCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);

   if (!vtvk_as_query_type_supported(pCreateInfo->queryType))
      return vk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   if (pCreateInfo->queryCount >
       (SIZE_MAX - sizeof(struct vtvk_query_pool)) /
          sizeof(struct vtvk_query))
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   const size_t size =
      sizeof(struct vtvk_query_pool) +
      (size_t)pCreateInfo->queryCount * sizeof(struct vtvk_query);
   struct vtvk_query_pool *pool =
      vk_query_pool_create(&device->vk, pCreateInfo, pAllocator, size);
   if (!pool)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pQueryPool = vtvk_query_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_DestroyQueryPool(VkDevice _device, VkQueryPool queryPool,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(vtvk_device, device, _device);
   VK_FROM_HANDLE(vtvk_query_pool, pool, queryPool);

   if (pool)
      vk_query_pool_destroy(&device->vk, pAllocator, &pool->vk);
}

static bool
vtvk_query_range_valid(const struct vtvk_query_pool *pool,
                       uint32_t first_query, uint32_t query_count)
{
   return pool && first_query <= pool->vk.query_count &&
          query_count <= pool->vk.query_count - first_query;
}

static void
vtvk_reset_queries(struct vtvk_query_pool *pool, uint32_t first_query,
                   uint32_t query_count)
{
   if (!vtvk_query_range_valid(pool, first_query, query_count))
      return;

   memset(pool->queries + first_query, 0,
          (size_t)query_count * sizeof(pool->queries[0]));
}

VKAPI_ATTR void VKAPI_CALL
vtvk_ResetQueryPool(VkDevice device, VkQueryPool queryPool,
                    uint32_t firstQuery, uint32_t queryCount)
{
   (void)device;
   VK_FROM_HANDLE(vtvk_query_pool, pool, queryPool);
   vtvk_reset_queries(pool, firstQuery, queryCount);
}

struct vtvk_recorded_query_reset_command {
   struct vtvk_recorded_command base;
   VkQueryPool query_pool;
   uint32_t first_query;
   uint32_t query_count;
};

static VkResult
vtvk_execute_recorded_query_reset_command(
   struct vtvk_cmd_buffer *cmd,
   const struct vtvk_recorded_command *command)
{
   (void)cmd;
   const struct vtvk_recorded_query_reset_command *reset =
      (const struct vtvk_recorded_query_reset_command *)command;
   VK_FROM_HANDLE(vtvk_query_pool, pool, reset->query_pool);

   if (!vtvk_query_range_valid(pool, reset->first_query, reset->query_count))
      return VK_ERROR_VALIDATION_FAILED_EXT;
   vtvk_reset_queries(pool, reset->first_query, reset->query_count);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool queryPool,
                       uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct vtvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct vtvk_cmd_buffer, vk);
   struct vtvk_recorded_query_reset_command *reset =
      vtvk_cmd_buffer_alloc_command(
         cmd, sizeof(*reset), vtvk_execute_recorded_query_reset_command,
         NULL);
   if (!reset)
      return;
   reset->query_pool = queryPool;
   reset->first_query = firstQuery;
   reset->query_count = queryCount;
}

static size_t
vtvk_query_value_size(VkQueryResultFlags flags)
{
   return flags & VK_QUERY_RESULT_64_BIT ? sizeof(uint64_t) :
                                           sizeof(uint32_t);
}

static size_t
vtvk_query_record_size(VkQueryResultFlags flags)
{
   return vtvk_query_value_size(flags) *
          (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT ? 2 : 1);
}

static void
vtvk_store_query_result(void *destination, const struct vtvk_query *query,
                        VkQueryResultFlags flags)
{
   const size_t value_size = vtvk_query_value_size(flags);
   if (query->available || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
      if (flags & VK_QUERY_RESULT_64_BIT) {
         memcpy(destination, &query->value, sizeof(query->value));
      } else {
         const uint32_t value =
            query->value > UINT32_MAX ? UINT32_MAX : (uint32_t)query->value;
         memcpy(destination, &value, sizeof(value));
      }
   }

   if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
      if (flags & VK_QUERY_RESULT_64_BIT) {
         const uint64_t available = query->available;
         memcpy((uint8_t *)destination + value_size, &available,
                sizeof(available));
      } else {
         const uint32_t available = query->available;
         memcpy((uint8_t *)destination + value_size, &available,
                sizeof(available));
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
vtvk_GetQueryPoolResults(VkDevice device, VkQueryPool queryPool,
                         uint32_t firstQuery, uint32_t queryCount,
                         size_t dataSize, void *pData, VkDeviceSize stride,
                         VkQueryResultFlags flags)
{
   (void)device;
   VK_FROM_HANDLE(vtvk_query_pool, pool, queryPool);

   if (!vtvk_query_range_valid(pool, firstQuery, queryCount))
      return VK_ERROR_UNKNOWN;
   if (!queryCount)
      return VK_SUCCESS;

   const size_t record_size = vtvk_query_record_size(flags);
   if (stride > SIZE_MAX ||
       (queryCount > 1 &&
        (size_t)stride > (SIZE_MAX - record_size) / (queryCount - 1)))
      return VK_ERROR_UNKNOWN;
   const size_t required_size =
      (size_t)stride * (queryCount - 1) + record_size;
   if (!pData || dataSize < required_size)
      return VK_ERROR_UNKNOWN;

   VkResult result = VK_SUCCESS;
   for (uint32_t i = 0; i < queryCount; i++) {
      const struct vtvk_query *query = &pool->queries[firstQuery + i];
      vtvk_store_query_result((uint8_t *)pData + (size_t)stride * i,
                              query, flags);
      if (!query->available && !(flags & VK_QUERY_RESULT_PARTIAL_BIT))
         result = VK_NOT_READY;
   }
   return result;
}

struct vtvk_recorded_query_copy_command {
   struct vtvk_recorded_command base;
   VkQueryPool query_pool;
   uint32_t first_query;
   uint32_t query_count;
   VkBuffer dst_buffer;
   VkDeviceSize dst_offset;
   VkDeviceSize stride;
   VkQueryResultFlags flags;
};

static VkResult
vtvk_execute_recorded_query_copy_command(
   struct vtvk_cmd_buffer *cmd,
   const struct vtvk_recorded_command *command)
{
   (void)cmd;
   const struct vtvk_recorded_query_copy_command *copy =
      (const struct vtvk_recorded_query_copy_command *)command;
   VK_FROM_HANDLE(vtvk_query_pool, pool, copy->query_pool);

   if (!vtvk_query_range_valid(pool, copy->first_query, copy->query_count))
      return VK_ERROR_VALIDATION_FAILED_EXT;
   if (!copy->query_count)
      return VK_SUCCESS;

   const size_t record_size = vtvk_query_record_size(copy->flags);
   if (copy->stride > SIZE_MAX ||
       (copy->query_count > 1 &&
        (size_t)copy->stride >
           (SIZE_MAX - record_size) / (copy->query_count - 1)))
      return VK_ERROR_VALIDATION_FAILED_EXT;
   const size_t required_size =
      (size_t)copy->stride * (copy->query_count - 1) + record_size;
   uint8_t *destination =
      vtvk_buffer_host_ptr(copy->dst_buffer, copy->dst_offset, required_size);
   if (!destination)
      return VK_ERROR_VALIDATION_FAILED_EXT;

   for (uint32_t i = 0; i < copy->query_count; i++) {
      vtvk_store_query_result(destination + (size_t)copy->stride * i,
                              &pool->queries[copy->first_query + i],
                              copy->flags);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
vtvk_CmdCopyQueryPoolResults(VkCommandBuffer commandBuffer,
                             VkQueryPool queryPool, uint32_t firstQuery,
                             uint32_t queryCount, VkBuffer dstBuffer,
                             VkDeviceSize dstOffset, VkDeviceSize stride,
                             VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(vk_command_buffer, vk_cmd, commandBuffer);
   struct vtvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct vtvk_cmd_buffer, vk);
   struct vtvk_recorded_query_copy_command *copy =
      vtvk_cmd_buffer_alloc_command(
         cmd, sizeof(*copy), vtvk_execute_recorded_query_copy_command,
         NULL);
   if (!copy)
      return;
   copy->query_pool = queryPool;
   copy->first_query = firstQuery;
   copy->query_count = queryCount;
   copy->dst_buffer = dstBuffer;
   copy->dst_offset = dstOffset;
   copy->stride = stride;
   copy->flags = flags;
}
