#pragma once
#include <cstddef>
#include "batch_get_abi.h"

namespace kvaf::cpu_push {
struct alignas(64) AivMailboxContext {
  KvafBatchGetDirectUrmaEntity entity;
  KvafBatchGetQueueState queue;
  std::uint32_t max_write_bytes{}, reserved[15]{};
  std::uint32_t write_templates[16][16]{};
};
static_assert(offsetof(AivMailboxContext, write_templates) == 320);
static_assert(sizeof(AivMailboxContext) == 1344);
}  // namespace kvaf::cpu_push
