#pragma once
#include <cstring>
#include "scatter_batch_get_abi.h"

namespace kvaf {
// Setup-only Host helper; never called by the AIV kernel.
inline void PrepareScatterMailboxTemplates(KvafScatterMailboxContext* context) {
  const auto& e = context->entity;
  for (auto& words : context->write_templates) {
    std::memset(words, 0, sizeof(words));
    words[0] = (1U << 16) | (1U << 18) | (1U << 20) | (1U << 28) | (1U << 29);
    words[1] = 3U << 8;
    words[2] = (e.tp_id & 0xffffffU) | (1U << 24);
    words[3] = e.remote_token_id & 0xfffffU;
    std::memcpy(words + 4, e.remote_eid, 16);
    words[8] = e.remote_token_value;
    words[13] = e.local_token_id;
  }
}
}  // namespace kvaf
