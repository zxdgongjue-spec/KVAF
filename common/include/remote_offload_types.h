#pragma once
#include <cstdint>
#include <vector>
#include "types.h"

namespace kvaf {
// Opaque identities, never process-local MR/Jetty handles or dereferenceable
// pointers. The source address space belongs to the exporting NPU (card B).
struct RemoteSourceIdentity {
  std::uint64_t region_epoch{}, region_id{}, source_id{}, sequence{}, source_bytes{};
};
struct RemoteOffloadRequest {
  RemoteSourceIdentity source;
  std::vector<KeyId> keys;
  std::vector<std::uint64_t> input_offsets, lengths;
};
}
