#pragma once
#include <vector>

#include "gather_output.h"
#include "gather_read_profile.h"
#include "gather_span.h"
#include "types.h"

namespace kvaf {
// Caller supplies valid, matching lists and reserved nonoverlapping DDR/HBM
// workspace. GatherRead rebuilds all active metadata from this request.
struct CpuGatherReadRequest {
  std::vector<peer::GatherSpan> spans;
  std::uint64_t staging_address{0U};
  std::uint64_t source_descriptor_address{0U};
  MutableBufferView output;
  std::vector<GatherOutputSpan> output_spans;
  std::uint32_t sgl_size{kGatherMaxSglSize};
  void* stream{nullptr};
  GatherReadTiming* timing{nullptr};
  bool profile_detail{false};
  std::uint32_t sq_batch_wqes{32U};
  std::uint32_t pipeline_chunk_bytes{131072U};  // 0 waits for the complete CPU ACK.
};
}  // namespace kvaf
