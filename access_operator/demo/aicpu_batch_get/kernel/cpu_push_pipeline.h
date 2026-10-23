#pragma once

#include "cpu_push_get_layout.h"

namespace kvaf::cpu_push {
// A marker on one Jetty says nothing about another Jetty's data.
template <class Read>
Result ReadyPrefix(std::uint64_t sequence, std::uint32_t chunks,
                   std::uint32_t consumed, Read read, std::uint32_t* prefix) {
  auto progress = chunks;
  bool stale = false;
  for (std::uint32_t q = 0; q < kJettyCount; ++q) {
    const auto token = read(q), seq = token >> 32;
    const auto value = static_cast<std::uint32_t>(token);
    if (seq > sequence || (seq == sequence && (!value || value > chunks || value < consumed)))
      return Result::kSequenceError;
    stale |= seq < sequence;
    if (seq == sequence) progress = std::min(progress, value);
  }
  *prefix = stale ? consumed : progress;
  return Result::kPending;
}

// IO supplies the actual clock, progress visibility and scatter operation.
template <class IO>
Result Consume(Range range, std::uint64_t sequence, IO& io, std::uint32_t* consumed) {
  const auto chunks = ChunkCount(range);
  while (*consumed < chunks) {
    auto status = io.Check();
    if (status != Result::kPending) return status;
    std::uint32_t prefix = *consumed;
    status = ReadyPrefix(sequence, chunks, *consumed,
                         [&](auto q) { return io.ReadReady(q); }, &prefix);
    if (status != Result::kPending) return status;
    if (prefix > *consumed) {
      const auto first = ChunkRange(range, *consumed), last = ChunkRange(range, prefix - 1);
      if (!io.Copy({first.begin, last.end})) return Result::kCopyError;
      *consumed = prefix;
    } else io.Relax();
  }
  return Result::kOk;
}
}  // namespace kvaf::cpu_push
