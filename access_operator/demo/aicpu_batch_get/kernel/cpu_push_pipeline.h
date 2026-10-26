#pragma once

#include "cpu_push_get_layout.h"
#include "cpu_push/a12bb16/pipeline.h"

namespace kvaf::cpu_push {
static_assert(kChunkRows == aicpu_test::kPipelineChunkRows && kJettyCount == aicpu_test::kJettyCount);
static_assert(static_cast<std::uint64_t>(Result::kPending) ==
                  static_cast<std::uint64_t>(aicpu_test::Result::kPending) &&
              static_cast<std::uint64_t>(Result::kOk) ==
                  static_cast<std::uint64_t>(aicpu_test::Result::kOk) &&
              static_cast<std::uint64_t>(Result::kTimeout) ==
                  static_cast<std::uint64_t>(aicpu_test::Result::kTimeout) &&
              static_cast<std::uint64_t>(Result::kSequenceError) ==
                  static_cast<std::uint64_t>(aicpu_test::Result::kSequenceError) &&
              static_cast<std::uint64_t>(Result::kCopyError) ==
                  static_cast<std::uint64_t>(aicpu_test::Result::kCopyError));
// Adapt only Runtime's error codes, output-index Tensor and completion status.
// The ready-prefix/chunk loop is the unmodified a12bb16 V24 implementation.
template <class IO>
Result Consume(std::uint32_t count, std::uint32_t bytes, std::uint32_t workers,
               std::uint32_t worker, std::uint64_t sequence, std::uint64_t start,
               std::uint64_t timeout, IO& io, std::uint32_t* consumed) {
  const auto config = ReferenceConfig(count, bytes, workers);
  if (worker >= aicpu_test::ActiveThreads(config)) return Result::kOk;
  const auto range = WorkerRange(count, bytes, worker, workers);
  struct Adapter {
    IO& io;
    std::uint32_t begin;
    std::uint32_t* consumed;
    Result copy_status{Result::kPending};
    aicpu_test::Result Status() {
      return static_cast<aicpu_test::Result>(copy_status != Result::kPending ? copy_status : io.Status());
    }
    std::uint64_t Now() { return io.Now(); }
    std::uint64_t ReadRangeDone(std::uint32_t w) { return io.ReadRangeDone(w); }
    void RangeReady(std::uint64_t) {}
    void Relax() { io.Relax(); }
    void CopySpan(aicpu_test::RowRange span) {
      if (!io.Copy({static_cast<std::uint32_t>(span.begin), static_cast<std::uint32_t>(span.end)}))
        copy_status = Result::kCopyError;
      else *consumed = static_cast<std::uint32_t>((span.end - begin + kChunkRows - 1) / kChunkRows);
    }
    void FinishRange() { io.FinishRange(); }
  } adapter{io, range.begin, consumed};
  return static_cast<Result>(aicpu_test::RunPipelineChunks(config, worker, sequence, start, timeout, adapter));
}

// Preserve caller request order. Each row uses V24's securec copy, including
// adjacent destinations: there is no device sort or merged payload copy.
inline bool CopyMappedRows(std::uint8_t* output, const std::uint8_t* staging,
                           const std::int32_t* output_indices, Range range,
                           std::uint32_t bytes) {
  for (auto i = range.begin; i < range.end; ++i)
    if (!aicpu_test::CopyScatterRows(output + std::uint64_t(output_indices[i]) * bytes,
        staging + std::uint64_t{i} * bytes, 1, bytes, bytes)) return false;
  return true;
}
}  // namespace kvaf::cpu_push
