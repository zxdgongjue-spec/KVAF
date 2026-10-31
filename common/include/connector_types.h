#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>
#include "status.h"

namespace kvaf {
struct CpuPullSourceDesc {
  std::uint64_t source_id{}, source_bytes{}, region_id{}, region_epoch{};
};
struct NpuPullSourceDesc {
  std::uint64_t source_id{}, source_bytes{}, source_epoch{};
};
struct ReadyToken {
  std::variant<CpuPullSourceDesc, NpuPullSourceDesc> source;
  std::uint64_t sequence{};
};
inline bool SameSource(const CpuPullSourceDesc& a, const CpuPullSourceDesc& b) {
  return a.source_id == b.source_id && a.source_bytes == b.source_bytes &&
      a.region_id == b.region_id && a.region_epoch == b.region_epoch;
}
inline bool SameSource(const NpuPullSourceDesc& a, const NpuPullSourceDesc& b) {
  return a.source_id == b.source_id && a.source_bytes == b.source_bytes && a.source_epoch == b.source_epoch;
}

// Host-only, read-only contiguous arrays. The caller keeps them stable during
// the call; asynchronous operations retain an owned snapshot before any RPC.
struct Uint64ArrayView {
  const std::uint64_t* data{};
  std::size_t size{};
  Uint64ArrayView() = default;
  Uint64ArrayView(const std::uint64_t* p, std::size_t n) : data(p), size(n) {}
  Uint64ArrayView(const std::vector<std::uint64_t>& v) : data(v.data()), size(v.size()) {}
};
// Device-only int64 metadata for NPU raw READ. Never dereferenced on the Host.
// The caller keeps storage alive and unchanged until completion/recovery.
struct NpuInt64ArrayView {
  const std::int64_t* data{};
  std::size_t size{};
};
struct CopyArrays {
  std::vector<std::uint64_t> source, destination, lengths;
};
inline Status SnapshotCopyArrays(Uint64ArrayView src, Uint64ArrayView dst,
                                 Uint64ArrayView lengths, CopyArrays* out) {
  // The control protocols share this bounded request size.
  constexpr std::size_t max_segments = 4096;
  if (!out || !src.data || !dst.data || !lengths.data || !src.size ||
      src.size != dst.size || src.size != lengths.size || src.size > max_segments ||
      reinterpret_cast<std::uintptr_t>(src.data) % alignof(std::uint64_t) ||
      reinterpret_cast<std::uintptr_t>(dst.data) % alignof(std::uint64_t) ||
      reinterpret_cast<std::uintptr_t>(lengths.data) % alignof(std::uint64_t))
    return Status::InvalidArgument("copy arrays must be aligned, nonempty and equally sized uint64 host arrays");
  for (std::size_t i = 0; i < src.size; ++i)
    if (!lengths.data[i] || lengths.data[i] > UINT64_MAX-src.data[i] ||
        lengths.data[i] > UINT64_MAX-dst.data[i])
      return Status::InvalidArgument("zero length or overflowing copy range");
  out->source.assign(src.data, src.data+src.size);
  out->destination.assign(dst.data, dst.data+dst.size);
  out->lengths.assign(lengths.data, lengths.data+lengths.size);
  return Status::OK();
}
struct CpuPullRequest {
  CpuPullSourceDesc source;
  std::uint64_t sequence{};
  CopyArrays copies;
};
enum class CpuPullState { Prepared, Submitted, Running, Done, Failed, InDoubt, Committed, Aborted };
}
