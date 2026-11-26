#pragma once
#include "cpu_sgl_push_abi.h"
#ifndef KVAF_SGL_LAYOUT_INLINE
#define KVAF_SGL_LAYOUT_INLINE inline
#endif
namespace kvaf::cpu_push {
// Mode 5 and its registered layout stay compatible. Mode 6 carries Head rows
// in bits 8..31, so older CPU services reject it before any device launch.
constexpr std::uint32_t kPushHeadOutput = 6;
constexpr std::uint64_t kPushHeadLaunch = std::uint64_t{1} << 60;
KVAF_SGL_LAYOUT_INLINE bool ValidPushHeadRows(std::uint32_t head) { return head && head <= kChunkRows; }
KVAF_SGL_LAYOUT_INLINE std::uint32_t PushOutputMode(std::uint32_t head) {
  return head == kChunkRows ? kPushChunkedOutput : kPushHeadOutput | (head << 8);
}
KVAF_SGL_LAYOUT_INLINE std::uint32_t PushSetupHeadRows(std::uint32_t mode) {
  const auto head = mode >> 8;
  return (mode & 255U) == kPushHeadOutput && head && head < kChunkRows ? head : 0;
}
KVAF_SGL_LAYOUT_INLINE std::uint64_t PushLaunch(std::uint32_t head) {
  return kChunkPipelineLaunch | (head == kChunkRows ? 0 : kPushHeadLaunch | head);
}
KVAF_SGL_LAYOUT_INLINE std::uint32_t PushLaunchHeadRows(std::uint64_t hint) {
  return hint & kPushHeadLaunch ? std::uint32_t(hint & 0xffffU) : kChunkRows;
}
KVAF_SGL_LAYOUT_INLINE std::uint32_t PushChunkCount(std::uint32_t count, std::uint32_t head) {
  return !count ? 0 : count <= head ? 1 : 1 + (count - head + kChunkRows - 1) / kChunkRows;
}
KVAF_SGL_LAYOUT_INLINE std::uint32_t PushChunkBegin(std::uint32_t chunk, std::uint32_t head) {
  return !chunk ? 0 : head + (chunk - 1) * kChunkRows;
}
KVAF_SGL_LAYOUT_INLINE std::uint32_t PushChunkEnd(std::uint32_t count, std::uint32_t chunk, std::uint32_t head) {
  const auto end = PushChunkBegin(chunk, head) + (!chunk ? head : kChunkRows);
  return end < count ? end : count;
}
KVAF_SGL_LAYOUT_INLINE std::uint32_t PushChunkIndex(std::uint32_t row, std::uint32_t head) {
  return row < head ? 0 : 1 + (row - head) / kChunkRows;
}
// A small Head can yield 65 chunks. Its final commit uses one appended Word;
// keep the existing error and publisher-state offsets unchanged.
KVAF_SGL_LAYOUT_INLINE std::uint64_t PushCommitOffset(std::uint32_t chunk) {
  return chunk < kPushChunkCount ? std::uint64_t(chunk) * sizeof(Word) : sizeof(ChunkPushState);
}
KVAF_SGL_LAYOUT_INLINE std::uint64_t PushWorkspaceBytes(std::uint32_t head) {
  return sizeof(ChunkPushState) + (head < kChunkRows ? sizeof(Word) : 0);
}
}  // namespace kvaf::cpu_push
