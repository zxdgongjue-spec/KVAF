#pragma once
#include "cpu_sgl_push_abi.h"
#ifndef KVAF_SGL_LAYOUT_INLINE
#define KVAF_SGL_LAYOUT_INLINE inline
#endif
namespace kvaf::cpu_push {
// Mode 5 and its registered layout stay compatible. Mode 6 carries Head rows
// in bits 8..31. Mode 7 additionally selects cooperative builders; older CPU
// services reject these modes before any device launch.
constexpr std::uint32_t kPushHeadOutput = 6;
constexpr std::uint32_t kPushCooperativeOutput = 7;
constexpr std::uint64_t kPushHeadLaunch = std::uint64_t{1} << 60;
constexpr std::uint64_t kPushCooperativeLaunch = std::uint64_t{1} << 59;
KVAF_SGL_LAYOUT_INLINE bool ValidPushHeadRows(std::uint32_t head) { return head && head <= kChunkRows; }
KVAF_SGL_LAYOUT_INLINE bool ValidCooperativeHeadRows(std::uint32_t head) {
  return ValidPushHeadRows(head) && head % 8 == 0;
}
KVAF_SGL_LAYOUT_INLINE std::uint32_t PushOutputMode(std::uint32_t head, bool cooperative = false) {
  if (cooperative) return kPushCooperativeOutput | (head << 8);
  return head == kChunkRows ? kPushChunkedOutput : kPushHeadOutput | (head << 8);
}
KVAF_SGL_LAYOUT_INLINE std::uint32_t PushSetupHeadRows(std::uint32_t mode) {
  const auto head = mode >> 8;
  return head && (((mode & 255U) == kPushHeadOutput && head < kChunkRows) ||
      ((mode & 255U) == kPushCooperativeOutput && ValidCooperativeHeadRows(head))) ? head : 0;
}
KVAF_SGL_LAYOUT_INLINE std::uint64_t PushLaunch(std::uint32_t head, bool cooperative = false) {
  return kChunkPipelineLaunch | (head == kChunkRows ? 0 : kPushHeadLaunch | head) |
      (cooperative ? kPushCooperativeLaunch : 0);
}
KVAF_SGL_LAYOUT_INLINE bool PushLaunchCooperative(std::uint64_t hint) {
  return (hint & kPushCooperativeLaunch) != 0;
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
// begin must be 8-row aligned (cooperative Heads require multiples of 8).
// Each 64-byte descriptor cache line has exactly one builder, even when a
// short Head has fewer groups than workers. Empty builders still publish done.
KVAF_SGL_LAYOUT_INLINE std::uint32_t PushBuildBoundary(std::uint32_t begin, std::uint32_t end,
    std::uint32_t worker, std::uint32_t workers) {
  const auto groups = (end - begin + 7) / 8;
  const auto boundary = begin + groups * worker / workers * 8;
  return boundary < end ? boundary : end;
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
