#pragma once
#include "cpu_sgl_push_layout.h"
#ifndef KVAF_SGL_CHUNK_INLINE
#define KVAF_SGL_CHUNK_INLINE inline
#endif
namespace kvaf::cpu_push {
// The sole publisher relays one ready chunk per owner on each pass. Builder
// progress is local; every remote commit has an immutable registered source.
template<class IO> KVAF_SGL_CHUNK_INLINE std::uint64_t RetireChunkControl(IO& io) {
  const auto result = io.Wait();
  if (!result) io.SetPending(io.Pending() - 1);
  return result;
}
template<class IO> KVAF_SGL_CHUNK_INLINE std::uint64_t FailChunkControl(IO& io, std::uint32_t owner,
                                                                     std::uint32_t result) {
  if (!io.Notified() || io.ErrorSent()) return result;
  if (io.Pending() == kPushControlWindow) {
    const auto retired = RetireChunkControl(io); if (retired) return retired;
  }
  io.SetErrorSent();
  const auto posted = io.Error(owner, result);
  if (!posted) io.SetPending(io.Pending() + 1);
  return posted ? posted : result;
}
template<class IO> KVAF_SGL_CHUNK_INLINE std::uint64_t RelayReadyChunks(IO& io, std::uint32_t count,
                                                std::uint32_t workers, std::uint64_t sequence,
                                                std::uint32_t head = kChunkRows) {
  const auto chunks = PushChunkCount(count, head);
  for (std::uint32_t owner = 0; owner < workers; ++owner) {
    const auto token = io.ProgressWord(owner), epoch = token >> 32;
    if (epoch < sequence) continue;
    if (epoch > sequence) return FailChunkControl(io, owner, std::uint32_t(Result::kSequenceError));
    const auto progress = std::uint32_t(token), sent = io.Sent(owner);
    if (progress & kChunkFailed) {
      const auto error = progress & ~kChunkFailed;
      return FailChunkControl(io, owner, error >= 2 && error <= 6 ? error : std::uint32_t(Result::kInvalidRequest));
    }
    const auto owned = chunks / workers + (owner < chunks % workers);
    if (progress > owned || progress < sent)
      return FailChunkControl(io, owner, std::uint32_t(Result::kInvalidRequest));
    if (progress == sent || (!io.Notified() && owner)) continue;
    const auto chunk = sent * workers + owner;
    if (io.Pending() == kPushControlWindow) {
      const auto retired = RetireChunkControl(io); if (retired) return retired;
    }
    const bool first = !io.Notified();
    if (first) io.SetNotified(); // A partial post may already have reached CPU.
    const auto posted = io.Post(chunk, owner, sent + 1, first);
    if (posted) return posted;
    io.SetPending(io.Pending() + 1);
    io.SetSent(owner, sent + 1);
    io.SetTotal(io.Total() + 1);
  }
  return 0;
}
template<class IO> KVAF_SGL_CHUNK_INLINE std::uint64_t DrainChunkControls(IO& io) {
  while (io.Pending()) { const auto result = RetireChunkControl(io); if (result) return result; }
  return 0;
}
// Low word: complete prefix shared by all builders. High word: first error.
// Read only builder-owned Ready lines; publisher bookkeeping may still belong
// to a previous replay while the cooperative Head is being built.
template<class IO> KVAF_SGL_CHUNK_INLINE std::uint64_t CooperativeProgress(IO& io,
    std::uint32_t chunks, std::uint32_t workers, std::uint64_t sequence) {
  auto ready = chunks;
  for (std::uint32_t w = 0; w < workers; ++w) {
    const auto token = io.ProgressWord(w), epoch = token >> 32;
    if (epoch < sequence) { ready = 0; continue; }
    if (epoch > sequence) return std::uint64_t(Result::kSequenceError) << 32;
    const auto progress = std::uint32_t(token);
    if (progress & kChunkFailed) {
      const auto error = progress & ~kChunkFailed;
      return std::uint64_t(error >= 2 && error <= 6 ? error : std::uint32_t(Result::kInvalidRequest)) << 32;
    }
    if (progress > chunks) return std::uint64_t(Result::kInvalidRequest) << 32;
    if (progress < ready) ready = progress;
  }
  return ready;
}
template<class IO> KVAF_SGL_CHUNK_INLINE std::uint64_t RelayCooperativeChunks(IO& io,
    std::uint32_t count, std::uint32_t workers, std::uint64_t sequence, std::uint32_t head) {
  const auto snapshot = CooperativeProgress(io, PushChunkCount(count, head), workers, sequence);
  if (snapshot >> 32) return FailChunkControl(io, 0, std::uint32_t(snapshot >> 32));
  const auto ready = std::uint32_t(snapshot);
  if (ready < io.Total()) return FailChunkControl(io, 0, std::uint32_t(Result::kInvalidRequest));
  while (io.Total() < ready) {
    if (io.Pending() == kPushControlWindow) {
      const auto result = RetireChunkControl(io); if (result) return result;
    }
    const auto chunk = io.Total();
    const bool first = !io.Notified();
    if (first) io.SetNotified();
    // Preserve the CPU's cyclic progress format independently of builder roles.
    const auto result = io.Post(chunk, chunk % workers, chunk / workers + 1, first);
    if (result) return result;
    io.SetPending(io.Pending() + 1);
    io.SetTotal(chunk + 1);
  }
  return 0;
}
}  // namespace kvaf::cpu_push
