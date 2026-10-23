#include "cpu_push_get_worker.h"

#include <cstdio>
#include <stdexcept>

namespace kvaf::cpu_push {
Worker::~Worker() {
  if (!Stop().ok()) {
    // Keep registered buffers and WR/SGL storage alive when queue destruction fails.
    (void)transport_.release();
  }
}
Status Worker::Start() {
  if (thread_.joinable() || stop_.load()) return Status::InvalidState("CPU push worker cannot restart");
  // Both peers have imported their Jettys before START. Check the actual
  // READ/WRITE path here so initialization can reject a disconnected EID pair.
  std::uint64_t sequence = 0;
  auto linked = transport_->ReadRequest(&sequence);
  if (!linked.ok()) return linked;
  if (sequence != 0) return Status::InvalidState("CPU push target is not idle at startup");
  linked = transport_->Complete(0, Result::kPending);
  if (!linked.ok()) return linked;
  try { thread_ = std::thread(&Worker::Run, this); }
  catch (const std::exception& e) { return Status::Internal(e.what()); }
  return Status::OK();
}
Status Worker::Stop() {
  stop_.store(true, std::memory_order_release);
  if (thread_.joinable()) thread_.join();
  return transport_ ? transport_->Shutdown() : Status::OK();
}
void Worker::Run() {
  std::uint64_t last = 0;
  while (!stop_.load(std::memory_order_acquire)) {
    std::uint64_t sequence = 0;
    auto status = transport_->ReadRequest(&sequence);
    if (!status.ok()) { std::fprintf(stderr, "CPU_PUSH: %s\n", status.message().c_str()); return; }
    if (!sequence || sequence == last) { std::this_thread::yield(); continue; }
    if (sequence <= last || sequence > UINT32_MAX) {
      (void)transport_->Complete(sequence, Result::kSequenceError);
      return;
    }
    last = sequence;
    status = transport_->ReadDescriptor(sequence);
    if (!status.ok()) {
      // Header/source errors occur before any data WRITE. A transport error may
      // leave the queue poisoned; Complete then fails without reusing its slots.
      (void)transport_->Complete(sequence, Result::kInvalidRequest);
      std::fprintf(stderr, "CPU_PUSH: %s\n", status.message().c_str());
      return;
    }
    const auto& r = transport_->request();
    std::uint32_t chunks = 0;
    for (std::uint32_t w = 0; w < r.workers; ++w)
      chunks = std::max(chunks, ChunkCount(WorkerRange(r.count, r.token_bytes, w, r.workers)));
    for (std::uint32_t chunk = 0; chunk < chunks && status.ok(); ++chunk)
      for (std::uint32_t w = 0; w < r.workers && status.ok(); ++w) {
        const auto range = WorkerRange(r.count, r.token_bytes, w, r.workers);
        if (chunk < ChunkCount(range)) status = transport_->WriteChunk(w, chunk, ChunkRange(range, chunk));
      }
    if (status.ok()) status = transport_->Complete(sequence, Result::kOk);
    if (!status.ok()) { std::fprintf(stderr, "CPU_PUSH: %s\n", status.message().c_str()); return; }
  }
}
}  // namespace kvaf::cpu_push
