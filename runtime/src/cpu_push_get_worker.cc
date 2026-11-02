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
  if (!linked.ok()) return Status::Error(linked.code(), "startup notify READ: " + linked.message());
  if (sequence != 0) return Status::InvalidState("CPU push target is not idle at startup");
  linked = transport_->Complete(0, Result::kPending);
  if (!linked.ok()) return Status::Error(linked.code(), "startup completion WRITE: " + linked.message());
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
  const auto report = [&](const char* phase, const Status& status) {
    std::fprintf(stderr,
        "[KVAF_CPU] CPU_PUSH_WORKER_FAILED session=%llu phase=%s last_sequence=%llu reason=%s\n",
        static_cast<unsigned long long>(transport_->session()), phase,
        static_cast<unsigned long long>(last), status.message().c_str());
  };
  while (!stop_.load(std::memory_order_acquire)) {
    std::uint64_t sequence = 0;
    auto status = transport_->ReadRequest(&sequence);
    if (!status.ok()) { report("read_notify", status); return; }
    if (!sequence || sequence == last) continue;
    if (sequence <= last || sequence > UINT32_MAX) {
      report("validate_sequence", Status::InvalidState("invalid notify sequence=" +
                                                       std::to_string(sequence)));
      const auto completion = transport_->Complete(sequence, Result::kSequenceError);
      if (!completion.ok()) report("write_error_completion", completion);
      return;
    }
    last = sequence;
    status = transport_->ReadDescriptor(sequence);
    if (!status.ok()) {
      // Header errors occur before any data WRITE. A transport error may
      // leave the queue poisoned; Complete then fails without reusing its slots.
      report("read_descriptor", status);
      const auto completion = transport_->Complete(sequence, Result::kInvalidRequest);
      if (!completion.ok()) report("write_error_completion", completion);
      return;
    }
    status = transport_->WriteBatch();
    if (!status.ok()) { report("write_payload", status); return; }
    status = transport_->Complete(sequence, Result::kOk);
    if (!status.ok()) { report("write_completion", status); return; }
    transport_->PrintProfile();
  }
}
Status Worker::BenchmarkWrite(std::uint8_t* memory, std::uint64_t base, std::uint64_t bytes,
                             const WriteBenchmarkOptions& options, WriteBenchmarkResult* result) {
  stop_.store(true, std::memory_order_release);
  if (thread_.joinable()) thread_.join();
  return transport_->BenchmarkWrite(memory, base, bytes, options, result);
}
}  // namespace kvaf::cpu_push
