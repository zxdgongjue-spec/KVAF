#include "cpu_push_get_worker.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  const auto* mode = std::getenv("KVAF_CPU_PUSH_STARTUP_PROBE");
  if (!mode) mode = "basic";
  if (std::strcmp(mode, "basic") && std::strcmp(mode, "diagnostic") && std::strcmp(mode, "off"))
    return Status::InvalidArgument("KVAF_CPU_PUSH_STARTUP_PROBE must be basic, diagnostic or off");
  std::fprintf(stderr, "[KVAF_CPU] CPU_PUSH_STARTUP_PROBE session=%llu mode=%s\n",
               static_cast<unsigned long long>(transport_->session()), mode);
  // The default only checks one READ so unreachable EID pairs can fall back.
  // Diagnostic mode repeats READ/WRITE; off allows a zero-probe comparison.
  // The background worker still performs and validates every business I/O.
  if (std::strcmp(mode, "off")) {
    const auto linked = transport_->ProbeIdleLink(std::strcmp(mode, "diagnostic") == 0);
    if (!linked.ok()) return linked;
    std::fprintf(stderr, "[KVAF_CPU] CPU_PUSH_LINK_PROBE_READY session=%llu mode=%s\n",
                 static_cast<unsigned long long>(transport_->session()), mode);
  }
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
                             const WriteBenchmarkOptions& options, WriteBenchmarkResult* result,
                             const WriteBenchmarkDiagnosticOptions* diagnostics, WriteBenchmarkDiagnostics* records) {
  stop_.store(true, std::memory_order_release);
  if (thread_.joinable()) thread_.join();
  return transport_->BenchmarkWrite(memory, base, bytes, options, result, diagnostics, records);
}
}  // namespace kvaf::cpu_push
