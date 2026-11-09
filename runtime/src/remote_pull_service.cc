#include "remote_pull_service.h"
#include "offload_client.h"
#include <cstdio>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <sys/socket.h>

namespace kvaf::remote_pull {
namespace {
using Clock = std::chrono::steady_clock;
std::uint64_t Nanos(Clock::duration duration) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}
struct ReadTrace { Clock::time_point launch, begin, end; };
Result Convert(const Status& s) { return s.ok() ? Result::Ok : s.code() == StatusCode::kInDoubt ? Result::InDoubt : Result::Failed; }
struct Session {
  RegionInfo region{};
  MemoryDesc source{};
  ConnectionDesc peer{}, local{};
  std::unordered_map<std::uint64_t, MemoryDesc> memories;
  std::uint32_t memory_count{1};
  std::uint64_t mr_id{1};
  std::unique_ptr<offload::Backend> backend;
  std::uint64_t sequence{};
  std::vector<Span> submitted;
  State state{State::Connecting}, terminal{State::Connecting};
  std::mutex mutex;
  std::condition_variable completed;
  std::thread worker;
  ~Session() { if (worker.joinable()) worker.join(); }
};
bool Pending(State s) { return s == State::Ready || s == State::Running || s == State::InDoubt; }
}
struct Service::Trace {
  Clock::time_point begin{Clock::now()}, locked;
  std::shared_ptr<ReadTrace> read;
  bool completed{};
};
struct Service::Impl {
  Impl(Describe d, Attach a, std::mutex* shared)
      : describe(std::move(d)), attach(std::move(a)), control_mutex(shared ? *shared : owned_mutex) {}
  Describe describe;
  Attach attach;
  std::mutex owned_mutex;
  std::mutex& control_mutex;
  // Closed records are retained for idempotent Detach/Complete. Registration
  // is long-lived; bound the number rather than evicting unacknowledged IDs.
  std::unordered_map<std::uint64_t, std::unique_ptr<Session>> sources;
};
Service::Service(Describe describe, Attach attach, std::mutex* control_mutex)
    : impl_(new Impl(std::move(describe), std::move(attach), control_mutex)) {}
Service::~Service() {
  for (auto& item : impl_->sources) {
    auto& s = *item.second;
    if (s.worker.joinable()) s.worker.join();
    if (s.state != State::Closed && (Pending(s.state) || !s.backend->Close().ok()))
      (void)item.second.release();
  }
}
bool Service::RegionBusy(std::uint64_t id) const {
  for (const auto& item : impl_->sources) {
    auto& s = *item.second; std::lock_guard<std::mutex> lock(s.mutex);
    if (s.region.id == id && s.state != State::Closed) return true;
  }
  return false;
}
Status Service::Shutdown() {
  std::lock_guard<std::mutex> control(impl_->control_mutex);
  for (const auto& item : impl_->sources) {
    std::lock_guard<std::mutex> lock(item.second->mutex);
    if (item.second->state != State::Closed)
      return Status::InDoubt("detach registered HBM sources before CPU shutdown");
  }
  return Status::OK();
}
Response Service::Dispatch(const Request& r, const std::vector<Span>& spans) {
  if (r.header.command != static_cast<std::uint16_t>(Command::SubmitAndWait) || !(r.flags & kSubmitProfile))
    return DispatchImpl(r, spans, nullptr);
  Trace trace;
  auto response = DispatchImpl(r, spans, &trace);
  const auto end = Clock::now();
  if (trace.completed) {
    CpuPullServerTiming timing{};
    timing.magic = kTimingMagic;
    timing.control_lock_ns = Nanos(trace.locked - trace.begin);
    timing.validation_ns = Nanos(trace.read->launch - trace.locked);
    timing.worker_start_ns = Nanos(trace.read->begin - trace.read->launch);
    timing.backend_read_ns = Nanos(trace.read->end - trace.read->begin);
    timing.completion_ns = Nanos(end - trace.read->end);
    timing.dispatch_ns = timing.total_ns = Nanos(end - trace.begin);
    SetServerTiming(&response, timing);
  }
  return response;
}
Response Service::DispatchImpl(const Request& r, const std::vector<Span>& spans, Trace* trace) {
  std::unique_lock<std::mutex> control(impl_->control_mutex);
  if (trace) trace->locked = Clock::now();
  Response out{}; out.header = {kMagic, kVersion, std::uint16_t(r.header.command | 0x8000U), ResponseBytes(r.header.command)};
  out.region = r.region; out.epoch = r.epoch; out.source_id = r.source_id; out.task = r.task;
  out.mr_id = r.mr_id;
  if (r.header.magic != kMagic || r.header.version != kVersion || !r.region ||
      r.count != spans.size() || spans.size() > kMaxSpans ||
      r.header.bytes != RequestBytes(r.header.command) + spans.size()*sizeof(Span)) return out;
  const auto cmd = static_cast<Command>(r.header.command);
  if (cmd != Command::Ready && cmd != Command::Submit && cmd != Command::SubmitAndWait &&
      cmd != Command::Probe && !spans.empty()) return out;
  RegionInfo region{};
  auto status = impl_->describe(r.region, &region);
  if (!status.ok()) { out.result = Result::NotFound; return out; }
  out.epoch = region.epoch; out.pool = region;
  if (!r.token || r.token != region.token || (cmd != Command::Describe && r.epoch != region.epoch)) return out;
  if (cmd == Command::Describe) { out.reserved = RegionBusy(r.region) ? 1 : 0; out.result = Result::Ok; return out; }
  if (!r.source_id) return out;
  auto found = impl_->sources.find(r.source_id);
  if (cmd == Command::Attach && found == impl_->sources.end()) {
    if (region.closing || r.task || !r.source.bytes || r.source.token != region.token || r.connection.token != region.token || r.mr_id != 1 ||
        !r.memory_count || r.memory_count > kMaxSourceMrs) return out;
    if (impl_->sources.size() >= 4096) { out.result = Result::Busy; return out; }
    auto s = std::make_unique<Session>(); s->region = region; s->source = r.source; s->peer = r.connection;
    s->memory_count = r.memory_count; s->memories.emplace(1, r.source);
    s->backend = impl_->attach(region);
    if (!s->backend) { out.result = Result::Failed; return out; }
    s->local.token = region.token;
    std::memcpy(s->local.eid, region.eid, 16);
    try { status = s->backend->Open(r.connection, r.source, &s->local); }
    catch (...) { status = Status::InDoubt("exception attaching HBM source"); }
    s->state = status.ok() ? State::Connecting : status.code() == StatusCode::kInDoubt ? State::InDoubt : State::Failed;
    out.result = Convert(status); out.state = s->state; out.connection = s->local;
    std::fprintf(stderr, "[KVAF_CPU] REMOTE_PULL_ATTACH region=%llu source=%llu state=%u detail=%s\n",
        (unsigned long long)r.region, (unsigned long long)r.source_id, unsigned(s->state), status.message().c_str());
    impl_->sources.emplace(r.source_id, std::move(s)); return out;
  }
  if (found == impl_->sources.end()) {
    // An Attach that failed before creating a source has no DMA or lease.
    if (cmd == Command::Detach) { out.result = Result::Ok; out.state = State::Closed; }
    else out.result = Result::NotFound;
    return out;
  }
  auto& s = *found->second; std::unique_lock<std::mutex> lock(s.mutex);
  out.state = s.state; out.connection = s.local;
  if (s.region.id != r.region || s.region.epoch != r.epoch) return out;
  if (cmd == Command::Attach) {
    if (!r.task && r.mr_id == 1 && r.memory_count == s.memory_count &&
        !std::memcmp(&s.memories.at(1), &r.source, sizeof(MemoryDesc)) &&
        !std::memcmp(&s.peer, &r.connection, sizeof(ConnectionDesc)) && s.state != State::Closed)
      out.result = s.state == State::InDoubt ? Result::InDoubt : s.state == State::Failed ? Result::Failed : Result::Ok;
    return out;
  }
  if (cmd == Command::Detach) {
    if (Pending(s.state)) { out.result = Result::Busy; return out; }
    if (s.state != State::Closed) {
      if (s.worker.joinable()) s.worker.join();
      status = s.backend->Close();
      if (!status.ok()) { s.state = State::InDoubt; out.state = s.state; out.result = Result::InDoubt; return out; }
      s.backend.reset(); s.state = State::Closed;
    }
    out.state = s.state; out.result = Result::Ok; return out;
  }
  if (cmd == Command::AddMemory) {
    if (r.task || !r.mr_id || r.mr_id > s.memory_count || !r.source.bytes ||
        r.source.token != region.token || s.state == State::Closed) return out;
    const auto existing = s.memories.find(r.mr_id);
    if (existing != s.memories.end()) {
      if (!std::memcmp(&existing->second, &r.source, sizeof(MemoryDesc))) out.result = Result::Ok;
      return out;
    }
    if (s.state != State::Connecting) return out;
    // The source session owns the connection; this message contains only an MR.
    try { status = s.backend->AddSource(r.mr_id, r.source); }
    catch (...) { status = Status::InDoubt("exception importing HBM MR"); }
    if (status.ok()) s.memories.emplace(r.mr_id, r.source);
    else s.state = status.code() == StatusCode::kInDoubt ? State::InDoubt : State::Failed;
    out.state = s.state; out.result = Convert(status); return out;
  }
  if (cmd == Command::Bind && !r.task && (s.state == State::Connecting || s.state == State::Idle)) {
    if (s.memories.size() != s.memory_count) return out;
    s.state = out.state = State::Idle; out.result = Result::Ok; return out;
  }
  if (cmd == Command::Probe && !r.task && !s.sequence && s.state == State::Idle) {
    const auto memory = s.memories.find(r.mr_id);
    if (memory == s.memories.end()) return out;
    if (spans.size() != 1 || spans[0].destination || spans[0].bytes > 64 ||
        !offload::Contains(memory->second.bytes, spans[0].source, spans[0].bytes)) return out;
    try {
      status = s.backend->SelectSource(r.mr_id);
      if (status.ok()) status = s.backend->ProbeToScratch(spans[0].source, spans[0].bytes, &out.checksum);
    }
    catch (...) { status = Status::InDoubt("exception probing HBM source"); }
    if (!status.ok()) s.state = status.code() == StatusCode::kInDoubt ? State::InDoubt : State::Failed;
    std::fprintf(stderr, "[KVAF_CPU] CPU_PULL_PROBE_END region=%llu source=%llu mr=%llu state=%u status=%u detail=%s\n",
        (unsigned long long)r.region, (unsigned long long)r.source_id, (unsigned long long)r.mr_id,
        unsigned(s.state), unsigned(status.code()), status.message().c_str());
    out.state = s.state; out.result = Convert(status); return out;
  }
  if (cmd == Command::InspectSource && !r.task && s.state != State::Closed) {
    const auto memory = s.memories.find(r.mr_id);
    if (memory == s.memories.end()) return out;
    out.memory = memory->second; out.reserved = s.memory_count; out.result = Result::Ok; return out;
  }
  const auto memory = s.memories.find(r.mr_id);
  if (!r.task || memory == s.memories.end()) return out;
  if (cmd == Command::Ready) {
    if (r.flags || !spans.empty()) return out;
    if (r.task == s.sequence && s.state != State::Closed) {
      if (r.mr_id == s.mr_id) out.result = Result::Ok;
      return out;
    }
    if (r.task != s.sequence + 1 || (s.state != State::Idle && s.state != State::Committed && s.state != State::Cancelled)) return out;
    if (s.worker.joinable()) s.worker.join();
    s.sequence = r.task; s.mr_id = r.mr_id; s.source = memory->second; s.submitted.clear();
    s.state = out.state = State::Ready; out.result = Result::Ok; return out;
  }
  // A Ready packet may never have reached the service. Cancel also records
  // the next unused sequence, preventing a delayed Ready from reviving it.
  if (cmd == Command::Cancel && r.task == s.sequence + 1 &&
      (s.state == State::Idle || s.state == State::Committed || s.state == State::Cancelled)) {
    s.sequence = r.task; s.mr_id = r.mr_id; s.source = memory->second; s.submitted.clear();
    s.state = s.terminal = out.state = State::Cancelled; out.result = Result::Ok; return out;
  }
  if (r.task != s.sequence || r.mr_id != s.mr_id) return out; // Bind sequence AND MR identity.
  if (cmd == Command::Query) { out.memory = s.source; out.result = Result::Ok; return out; }
  if (cmd == Command::Checksum) {
    // Diagnostic reads of CURRENT DDR are explicit and outside transfer/commit.
    // The caller must retain these slots unchanged until this RPC returns.
    if (s.state != State::Done && s.state != State::Committed) { out.result = Result::Busy; return out; }
    for (const auto& item : impl_->sources) {
      if (item.second.get() == &s || item.second->region.id != r.region) continue;
      std::lock_guard<std::mutex> guard(item.second->mutex);
      if (item.second->state == State::Running || item.second->state == State::InDoubt) {
        out.result = Result::Busy; return out;
      }
    }
    try { out.checksum = s.backend->Checksum(s.submitted); out.result = Result::Ok; }
    catch (...) { out.result = Result::Failed; }
    return out;
  }
  if (cmd == Command::Complete) {
    if (r.flags > 1) return out;
    const auto desired = r.flags ? State::Committed : State::Cancelled;
    if (s.terminal == desired && (s.state == desired || s.state == State::Closed)) { out.result = Result::Ok; return out; }
    if ((r.flags && s.state != State::Done) || (!r.flags && s.state != State::Ready && s.state != State::Failed && s.state != State::Cancelled && s.state != State::Done)) return out;
    s.state = s.terminal = out.state = desired; out.result = Result::Ok; return out;
  }
  if (cmd == Command::Cancel) {
    if (s.state == State::Ready || s.state == State::Cancelled) {
      s.state = s.terminal = out.state = State::Cancelled; out.result = Result::Ok;
    }
    return out;
  }
  if (cmd != Command::Submit && cmd != Command::SubmitAndWait) return out;
  // A new convenience call is not recovery of an already committed/cancelled
  // batch. Reject it before claiming acceptance or touching the caller's slots.
  if (cmd == Command::SubmitAndWait &&
      (s.state == State::Committed || s.state == State::Cancelled || s.state == State::Closed)) return out;
  const auto wait_for_completion = [&] {
    if (cmd == Command::SubmitAndWait) {
      // The READ worker publishes state under this same mutex. Waiting releases
      // it, so the response is driven by completion rather than 1 ms polling.
      control.unlock();
      s.completed.wait_for(lock, std::chrono::milliseconds(r.timeout_ms),
                           [&] { return s.state != State::Running; });
      // A reconnecting client may already have finished this task. Never
      // report the state of a newer batch as completion of this request.
      out.state = s.sequence == r.task && s.mr_id == r.mr_id ? s.state : State::InDoubt;
      if (trace) trace->completed = trace->read && out.state == State::Done;
    }
  };
  if (!s.submitted.empty()) {
    if (cmd == Command::SubmitAndWait) out.reserved = 1;
    if (SameSpans(spans, s.submitted) &&
        (cmd != Command::SubmitAndWait || r.source.bytes == s.source.bytes)) {
      out.result = Result::Ok;
      wait_for_completion();
    }
    return out;
  }
  // The synchronous request carries the expected selected-MR length, removing
  // the separate pre-submit Query without weakening its source validation.
  if (cmd == Command::SubmitAndWait && r.source.bytes != s.source.bytes) return out;
  if (s.state != State::Ready || !offload::ValidateSpans(spans, s.source.bytes, region.bytes)) return out;
  // One unresolved write per pool, including CQ failures with unknown DMA state.
  for (const auto& item : impl_->sources) {
    if (item.second.get() == &s || item.second->region.id != r.region) continue;
    std::lock_guard<std::mutex> guard(item.second->mutex);
    const auto state = item.second->state;
    if (state == State::Running || state == State::InDoubt || state == State::Done) {
      out.result = Result::Busy; return out;
    }
  }
  // Resolve the MR once per batch, after prior work has completed. All WQEs
  // then use its imported segment/base with the existing shared queues.
  status = s.backend->SelectSource(r.mr_id);
  if (!status.ok()) {
    if (cmd == Command::SubmitAndWait) out.reserved = 1;
    s.state = out.state = status.code() == StatusCode::kInDoubt ? State::InDoubt : State::Failed;
    out.result = Convert(status); return out;
  }
  s.submitted = spans; s.state = State::Running;
  if (cmd == Command::SubmitAndWait) out.reserved = 1;
  try {
    std::shared_ptr<ReadTrace> read_trace;
    if (trace) {
      trace->read = read_trace = std::make_shared<ReadTrace>();
      read_trace->launch = Clock::now();
    }
    s.worker = std::thread([&s, id = r.source_id, read_trace] {
      Status result;
      if (read_trace) read_trace->begin = Clock::now();
      try { result = s.backend->Read(s.submitted); }
      catch (...) { result = Status::InDoubt("exception in remote READ worker"); }
      if (read_trace) read_trace->end = Clock::now();
      std::lock_guard<std::mutex> guard(s.mutex);
      s.state = result.ok() ? State::Done : result.code() == StatusCode::kInDoubt ? State::InDoubt : State::Failed;
      s.completed.notify_all();
      std::fprintf(stderr, "[KVAF_CPU] REMOTE_PULL_DONE region=%llu source=%llu sequence=%llu mr=%llu state=%u detail=%s\n",
          (unsigned long long)s.region.id, (unsigned long long)id, (unsigned long long)s.sequence,
          (unsigned long long)s.mr_id, unsigned(s.state), result.message().c_str());
    });
  } catch (...) { s.state = State::Failed; out.state = s.state; out.result = Result::Failed; return out; }
  out.state = State::Running; out.result = Result::Ok;
  wait_for_completion();
  return out;
}
void Service::Serve(int fd, const peer::MessageHeader& header) {
  offload::SetSocketTimeout(fd);
  const auto request_bytes = RequestBytes(header.command);
  if (header.magic != kMagic || header.version != kVersion || header.bytes < request_bytes ||
      header.bytes > request_bytes+kMaxSpans*sizeof(Span)) { shutdown(fd, SHUT_RDWR); return; }
  Request r{}; r.header = header;
  if (!offload::ReadBytes(fd, reinterpret_cast<char*>(&r)+sizeof(header), request_bytes-sizeof(header)) ||
      r.count > kMaxSpans || header.bytes != request_bytes+r.count*sizeof(Span)) { shutdown(fd, SHUT_RDWR); return; }
  const bool profiling = header.command == static_cast<std::uint16_t>(Command::SubmitAndWait) && (r.flags & kSubmitProfile);
  const auto receive_begin = profiling ? Clock::now() : Clock::time_point{};
  std::vector<Span> spans(r.count);
  if (!offload::ReadBytes(fd, spans.data(), spans.size()*sizeof(Span))) { shutdown(fd, SHUT_RDWR); return; }
  const auto received = profiling ? Clock::now() : Clock::time_point{};
  auto response = Dispatch(r, spans);
  if (profiling) {
    auto timing = ServerTiming(response);
    if (timing.magic == kTimingMagic) {
      const auto reply_ready = Clock::now();
      timing.receive_ns = Nanos(received - receive_begin);
      timing.reply_prepare_ns = Nanos(reply_ready - received) - timing.dispatch_ns;
      timing.total_ns = Nanos(reply_ready - receive_begin);
      SetServerTiming(&response, timing);
    }
  }
  if (!offload::WriteBytes(fd, &response, ResponseBytes(header.command))) shutdown(fd, SHUT_RDWR);
}
}
