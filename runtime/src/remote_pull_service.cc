#include "remote_pull_service.h"
#include "offload_client.h"
#include <cstdio>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace kvaf::remote_pull {
namespace {
Result Convert(const Status& s) { return s.ok() ? Result::Ok : s.code() == StatusCode::kInDoubt ? Result::InDoubt : Result::Failed; }
struct Session {
  RegionInfo region{};
  Endpoint source{}, local{};
  std::unique_ptr<offload::Backend> backend;
  std::uint64_t sequence{};
  std::vector<Span> submitted;
  State state{State::Connecting}, terminal{State::Connecting};
  std::mutex mutex;
  std::thread worker;
  ~Session() { if (worker.joinable()) worker.join(); }
};
bool Pending(State s) { return s == State::Ready || s == State::Running || s == State::InDoubt; }
}
struct Service::Impl {
  Describe describe;
  Attach attach;
  // Closed records are retained for idempotent Detach/Complete. Registration
  // is long-lived; bound the number rather than evicting unacknowledged IDs.
  std::unordered_map<std::uint64_t, std::unique_ptr<Session>> sources;
};
Service::Service(Describe describe, Attach attach) : impl_(new Impl{std::move(describe), std::move(attach), {}}) {}
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
  for (const auto& item : impl_->sources) {
    std::lock_guard<std::mutex> lock(item.second->mutex);
    if (item.second->state != State::Closed)
      return Status::InDoubt("detach registered HBM sources before CPU shutdown");
  }
  return Status::OK();
}
Response Service::Dispatch(const Request& r, const std::vector<Span>& spans) {
  Response out{}; out.header = {kMagic, kVersion, std::uint16_t(r.header.command | 0x8000U), sizeof(out)};
  out.region = r.region; out.epoch = r.epoch; out.source_id = r.source_id; out.task = r.task;
  if (r.header.magic != kMagic || r.header.version != kVersion || !r.region ||
      r.count != spans.size() || spans.size() > kMaxSpans ||
      r.header.bytes != sizeof(r) + spans.size()*sizeof(Span)) return out;
  const auto cmd = static_cast<Command>(r.header.command);
  if (cmd != Command::Ready && cmd != Command::Submit && cmd != Command::Probe && !spans.empty()) return out;
  RegionInfo region{};
  auto status = impl_->describe(r.region, &region);
  if (!status.ok()) { out.result = Result::NotFound; return out; }
  out.epoch = region.epoch; out.pool = region;
  if (!r.token || r.token != region.token || (cmd != Command::Describe && r.epoch != region.epoch)) return out;
  if (cmd == Command::Describe) { out.reserved = RegionBusy(r.region) ? 1 : 0; out.result = Result::Ok; return out; }
  if (!r.source_id) return out;
  auto found = impl_->sources.find(r.source_id);
  if (cmd == Command::Attach && found == impl_->sources.end()) {
    if (region.closing || r.task || !r.source.bytes || r.source.token != region.token) return out;
    if (impl_->sources.size() >= 4096) { out.result = Result::Busy; return out; }
    auto s = std::make_unique<Session>(); s->region = region; s->source = r.source;
    s->backend = impl_->attach(region);
    if (!s->backend) { out.result = Result::Failed; return out; }
    s->local.base = region.base; s->local.bytes = region.bytes; s->local.token = region.token;
    std::memcpy(s->local.eid, region.eid, 16);
    try { status = s->backend->Open(r.source, &s->local); }
    catch (...) { status = Status::InDoubt("exception attaching HBM source"); }
    s->state = status.ok() ? State::Connecting : status.code() == StatusCode::kInDoubt ? State::InDoubt : State::Failed;
    out.result = Convert(status); out.state = s->state; out.endpoint = s->local;
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
  out.state = s.state; out.endpoint = s.local;
  if (s.region.id != r.region || s.region.epoch != r.epoch) return out;
  if (cmd == Command::Attach) {
    if (!r.task && !std::memcmp(&s.source, &r.source, sizeof(Endpoint)) && s.state != State::Closed)
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
  if (cmd == Command::Bind && !r.task && (s.state == State::Connecting || s.state == State::Idle)) {
    s.state = out.state = State::Idle; out.result = Result::Ok; return out;
  }
  if (cmd == Command::Probe && !r.task && !s.sequence && s.state == State::Idle) {
    if (spans.size() != 1 || spans[0].destination || spans[0].bytes > 64 ||
        !offload::Contains(s.source.bytes, spans[0].source, spans[0].bytes)) return out;
    try { status = s.backend->ProbeToScratch(spans[0].source, spans[0].bytes, &out.checksum); }
    catch (...) { status = Status::InDoubt("exception probing HBM source"); }
    if (!status.ok()) s.state = status.code() == StatusCode::kInDoubt ? State::InDoubt : State::Failed;
    out.state = s.state; out.result = Convert(status); return out;
  }
  if (cmd == Command::InspectSource && !r.task && s.state != State::Closed) {
    out.endpoint = s.source; out.result = Result::Ok; return out;
  }
  if (!r.task) return out;
  if (cmd == Command::Ready) {
    if (r.flags || !spans.empty()) return out;
    if (r.task == s.sequence && s.state != State::Closed) { out.result = Result::Ok; return out; }
    if (r.task != s.sequence + 1 || (s.state != State::Idle && s.state != State::Committed && s.state != State::Cancelled)) return out;
    if (s.worker.joinable()) s.worker.join();
    s.sequence = r.task; s.submitted.clear();
    s.state = out.state = State::Ready; out.result = Result::Ok; return out;
  }
  // A Ready packet may never have reached the service. Cancel also records
  // the next unused sequence, preventing a delayed Ready from reviving it.
  if (cmd == Command::Cancel && r.task == s.sequence + 1 &&
      (s.state == State::Idle || s.state == State::Committed || s.state == State::Cancelled)) {
    s.sequence = r.task; s.submitted.clear();
    s.state = s.terminal = out.state = State::Cancelled; out.result = Result::Ok; return out;
  }
  if (r.task != s.sequence) return out; // Old notifications cannot consume a newer generation.
  if (cmd == Command::Query) { out.endpoint = s.source; out.result = Result::Ok; return out; }
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
  if (cmd != Command::Submit) return out;
  if (!s.submitted.empty()) {
    if (SameSpans(spans, s.submitted)) out.result = Result::Ok;
    return out;
  }
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
  s.submitted = spans; s.state = State::Running;
  try {
    s.worker = std::thread([&s, id = r.source_id] {
      Status result;
      try { result = s.backend->Read(s.submitted); }
      catch (...) { result = Status::InDoubt("exception in remote READ worker"); }
      std::lock_guard<std::mutex> guard(s.mutex);
      s.state = result.ok() ? State::Done : result.code() == StatusCode::kInDoubt ? State::InDoubt : State::Failed;
      std::fprintf(stderr, "[KVAF_CPU] REMOTE_PULL_DONE region=%llu source=%llu sequence=%llu state=%u detail=%s\n",
          (unsigned long long)s.region.id, (unsigned long long)id, (unsigned long long)s.sequence,
          unsigned(s.state), result.message().c_str());
    });
  } catch (...) { s.state = State::Failed; out.state = s.state; out.result = Result::Failed; return out; }
  out.state = State::Running; out.result = Result::Ok; return out;
}
void Service::Serve(int fd, const peer::MessageHeader& header) {
  offload::SetSocketTimeout(fd);
  if (header.version != kVersion || header.bytes < sizeof(Request) || header.bytes > sizeof(Request)+kMaxSpans*sizeof(Span)) return;
  Request r{}; r.header = header;
  if (!offload::ReadBytes(fd, reinterpret_cast<char*>(&r)+sizeof(header), sizeof(r)-sizeof(header)) ||
      r.count > kMaxSpans || header.bytes != sizeof(r)+r.count*sizeof(Span)) return;
  std::vector<Span> spans(r.count);
  if (!offload::ReadBytes(fd, spans.data(), spans.size()*sizeof(Span))) return;
  const auto response = Dispatch(r, spans); offload::WriteBytes(fd, &response, sizeof(response));
}
}
