#include "offload_service.h"
#include "offload_client.h"
#include <cstdio>
#include <exception>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>

namespace kvaf::offload {
namespace {
Result Convert(const Status& status) {
  if (status.ok()) return Result::Ok;
  if (status.code() == StatusCode::kInDoubt) return Result::InDoubt;
  return Result::Failed;
}
struct Session {
  std::unique_ptr<Backend> backend;
  std::uint64_t bytes{}, slot{}, task{};
  std::uint32_t candidate{};
  Endpoint source{}, local{};
  std::vector<Span> spans;
  mutable std::mutex mutex;
  std::thread worker;
  State state{State::Empty};
  bool released{true};
  ~Session() { if (worker.joinable()) worker.join(); }
};
}
struct Service::Impl {
  Factory factory;
  std::uint64_t epoch;
  std::unordered_map<std::uint64_t, std::unique_ptr<Session>> sessions;
};
Service::Service(Factory factory) : impl_(new Impl) {
  impl_->factory = std::move(factory);
  std::random_device random;
  impl_->epoch = (std::uint64_t(random()) << 32) ^ random();
  if (!impl_->epoch) impl_->epoch = 1;
}
Service::~Service() {
  for (auto& entry : impl_->sessions) {
    auto& s = *entry.second;
    if (s.worker.joinable()) s.worker.join();
    if (s.state == State::InDoubt || !s.backend->Close().ok())
      (void)entry.second.release(); // Quarantine DMA resources until process exit.
  }
}
bool Service::Busy() const {
  for (const auto& entry : impl_->sessions) {
    std::lock_guard<std::mutex> lock(entry.second->mutex);
    if (!entry.second->released || entry.second->state == State::InDoubt) return true;
  }
  return false;
}
Status Service::Shutdown() {
  if (Busy()) return Status::InDoubt("CPU Pull resources are still leased");
  for (auto it = impl_->sessions.begin(); it != impl_->sessions.end();) {
    auto& s = *it->second;
    if (s.worker.joinable()) s.worker.join();
    const auto status = s.backend->Close();
    if (!status.ok()) {
      s.state = State::InDoubt;
      return Status::InDoubt("CPU Pull pool teardown failed; retain its context");
    }
    it = impl_->sessions.erase(it);
  }
  return Status::OK();
}
Response Service::Dispatch(const Request& r, const std::vector<Span>& spans) {
  Response out{};
  out.header = {kMagic, kVersion, std::uint16_t(r.header.command | 0x8000U), sizeof(Response)};
  out.region = r.region; out.epoch = impl_->epoch; out.task = r.task;
  if (r.header.magic != kMagic || r.header.version != kVersion || !r.region ||
      spans.size() > kMaxSpans || r.count != spans.size() ||
      r.header.bytes != sizeof(Request) + spans.size() * sizeof(Span)) return out;
  const auto command = static_cast<Command>(r.header.command);
  if (command != Command::Submit && !spans.empty()) return out;
  auto found = impl_->sessions.find(r.region);
  if (command == Command::Allocate) {
    if (r.epoch || r.task || !r.bytes || !r.slot_size || r.bytes % r.slot_size) return out;
    if (found == impl_->sessions.end()) {
      if (impl_->sessions.size() >= 64) { out.result = Result::Busy; return out; }
      auto s = std::make_unique<Session>();
      s->bytes = r.bytes; s->slot = r.slot_size; s->candidate = r.candidate;
      s->backend = impl_->factory(r.candidate);
      if (!s->backend) { out.result = Result::Failed; return out; }
      const auto status = s->backend->Allocate(r.bytes, &s->local);
      if (!status.ok()) {
        out.result = Convert(status);
        if (!s->backend->Close().ok()) {
          s->state = State::InDoubt;
          impl_->sessions.emplace(r.region, std::move(s));
          out.result = Result::InDoubt;
        }
        return out;
      }
      found = impl_->sessions.emplace(r.region, std::move(s)).first;
    }
    auto& s = *found->second;
    std::lock_guard<std::mutex> allocation_lock(s.mutex);
    if (s.state == State::InDoubt) { out.result = Result::InDoubt; return out; }
    if (s.bytes != r.bytes || s.slot != r.slot_size || s.candidate != r.candidate) return out;
    out.base = s.local.base; out.bytes = s.bytes; out.endpoint = s.local;
    out.result = Result::Ok;
    return out;
  }
  if (r.epoch != impl_->epoch) { out.result = Result::NotFound; return out; }
  if (found == impl_->sessions.end()) { out.result = Result::NotFound; return out; }
  auto& s = *found->second;
  std::unique_lock<std::mutex> lock(s.mutex);
  out.state = s.state; out.endpoint = s.local;
  if (command == Command::Close) {
    if (!s.released || s.state == State::InDoubt) { out.result = Result::Busy; return out; }
    if (s.worker.joinable()) s.worker.join();
    out.result = Convert(s.backend->Close());
    if (out.result == Result::Ok) { lock.unlock(); impl_->sessions.erase(found); }
    else { s.state = State::InDoubt; out.state = s.state; }
    return out;
  }
  if (!r.task) return out;
  if (command == Command::Open) {
    if (r.task == s.task) {
      if (std::memcmp(&r.source, &s.source, sizeof(Endpoint)) == 0 && !s.released) {
        out.result = s.state == State::InDoubt ? Result::InDoubt : Result::Ok;
      }
      return out;
    }
    if (r.task < s.task || !s.released || s.state == State::InDoubt) { out.result = Result::Busy; return out; }
    if (!r.source.bytes || r.source.base > UINT64_MAX - r.source.bytes) return out;
    if (s.worker.joinable()) s.worker.join();
    s.task = r.task; s.source = r.source; s.spans.clear(); s.released = false;
    const auto status = s.backend->Open(r.source, &s.local);
    s.state = status.ok() ? State::Open : (status.code() == StatusCode::kInDoubt ? State::InDoubt : State::Failed);
    out.state = s.state; out.endpoint = s.local; out.result = Convert(status);
    return out;
  }
  if (r.task != s.task) { out.result = Result::NotFound; return out; }
  if (command == Command::Query) { out.result = Result::Ok; return out; }
  if (command == Command::ReleaseSource) {
    if (s.state == State::Running || s.state == State::InDoubt) { out.result = Result::Busy; return out; }
    if (s.worker.joinable()) s.worker.join();
    const auto status = s.released ? Status::OK() : s.backend->ReleaseSource();
    if (status.ok()) s.released = true;
    else s.state = State::InDoubt;
    out.state = s.state; out.result = Convert(status); return out;
  }
  if (command == Command::Checksum) {
    if (s.state != State::Done) { out.result = Result::Busy; return out; }
    out.checksum = s.backend->Checksum(s.spans); out.result = Result::Ok; return out;
  }
  if (command != Command::Submit || s.released) return out;
  if (!s.spans.empty()) {
    if (spans.size() == s.spans.size() &&
        std::memcmp(spans.data(), s.spans.data(), spans.size()*sizeof(Span)) == 0)
      out.result = Result::Ok;
    return out;
  }
  if (s.state != State::Open || !ValidateSpans(spans, s.source.bytes, s.bytes)) return out;
  // Match the allocator's fixed slots; a task cannot spill into another slot.
  for (const auto& span : spans)
    if (span.destination % s.slot || span.bytes > s.slot) return out;
  s.spans = spans;
  s.state = State::Running;
  try {
    s.worker = std::thread([&s, region = r.region, task = r.task] {
      std::fprintf(stderr, "[KVAF_CPU] CPU_PULL_TASK_BEGIN region=%llu task=%llu spans=%zu\n",
                   static_cast<unsigned long long>(region), static_cast<unsigned long long>(task),
                   s.spans.size());
      Status status;
      try { status = s.backend->Read(s.spans); }
      catch (const std::exception& e) { status = Status::InDoubt(std::string("CPU Pull worker exception: ") + e.what()); }
      catch (...) { status = Status::InDoubt("CPU Pull worker exception"); }
      std::lock_guard<std::mutex> guard(s.mutex);
      s.state = status.ok() ? State::Done :
          (status.code() == StatusCode::kInDoubt ? State::InDoubt : State::Failed);
      std::fprintf(stderr, "[KVAF_CPU] CPU_PULL_TASK_END region=%llu task=%llu state=%u status=%u detail=%s\n",
                   static_cast<unsigned long long>(region), static_cast<unsigned long long>(task),
                   static_cast<unsigned>(s.state), static_cast<unsigned>(status.code()),
                   status.ok() ? "completed" : status.message().c_str());
    });
  } catch (...) { s.spans.clear(); s.state = State::Failed; out.result = Result::Failed; return out; }
  out.state = State::Running; out.result = Result::Ok;
  return out;
}
void Service::Serve(int fd, const peer::MessageHeader& header) {
  SetSocketTimeout(fd);
  if (header.version != kVersion || header.bytes < sizeof(Request) ||
      header.bytes > sizeof(Request) + kMaxSpans * sizeof(Span)) return;
  Request request{}; request.header = header;
  if (!ReadBytes(fd, reinterpret_cast<char*>(&request) + sizeof(header), sizeof(request)-sizeof(header)) ||
      request.count > kMaxSpans || header.bytes != sizeof(Request) + request.count*sizeof(Span)) return;
  std::vector<Span> spans(request.count);
  if (!ReadBytes(fd, spans.data(), spans.size()*sizeof(Span))) return;
  const auto response = Dispatch(request, spans);
  (void)WriteBytes(fd, &response, sizeof(response));
}
}
