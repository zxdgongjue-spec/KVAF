#include "cache_region_service.h"
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#define CHECK(c)                                        \
  do {                                                  \
    if (!(c)) {                                         \
      std::cerr << "line " << __LINE__ << ": " #c "\n"; \
      std::abort();                                     \
    }                                                   \
  } while (false)
using namespace kvaf;
using namespace kvaf::execution;
namespace {
struct Counts {
  int allocations = 0, opens = 0, calls = 0, closes = 0;
  bool fail_close = false;
  bool fail_call = false;
};
struct Pool : Allocation {
  explicit Pool(Counts& c) : c(c) { ++c.allocations; }
  Status Close() override {
    ++c.closes;
    return Status::OK();
  }
  Counts& c;
};
struct Fake : Session {
  explicit Fake(Counts& c) : c(c) {}
  Status Open(const Request&, Allocation& a, peer::PeerRegion* out) override {
    ++c.opens;
    *out = a.descriptor;
    out->base += 0x100000;
    return Status::OK();
  }
  Status Call(const std::vector<std::uint8_t>& in, std::vector<std::uint8_t>* out) override {
    ++c.calls;
    if (c.fail_call) throw std::runtime_error("control failed after side effect");
    *out = in;
    return Status::OK();
  }
  Status Close() override {
    if (c.fail_close) return Status::Busy("DMA active");
    ++c.closes;
    return Status::OK();
  }
  Counts& c;
};
CacheRegionService Service(Counts& c) {
  return CacheRegionService(
      kFabric | kUrma,
      [&](const Request& q, std::unique_ptr<Allocation>* out) {
        auto a = std::make_unique<Pool>(c);
        a->descriptor.base = 0x1000;
        a->descriptor.length = q.logical_bytes + q.auxiliary_bytes;
        if (q.mode != TransportMode::Urma) {
          a->descriptor.fabric_mapping_bytes = 2 * 1024 * 1024;
          a->descriptor.fabric_handle[0] = 19;
        }
        *out = std::move(a);
        return Status::OK();
      },
      [&](const Request&) { return std::make_unique<Fake>(c); });
}
}  // namespace
int main() {
  Counts c;
  auto service = Service(c);
  Request create;
  create.op = Op::CreateRegion;
  create.client = 17;
  create.request = 1;
  create.mode = TransportMode::UrmaAndUbmem;
  create.logical_bytes = 4096;
  create.slot_size = 64;
  create.auxiliary_bytes = 8192;
  create.epoch = service.Handle(Request{}).epoch;
  auto pool = service.Handle(create);
  CHECK(pool.status.ok());
  CHECK(c.allocations == 1);
  auto stale = create;
  stale.epoch++;
  CHECK(!service.Handle(stale).status.ok() && c.allocations == 1);
  auto duplicate = service.Handle(create);
  CHECK(duplicate.region == pool.region && c.allocations == 1);
  auto altered = create;
  altered.logical_bytes *= 2;
  CHECK(!service.Handle(altered).status.ok());
  Request query = create;
  query.op = Op::Query;
  CHECK(service.Handle(query).region == pool.region);
  Request open;
  open.op = Op::OpenSession;
  open.client = 17;
  open.request = 2;
  open.region = pool.region;
  open.epoch = pool.epoch;
  open.executor = Executor::Aicpu;
  open.algorithm = GetAlgorithm::Gather;
  auto gather = service.Handle(open);
  CHECK(gather.status.ok());
  CHECK(gather.binding.base == pool.binding.base + 0x100000);
  open.request = 3;
  open.algorithm = GetAlgorithm::Scatter;
  auto scatter = service.Handle(open);
  CHECK(scatter.status.ok());
  open.request = 4;
  open.transport = Transport::Ubmem;
  open.executor = Executor::Aiv;
  open.algorithm = GetAlgorithm::Direct;
  auto fabric = service.Handle(open);
  CHECK(fabric.status.ok());
  CHECK(c.allocations == 1 && c.opens == 3);
  Request release;
  release.op = Op::ReleaseRegion;
  release.client = 17;
  release.region = pool.region;
  release.epoch = pool.epoch;
  CHECK(service.Handle(release).status.code() == StatusCode::kBusy);
  Request call = release;
  call.op = Op::SessionCall;
  call.session = gather.session;
  call.request = 5;
  call.payload = {1, 2, 3};
  CHECK(service.Handle(call).status.ok());
  CHECK(service.Handle(call).payload == call.payload && c.calls == 1);
  call.payload = {7};
  CHECK(!service.Handle(call).status.ok());
  CHECK(c.calls == 1);
  auto failing_call = call;
  failing_call.request = 6;
  c.fail_call = true;
  CHECK(!service.Handle(failing_call).status.ok());
  c.fail_call = false;
  CHECK(!service.Handle(failing_call).status.ok() && c.calls == 2);
  call.payload = {1, 2, 3};
  call.epoch++;
  CHECK(!service.Handle(call).status.ok());
  call.epoch--;
  call.client++;
  CHECK(!service.Handle(call).status.ok());
  call.client--;
  c.fail_close = true;
  call.op = Op::CloseSession;
  CHECK(service.Handle(call).state == State::Recovering);
  CHECK(service.Handle(release).status.code() == StatusCode::kBusy);
  c.fail_close = false;
  CHECK(service.Handle(call).status.ok());
  CHECK(service.Handle(call).state == State::Closed);
  call.session = scatter.session;
  CHECK(service.Handle(call).status.ok());
  call.session = fabric.session;
  CHECK(service.Handle(call).status.ok());
  CHECK(service.Handle(release).status.ok());
  CHECK(service.Handle(create).state == State::Closed);
  CHECK(c.allocations == 1);
  // Truncation, trailing data, invalid version/enum and integer overflows never deserialize.
  auto bytes = Encode(open);
  Request decoded;
  CHECK(Decode(bytes, &decoded).ok());
  CHECK(Encode(decoded) == bytes);
  for (std::size_t n = 0; n < bytes.size(); ++n)
    CHECK(!Decode(std::vector<std::uint8_t>(bytes.begin(), bytes.begin() + n), &decoded).ok());
  bytes.push_back(0);
  CHECK(!Decode(bytes, &decoded).ok());
  bytes = Encode(open);
  bytes[0]++;
  CHECK(!Decode(bytes, &decoded).ok());
  auto replace_word = [](std::vector<std::uint8_t>& wire, std::size_t word, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) wire[word * 8 + i] = (value >> (8 * i)) & 255;
  };
  for (auto word : {1U, 7U, 8U, 9U, 10U}) {
    for (auto value : {UINT64_C(0xffffffff), UINT64_C(0x100000000), UINT64_MAX}) {
      bytes = Encode(open);
      replace_word(bytes, word, value);
      CHECK(!Decode(bytes, &decoded).ok());
    }
  }
  for (auto word : {12U, 13U, 14U, 15U, 16U, 17U}) {
    bytes = Encode(open);
    replace_word(bytes, word, UINT64_C(0x100000000));
    CHECK(!Decode(bytes, &decoded).ok());
  }
  Response decoded_response;
  bytes = Encode(gather);
  CHECK(Decode(bytes, &decoded_response).ok());
  CHECK(Encode(decoded_response) == bytes);
  for (auto word : {1U, 2U}) {
    bytes = Encode(gather);
    replace_word(bytes, word, UINT64_MAX);
    CHECK(!Decode(bytes, &decoded_response).ok());
  }
  // Slow session creation must not serialize unrelated creates or pool queries.
  Counts parallel;
  std::promise<void> entered, resume;
  auto gate = resume.get_future().share();
  CacheRegionService concurrent(
      kUrma,
      [&](const Request& q, std::unique_ptr<Allocation>* out) {
        auto a = std::make_unique<Pool>(parallel);
        a->descriptor.base = 0x1000;
        a->descriptor.length = q.logical_bytes;
        *out = std::move(a);
        return Status::OK();
      },
      [&](const Request& q) -> std::unique_ptr<Session> {
        if (q.request == 2) {
          entered.set_value();
          gate.wait();
        }
        return std::make_unique<Fake>(parallel);
      });
  create.mode = TransportMode::Urma;
  create.auxiliary_bytes = 0;
  create.epoch = concurrent.Handle(Request{}).epoch;
  pool = concurrent.Handle(create);
  CHECK(pool.status.ok());
  open.transport = Transport::Urma;
  open.executor = Executor::Aiv;
  open.request = 2;
  open.region = pool.region;
  open.epoch = pool.epoch;
  auto slow = std::async(std::launch::async, [&] { return concurrent.Handle(open); });
  entered.get_future().wait();
  auto other = open;
  other.request = 3;
  other.put = true;
  auto fast = std::async(std::launch::async, [&] { return concurrent.Handle(other); });
  CHECK(fast.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
  CHECK(fast.get().status.ok());
  resume.set_value();
  CHECK(slow.get().status.ok());
  CHECK(concurrent.Shutdown().ok());
  std::cout << "cache region lifecycle, idempotency, protocol and independent creation passed\n";
}
