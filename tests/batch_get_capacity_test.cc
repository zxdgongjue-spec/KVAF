#include "cache_region_service.h"
#include "session_access_operator.h"
#include "npu_runtime_config.h"
#include "gather_workspace.h"
#include <array>
#include <cstdlib>
#include <iostream>

using namespace kvaf;
#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::abort(); \
} } while (false)

struct Allocation final : execution::Allocation {
  Status Close() override { return Status::OK(); }
};
struct Session final : execution::Session {
  Status Open(const execution::Request&, execution::Allocation& pool, peer::PeerRegion* out) override {
    *out = pool.descriptor;
    return Status::OK();
  }
  Status Call(const std::vector<std::uint8_t>&, std::vector<std::uint8_t>*) override { return Status::OK(); }
  Status Close() override { return Status::OK(); }
};

int main() {
  ExecutionResources configured;
  CHECK(!configured.gather_read_wave_wrs);
  CHECK(!configured.gather_address_groups_per_post);
  configured.gather_read_wave_wrs = 64;
  configured.gather_address_groups_per_post = 9;
  const auto resources = detail::ResolveExecutionResources(Transport::Urma, Executor::Aicpu,
                                                           true, true, &configured);
  CHECK(resources.gather_read_wave_wrs && *resources.gather_read_wave_wrs == 64);
  CHECK(resources.gather_address_groups_per_post == 9);
  uint32_t groups{};
  CHECK(ResolveGatherAddressGroupsPerPost(std::nullopt, nullptr, &groups) && groups == 3);
  CHECK(ResolveGatherAddressGroupsPerPost(std::nullopt, "12", &groups) && groups == 12);
  CHECK(ResolveGatherAddressGroupsPerPost(7, "12", &groups) && groups == 7);
  CHECK(ResolveGatherAddressGroupsPerPost(3, "invalid", &groups) && groups == 3);
  CHECK(ResolveGatherAddressGroupsPerPost(UINT32_MAX, nullptr, &groups) && groups == UINT32_MAX);
  CHECK(ResolveGatherAddressGroupsPerPost(std::nullopt, "99999999999999999999999", &groups) && groups == UINT32_MAX);
  CHECK(!ResolveGatherAddressGroupsPerPost(0, "12", &groups));
  for (const char* value : {"", "0", "-1", "3x", "uniform:3"})
    CHECK(!ResolveGatherAddressGroupsPerPost(std::nullopt, value, &groups));
  std::array<uint8_t, 64> memory{};
  execution::CacheRegionService service(execution::kUrma | execution::kFabric,
      [&](const execution::Request& q, std::unique_ptr<execution::Allocation>* out) {
        auto allocation = std::make_unique<Allocation>();
        allocation->descriptor.base = reinterpret_cast<uintptr_t>(memory.data());
        allocation->descriptor.length = q.logical_bytes;
        *out = std::move(allocation);
        return Status::OK();
      }, [](const execution::Request&) { return std::make_unique<Session>(); });
  execution::Request q;
  q.epoch = service.Handle(q).epoch;
  q.client = q.request = 1; q.op = execution::Op::CreateRegion;
  q.mode = TransportMode::UrmaAndUbmem; q.logical_bytes = memory.size(); q.slot_size = 1;
  const auto region = service.Handle(q);
  CHECK(region.status.ok());
  q.op = execution::Op::OpenSession; q.region = region.region;
  auto check = [&](Transport t, Executor e, GetAlgorithm a, bool put, bool dynamic) {
    q.transport = t; q.executor = e; q.algorithm = a; q.put = put;
    auto factory = [](const PeerExecutionResources::Links&) -> std::unique_ptr<AccessOperator> { return {}; };
    SessionAccessOperator op({}, q, {}, factory, 65536);
    CHECK(op.ValidateRequestCapacity(65536).ok());
    CHECK(op.ValidateRequestCapacity(65537).ok() == dynamic);
    CHECK(op.ValidateRequestCapacity(UINT32_MAX).ok() == dynamic);
    SessionAccessOperator small({}, q, {}, factory, 128);
    CHECK(small.ValidateRequestCapacity(128).ok());
    CHECK(small.ValidateRequestCapacity(129).ok() == dynamic);
    BatchGetRequest input;
    input.indices.bytes = std::size_t(200000) * sizeof(int32_t);
    uint32_t count{};
    CHECK(op.DescribeRequirements(input, &count).ok() == dynamic);
    CHECK(count == 200000);
    input.indices.bytes = (std::size_t(UINT32_MAX) + 1U) * sizeof(int32_t);
    CHECK(!op.DescribeRequirements(input, &count).ok());
    for (uint32_t capacity : {65536U, 65537U, 1000000U, UINT32_MAX}) {
      ++q.request; q.request_capacity = capacity;
      const auto response = service.Handle(q);
      CHECK(response.status.ok() == (capacity == 65536 || dynamic));
    }
  };
  check(Transport::Urma, Executor::Aiv, GetAlgorithm::Direct, false, true);
  check(Transport::Ubmem, Executor::Aiv, GetAlgorithm::Direct, false, true);
  check(Transport::Urma, Executor::Aicpu, GetAlgorithm::Gather, false, true);
  check(Transport::Urma, Executor::Aicpu, GetAlgorithm::Scatter, false, true);
  check(Transport::Urma, Executor::Aiv, GetAlgorithm::Direct, true, false);
  check(Transport::Ubmem, Executor::Aiv, GetAlgorithm::Direct, true, false);
  check(Transport::Urma, Executor::Aicpu, GetAlgorithm::Direct, true, false);
  CHECK(service.Shutdown().ok());
  std::cout << "AIV Direct Get capacity: client/server limits and ABI narrowing passed\n";
}
