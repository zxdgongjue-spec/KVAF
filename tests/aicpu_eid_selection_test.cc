#include "transport_internal.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

using namespace kvaf::aicpu_urma;
using namespace kvaf::aicpu_urma::detail;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); std::abort(); } } while (0)

namespace {
std::array<urma_device_t, 3> devices{};
std::array<urma_device_t*, 3> device_list{};
std::array<std::array<urma_eid_info_t, 2>, 3> endpoints{};
urma_context_t context{};
std::vector<std::pair<int, unsigned>> attempts;
bool fail_first = false, fail_all = false, fail_enumeration = false;
unsigned freed_lists = 0, freed_eids = 0;

urma_device_t** Devices(int* count) {
  *count = device_list.size();
  return fail_enumeration ? nullptr : device_list.data();
}
void FreeDevices(urma_device_t**) { ++freed_lists; }
urma_eid_info_t* Eids(urma_device_t* device, std::uint32_t* count) {
  *count = 2;
  return endpoints[device - devices.data()].data();
}
void FreeEids(urma_eid_info_t*) { ++freed_eids; }
urma_context_t* Context(urma_device_t* device, std::uint32_t index) {
  const auto id = static_cast<int>(device - devices.data());
  attempts.emplace_back(id, index);
  if (fail_all || (fail_first && id == 1)) { errno = EIO; return nullptr; }
  for (const auto& eid : endpoints[id]) {
    if (eid.eid_index != index) continue;
    context.dev = device;
    context.eid_index = index;
    context.eid = eid.eid;
    return &context;
  }
  std::abort();
}
void Reset() {
  state = {};
  state.api.urma_get_device_list = Devices;
  state.api.urma_free_device_list = FreeDevices;
  state.api.urma_get_eid_list = Eids;
  state.api.urma_free_eid_list = FreeEids;
  state.api.urma_create_context = Context;
  attempts.clear();
  fail_first = fail_all = fail_enumeration = false;
  freed_lists = freed_eids = 0;
}
}

int main() {
  for (unsigned i = 0; i < devices.size(); ++i) {
    std::snprintf(devices[i].name, sizeof(devices[i].name), "ub%u", i);
    devices[i].type = i == 0 ? URMA_TRANSPORT_IB : URMA_TRANSPORT_UB;
    device_list[i] = &devices[i];
  }
  endpoints[1][0].eid.raw[15] = 1; // A different EID appears first.
  endpoints[1][0].eid_index = 0;
  endpoints[1][1].eid.raw[15] = 2;
  endpoints[1][1].eid_index = 3;
  // ub2 starts with a zero EID, which must be skipped.
  endpoints[2][1].eid.raw[15] = 2;
  endpoints[2][1].eid_index = 5;
  const auto* requested = endpoints[1][1].eid.raw;
  UmdkControlArgs args{};
  std::memcpy(args.local_eid, requested, 16);
  UmdkControlStatus status{};

  Reset();
  CHECK(CreateContext(args, status));
  CHECK((attempts == std::vector<std::pair<int, unsigned>>{{1, 3}}));
  CHECK(std::memcmp(status.eid, requested, 16) == 0 && freed_lists == 1 && freed_eids == 1);

  Reset(); fail_first = true; status = {};
  CHECK(CreateContext(args, status));
  CHECK((attempts == std::vector<std::pair<int, unsigned>>{{1, 3}, {2, 5}}));
  CHECK(status.result == 0 && status.error_number == 0 && freed_eids == 2);

  Reset(); status = {};
  std::array<std::uint8_t, 16> missing{}; missing[15] = 9;
  std::memcpy(args.local_eid, missing.data(), 16);
  CHECK(!CreateContext(args, status));
  CHECK(attempts.empty() && status.error_number == ENODEV && freed_lists == 1);
  Reset(); status = {};
  std::memset(args.local_eid, 0, 16);
  CHECK(!CreateContext(args, status));
  CHECK(attempts.empty() && freed_lists == 0);
  CHECK(std::strcmp(status.stage, "Host-selected EID is zero") == 0);
  std::memcpy(args.local_eid, requested, 16);

  Reset(); fail_all = true; status = {};
  CHECK(!CreateContext(args, status));
  CHECK(attempts.size() == 2 && status.error_number == EIO);

  Reset(); fail_enumeration = true; status = {};
  CHECK(!CreateContext(args, status));
  CHECK(std::strcmp(status.stage, "urma_get_device_list") == 0);

  std::puts("PASS Host EID resolves device/index automatically; retries, missing/zero EIDs and failures");
}
