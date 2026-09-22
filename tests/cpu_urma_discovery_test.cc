#include "cpu_urma/cpu_urma_discovery.h"
#include <urma_api.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
urma_device_t devices[3]{};
urma_device_t* list[3] = {devices, devices + 1, devices + 2};
urma_eid_info_t eids[2][3]{};
int init_rc = 0, uninit_count = 0, freed_devices = 0, freed_eids = 0;
bool list_missing = false;
void Check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void Reset() {
  init_rc = uninit_count = freed_devices = freed_eids = 0;
  list_missing = false;
  std::memset(devices, 0, sizeof(devices));
  std::memset(eids, 0, sizeof(eids));
  std::strcpy(devices[0].name, "ub1"); devices[0].type = URMA_TRANSPORT_UB;
  std::strcpy(devices[1].name, "ub0"); devices[1].type = URMA_TRANSPORT_UB;
  devices[2].type = URMA_TRANSPORT_IB;
  eids[0][0].eid_index = 7; eids[0][0].eid.raw[15] = 7;
  eids[0][1].eid_index = 2; eids[0][1].eid.raw[15] = 2;
  eids[1][0].eid_index = 3; eids[1][0].eid.raw[15] = 3;
  // Remaining entries are zero EIDs and must not become candidates.
}
}  // namespace

// Link the actual discovery implementation against a small provider double.
// The real UMDK declarations check the signatures; no liburma/device is used.
extern "C" {
urma_status_t urma_init(urma_init_attr_t*) { return static_cast<urma_status_t>(init_rc); }
urma_status_t urma_uninit() { ++uninit_count; return URMA_SUCCESS; }
urma_device_t** urma_get_device_list(int* count) { *count = 3; return list_missing ? nullptr : list; }
void urma_free_device_list(urma_device_t**) { ++freed_devices; }
urma_eid_info_t* urma_get_eid_list(urma_device_t* device, std::uint32_t* count) {
  *count = 3;
  return device == devices ? eids[0] : eids[1];
}
void urma_free_eid_list(urma_eid_info_t*) { ++freed_eids; }
}

int main() {
  try {
    Reset();
    {
      kvaf::CpuUrmaDiscovery discovery;
      Check(discovery.Initialize().ok(), "discovery failed");
      const auto& found = discovery.candidates();
      Check(found.size() == 3 && found[0].device_name == "ub0" && found[0].eid_index == 3 &&
            found[1].device_name == "ub1" && found[1].eid_index == 2 && found[2].eid_index == 7,
            "candidate order/filter changed");
      Check(freed_devices == 1 && freed_eids == 2, "provider lists leaked");
      Check(!discovery.Initialize().ok(), "double initialization accepted");
      discovery.Shutdown(); discovery.Shutdown();
      Check(uninit_count == 1 && discovery.candidates().empty(), "owned lifetime cleanup failed");
    }
    Reset(); init_rc = EEXIST;
    {
      kvaf::CpuUrmaDiscovery discovery;
      Check(discovery.Initialize("ub1", true, 7).ok() && discovery.candidates().size() == 1 &&
            discovery.candidates()[0].eid[15] == 7, "explicit filters failed");
    }
    Check(uninit_count == 0, "borrowed URMA lifetime was uninitialized");
    Reset(); list_missing = true;
    { kvaf::CpuUrmaDiscovery discovery; Check(!discovery.Initialize().ok(), "missing list accepted"); }
    Check(uninit_count == 1, "failure leaked owned URMA initialization");
    Reset();
    { kvaf::CpuUrmaDiscovery discovery; Check(!discovery.Initialize("missing").ok(), "empty filters accepted"); }
    Check(uninit_count == 1 && freed_devices == 1, "empty discovery leaked resources");
    std::cout << "CPU URMA discovery: PASS\n"; return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
