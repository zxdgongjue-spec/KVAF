#include "cpu_service/urma/cpu_urma_discovery.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <utility>
#include <tuple>
#include <mutex>

#include "urma_transport.h"

namespace kvaf {
namespace {
std::mutex provider_mutex;
std::size_t provider_users = 0;
bool provider_owned = false;
}  // namespace
CpuUrmaDiscovery::CpuUrmaDiscovery(std::string device, bool has_eid_index, std::uint32_t eid_index, Scope scope)
    : device_scope_(scope), scope_status_(scope == Scope::ServiceAffinity ? scope_.Capture() : Status::OK()), device_(std::move(device)), has_eid_index_(has_eid_index),
      eid_index_(eid_index) {}

Status CpuUrmaDiscovery::Initialize() {
  std::lock_guard<std::mutex> snapshot(mutex_);
  if (!scope_status_.ok()) return scope_status_;
  if (!candidates_.empty()) return Status::OK();
  if (!initialized_) {
    std::lock_guard<std::mutex> lock(provider_mutex);
    if (!provider_users && !provider_owned) {
      urma_init_attr_t init{};
      const int rc = urma_init(&init);
      if (rc != URMA_SUCCESS && rc != EEXIST && rc != -EEXIST)
        return Status::Internal("urma_init failed (rc=" + std::to_string(rc) + ")");
      provider_owned = rc == URMA_SUCCESS;
    }
    ++provider_users;
    initialized_ = true;
  }
  std::vector<CpuUrmaCandidate> discovered;
  const bool enumerated = urma::EnumerateEndpoints(
      urma::NativeApi{}, device_.c_str(), has_eid_index_ ? eid_index_ : UINT32_MAX,
      [&](auto* device, const auto& info) {
        CpuUrmaCandidate candidate{device->name, info.eid_index, {}};
        std::copy_n(info.eid.raw, candidate.eid.size(), candidate.eid.begin());
        discovered.push_back(std::move(candidate));
        return false;
      }, [&](auto* device) {
        if (device_scope_ == Scope::AllDevices || scope_.Allows(device->name)) return true;
        std::fprintf(stderr, "[KVAF_CPU] NUMA_DEVICE_SKIPPED device=%s reason=outside_or_unknown_chip\n",
                     device->name);
        return false;
      });
  if (!enumerated) return Status::NotFound("urma_get_device_list returned no devices");
  if (discovered.empty()) return Status::NotFound("no CPU UB/EID candidates match NUMA scope and device/index filters");
  std::sort(discovered.begin(), discovered.end(), [](const auto& a, const auto& b) {
    return std::tie(a.device_name, a.eid_index) < std::tie(b.device_name, b.eid_index);
  });
  candidates_ = std::move(discovered);
  return Status::OK();
}

CpuUrmaDiscovery::~CpuUrmaDiscovery() {
  std::lock_guard<std::mutex> lock(provider_mutex);
  if (initialized_ && --provider_users == 0 && provider_owned) {
    // If uninit fails retain ownership; the next initialization can acquire it.
    if (urma_uninit() == URMA_SUCCESS) provider_owned = false;
  }
  initialized_ = false;
}
}  // namespace kvaf
