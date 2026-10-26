// Build with the HCCP headers matching the server's installed libra.so.
// This queries device attributes; it does not create a Jetty or change QoS.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "hccp_ctx.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"

namespace kh = kvaf::direct_hccp;

template <typename T>
T Symbol(void* library, const char* name) {
  auto fn = reinterpret_cast<T>(dlsym(library, name));
  if (fn == nullptr) std::fprintf(stderr, "Missing HCCP symbol: %s\n", name);
  return fn;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: %s LOGICAL_DEVICE_ID EID_32_HEX_DIGITS\n", argv[0]);
    return 2;
  }
  char* end = nullptr;
  const long logical = std::strtol(argv[1], &end, 10);
  if (*argv[1] == '\0' || *end != '\0' || logical < 0 || logical > 65535) return 2;
  const std::string eid_text(argv[2]);
  if (eid_text.size() != 32 || eid_text.find_first_not_of("0123456789abcdefABCDEF") !=
                                   std::string::npos) return 2;
  HccpEid wanted{};
  for (std::size_t i = 0; i < sizeof(wanted.raw); ++i) {
    wanted.raw[i] = static_cast<uint8_t>(std::strtoul(eid_text.substr(i * 2, 2).c_str(), nullptr, 16));
  }

  kh::DirectHccpApi api{};
  if (!kh::LoadDirectHccp(&api)) return 1;
  auto get_num = Symbol<decltype(&RaGetDevEidInfoNum)>(api.library, "RaGetDevEidInfoNum");
  auto get_list = Symbol<decltype(&RaGetDevEidInfoList)>(api.library, "RaGetDevEidInfoList");
  auto ctx_init = Symbol<decltype(&RaCtxInit)>(api.library, "RaCtxInit");
  auto ctx_deinit = Symbol<decltype(&RaCtxDeinit)>(api.library, "RaCtxDeinit");
  auto get_attr = Symbol<decltype(&RaGetDevBaseAttr)>(api.library, "RaGetDevBaseAttr");
  if (!get_num || !get_list || !ctx_init || !ctx_deinit || !get_attr) return 1;
  Dl_info library_info{};
  if (dladdr(reinterpret_cast<void*>(get_attr), &library_info)) {
    std::fprintf(stderr, "HCCP_LIBRARY=%s\n", library_info.dli_fname);
  }
  int ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "aclInit failed: %d\n", ret);
    return 1;
  }
  ret = aclrtSetDevice(static_cast<int32_t>(logical));
  if (ret != ACL_SUCCESS) {
    std::fprintf(stderr, "aclrtSetDevice failed: %d\n", ret);
    return 1;
  }
  uint32_t physical = 0;
  if (!kh::GetCurrentNpuPhysicalDeviceId(&physical)) return 1;
  std::fprintf(stderr, "NPU logical_device=%ld physical_device=%u eid=%s\n",
               logical, physical, eid_text.c_str());
  kh::DirectHccpSession* session = nullptr;
  // Same HDC type and bootstrap as KVAF's AIV transport.
  if (!kh::AcquireDirectHccpSession(physical, 18U, &session)) return 1;
  const auto query = [&]() -> int {
    RaInfo info{};
    info.mode = NETWORK_OFFLINE;  // HDC query of the NPU, not peer-online Host URMA.
    info.phyId = physical;
    unsigned int count = 0;
    int rc = get_num(info, &count);
    if (rc != 0 || count == 0) {
      std::fprintf(stderr, "RaGetDevEidInfoNum failed: ret=%d count=%u\n", rc, count);
      return 1;
    }
    std::vector<HccpDevEidInfo> entries(count);
    rc = get_list(info, entries.data(), &count);
    if (rc != 0 || count > entries.size()) {
      std::fprintf(stderr, "RaGetDevEidInfoList failed: ret=%d count=%u\n", rc, count);
      return 1;
    }
    const HccpDevEidInfo* match = nullptr;
    for (unsigned int i = 0; i < count; ++i) {
      const auto& entry = entries[i];
      std::fprintf(stderr, "NPU_EID device=%.*s eid_index=%u eid=",
                   static_cast<int>(sizeof(entry.name)), entry.name, entry.eidIndex);
      for (uint8_t byte : entry.eid.raw) std::fprintf(stderr, "%02x", byte);
      std::fprintf(stderr, " die=%u chip=%u func=%u\n", entry.dieId, entry.chipId, entry.funcId);
      if (std::memcmp(entry.eid.raw, wanted.raw, sizeof(wanted.raw)) == 0) {
        if (match != nullptr) {
          std::fprintf(stderr, "EID matched multiple entries; refusing to pick a device.\n");
          return 1;
        }
        match = &entry;
      }
    }
    if (match == nullptr) {
      std::fprintf(stderr, "Requested EID is absent on physical NPU %u.\n", physical);
      return 1;
    }
    CtxInitCfg cfg{};
    cfg.mode = NETWORK_OFFLINE;
    CtxInitAttr init{};
    init.phyId = physical;
    init.ub.eidIndex = match->eidIndex;
    init.ub.eid = wanted;
    void* context = nullptr;
    rc = ctx_init(&cfg, &init, &context);
    if (rc != 0 || context == nullptr) {
      std::fprintf(stderr, "RaCtxInit failed: %d\n", rc);
      return 1;
    }
    DevBaseAttr attr{};
    rc = get_attr(context, &attr);
    if (rc == 0) {
      std::fprintf(stderr, "MAPPING_DEVICE=%.*s eid_index=%u die=%u chip=%u func=%u\n",
                   static_cast<int>(sizeof(match->name)), match->name, match->eidIndex,
                   attr.ub.dieId, attr.ub.chipId, attr.ub.funcId);
      std::puts("physical_device,eid,priority,sl,tp_type_mask");
      bool any_transport = false;
      for (unsigned int p = 0; p < MAX_PRIORITY_CNT; ++p) {
        const auto& mapping = attr.ub.priorityInfo[p];
        any_transport |= mapping.tpType.value != 0;
        std::printf("%u,%s,%u,%u,0x%x\n", physical, eid_text.c_str(), p,
                    mapping.SL, mapping.tpType.value);
      }
      if (!any_transport) {
        std::fprintf(stderr, "All transport masks are zero; verify SDK/driver support before interpreting this table.\n");
      }
    } else {
      std::fprintf(stderr, "RaGetDevBaseAttr failed: %d\n", rc);
    }
    const int close_rc = ctx_deinit(context);
    if (close_rc != 0) std::fprintf(stderr, "RaCtxDeinit failed: %d\n", close_rc);
    return rc == 0 && close_rc == 0 ? 0 : 1;
  };
  const int result = query();
  kh::ReleaseDirectHccpSession(session);
  // KVAF's RA/HDC bootstrap is process-owned; keep its DSOs and ACL runtime
  // alive until process exit, as the existing transport implementation does.
  return result;
}
