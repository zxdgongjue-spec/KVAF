// Exercise the real CPU initialization/channel diagnostic code with SDK types.
// The four query symbols are exported by this executable and resolved through
// real dlsym. If an unconfirmed ABI query runs, its fake provider deliberately
// writes beyond the SDK structure, reproducing the old-header/new-library risk.
// Only the two required private bootstrap symbol lookups are substituted.
// Compile this target independently of kvaf_cpu_runtime so its default case
// cannot accidentally inherit KVAF_CPU_URMA_QUERY_ABI_CONFIRMED=1.
#include <urma_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static int query_calls[4] = {};
static bool destructive_provider = true;
static void* TestDlsym(void*, const char*);
#define dlsym TestDlsym
#include "../runtime/src/cpu_runtime.cc"
#undef dlsym

extern "C" urma_device_t* urma_get_device_by_name(char*) {
  static urma_device_t dev{};
  dev.type = URMA_TRANSPORT_UB;
  return &dev;
}

extern "C" urma_context_t* urma_create_context(urma_device_t* dev, uint32_t eid) {
  static urma_context_t ctx{};
  ctx.dev = dev;
  ctx.eid_index = eid;
  ctx.eid.raw[15] = 1;
  return &ctx;
}

extern "C" urma_status_t urma_delete_context(urma_context_t*) { return URMA_SUCCESS; }

static int FakeTpList(urma_context_t*, kvaf::UrmaExtGetTpCfg*, uint32_t*,
                      kvaf::UrmaExtTpInfo*) {
  return 0;
}

static urma_target_jetty_t* FakeImport(urma_context_t*, kvaf::UrmaExtRjetty*,
                                      urma_token_t*, kvaf::UrmaExtActiveTpCfg*) {
  return nullptr;
}

template <class T>
static void WriteOutput(T* output) {
  const auto bytes = sizeof(*output) + (destructive_provider ? 512U : 0U);
  std::memset(output, 0, bytes);
  if (destructive_provider) std::abort();
}

extern "C" urma_status_t urma_query_device(urma_device_t*, urma_device_attr_t* attr) {
  ++query_calls[0];
  WriteOutput(attr);
  attr->port_cnt = 1;
  attr->dev_cap.max_jetty = 71;
  return URMA_SUCCESS;
}

extern "C" urma_status_t urma_query_jetty(urma_jetty_t*, urma_jetty_cfg_t* cfg,
                                          urma_jetty_attr_t* attr) {
  ++query_calls[1];
  WriteOutput(cfg);
  *attr = {};
  cfg->jfs_cfg.depth = 72;
  return URMA_SUCCESS;
}

extern "C" urma_status_t urma_query_jfr(urma_jfr_t*, urma_jfr_cfg_t* cfg,
                                        urma_jfr_attr_t* attr) {
  ++query_calls[2];
  WriteOutput(cfg);
  *attr = {};
  cfg->depth = 73;
  return URMA_SUCCESS;
}

#if defined(KVAF_HAVE_URMA_TP_ATTR_API) && KVAF_HAVE_URMA_TP_ATTR_API
extern "C" urma_status_t urma_get_tp_attr(const urma_context_t*, const uint64_t,
                                          uint8_t* count, uint32_t* bitmap,
                                          urma_tp_attr_value_t* attr) {
  ++query_calls[3];
  WriteOutput(attr);
  *count = 1;
  *bitmap = 1U << 10U;
  attr->sl = 7;
  return URMA_SUCCESS;
}
#else
extern "C" int urma_get_tp_attr(const void*, uint64_t, uint8_t*, uint32_t*, void* attr) {
  ++query_calls[3];
  std::memset(attr, 0, 1024);
  return 0;
}
#endif

static void* TestDlsym(void*, const char* name) {
  if (!std::strcmp(name, "urma_get_tp_list")) return reinterpret_cast<void*>(&FakeTpList);
  if (!std::strcmp(name, "urma_import_jetty_ex")) return reinterpret_cast<void*>(&FakeImport);
  return ::dlsym(RTLD_DEFAULT, name);
}

int main() {
  setenv("KVAF_LINK_DIAGNOSTICS", "1", 1);
#if defined(KVAF_CPU_URMA_TEST_EXPECT_CONFIRMED) && KVAF_CPU_URMA_TEST_EXPECT_CONFIRMED
  constexpr bool expect_confirmed = true;
#else
  constexpr bool expect_confirmed = false;
#endif
  if (kvaf::detail::CpuUrmaQueryAbiConfirmed() != expect_confirmed) return 2;
  destructive_provider = !expect_confirmed;
  for (const char* name : {"urma_query_device", "urma_query_jetty", "urma_query_jfr",
                           "urma_get_tp_attr"}) {
    if (::dlsym(RTLD_DEFAULT, name) == nullptr) {
      std::fprintf(stderr, "missing exported fake provider: %s\n", name);
      return 3;
    }
  }
  kvaf::UrmaResources resources;
  if (!resources.Initialize("fake0", 0)) return 1;
  urma_jetty_t local{};
  urma_jfr_t jfr{};
  urma_jfc_t jfc{};
  urma_target_jetty_t remote{};
  local.jetty_id.id = 31;
  jfr.jfr_id.id = 32;
  jfc.jfc_id.id = 33;
  remote.id.id = 34;
  kvaf::peer::CreateChannelRequest request{};
  request.local_psn = 44;
  request.local_tp_handle = 45;
  resources.DiagnoseChannel(1, 0, &local, &jfr, &jfc, &remote, request, 46, 47);
  const int expected = destructive_provider ? 0 : 1;
  for (int i = 0; i < 4; ++i) {
#if defined(KVAF_HAVE_URMA_TP_ATTR_API) && KVAF_HAVE_URMA_TP_ATTR_API
    const int want = expected;
#else
    const int want = i == 3 ? 0 : expected;
#endif
    if (query_calls[i] != want) {
      std::fprintf(stderr, "query[%d] expected calls=%d actual calls=%d\n", i, want, query_calls[i]);
      return 10 + i;
    }
  }
  std::printf("PASS device_attr_bytes=%zu ABI_confirmed=%d query_calls=%d,%d,%d,%d\n",
              sizeof(urma_device_attr_t), !destructive_provider, query_calls[0],
              query_calls[1], query_calls[2], query_calls[3]);
}
