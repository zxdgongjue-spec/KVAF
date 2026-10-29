// Isolated diagnostic executable: use the actual CPU service allocation and
// protocol while intercepting only its KV MR registration in this translation
// unit. Production binaries and source files are not changed.
#include "access_compat.h"
#include "wire.h"
#include <arpa/inet.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace probe {
using namespace ddr_probe;
std::string mode;
std::uint64_t allocation_bytes;
struct Registration {
  urma_context_t* context{};
  urma_context_t* extra_context{};
  urma_target_seg_t* remote{};
  urma_target_seg_t* local{};
  urma_token_id_t* local_token{};
  std::uint64_t va{}, bytes{};
  std::uint32_t token_value{};
  bool armed{};
};
std::map<urma_target_seg_t*, Registration> registrations;
[[noreturn]] void Fatal(const char* text) {
  std::fprintf(stderr, "PROBE_FATAL %s errno=%d; process teardown required\n", text, errno);
  // Never continue into CPU service's unchecked free after failed deregistration.
  std::_Exit(2);
}
void Check(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
constexpr std::uint32_t kMaxReadChunkBytes = 65536;  // Diagnostic software limit, not a device capability.
std::uint32_t ParseReadChunkBytes(const std::string& text) {
  constexpr const char* error = "read chunk bytes must be a decimal integer in [1,65536]";
  Check(!text.empty(), error);
  std::uint32_t value = 0;
  for (char c : text) {
    Check(c >= '0' && c <= '9', error);
    const auto digit = static_cast<std::uint32_t>(c - '0');
    Check(value <= (kMaxReadChunkBytes - digit) / 10, error);
    value = value * 10 + digit;
  }
  Check(value != 0, error);
  return value;
}
void ReleaseAlias(Registration& r) {
  if (r.local && r.local != r.remote && urma_unregister_seg(r.local)) Fatal("unregister local alias");
  if (r.local_token && urma_free_token_id(r.local_token)) Fatal("free alias token");
  if (r.extra_context && urma_delete_context(r.extra_context)) Fatal("delete alias context");
  r.local = nullptr; r.local_token = nullptr; r.extra_context = nullptr;
}
urma_target_seg_t* Register(urma_context_t* ctx, urma_seg_cfg_t* original) {
  if (original->len != allocation_bytes || original->flag.bs.access != kRemoteAccess)
    return urma_register_seg(ctx, original);
  Registration r;
  r.context = ctx; r.va = original->va; r.bytes = original->len;
  r.token_value = original->token_value.token;
  auto remote_cfg = *original;
  const bool dual = mode.rfind("dual", 0) == 0;
  if (mode == "single-pinned" || mode == "dual-both-pinned" || mode == "single-local-write-pinned")
    remote_cfg.flag.bs.non_pin = 0;
  if (mode == "single-plus-bit0" || mode == "single-local-write-pinned" || mode == "dual-compatible")
    remote_cfg.flag.bs.access |= kLocalWrite;
  auto local_cfg = *original;
  local_cfg.flag.bs.access = kLocalWrite;
  local_cfg.flag.bs.non_pin = 0;
  auto local_register = [&]() {
    if (mode == "dual-separate-context") {
      r.extra_context = urma_create_context(ctx->dev, ctx->eid_index);
      if (!r.extra_context) return false;
      r.context = r.extra_context;
    }
    r.local_token = urma_alloc_token_id(r.context);
    if (!r.local_token) return false;
    local_cfg.token_id = r.local_token;
    errno = 0;
    r.local = urma_register_seg(r.context, &local_cfg);
    const int saved = errno;
    std::fprintf(stderr, "REGISTER role=pull va=0x%llx bytes=%llu access=0x%x non_pin=%u flags=0x%x mr=%p context=%p errno=%d\n",
      (unsigned long long)r.va, (unsigned long long)r.bytes, local_cfg.flag.bs.access,
      local_cfg.flag.bs.non_pin, local_cfg.flag.value, (void*)r.local, (void*)r.context, saved);
    errno = saved;
    return r.local != nullptr;
  };
  if (mode == "dual-reverse" && !local_register()) {
    const int saved = errno; ReleaseAlias(r); errno = saved; return nullptr;
  }
  errno = 0;
  r.remote = urma_register_seg(ctx, &remote_cfg);
  const int saved = errno;
  std::fprintf(stderr, "REGISTER role=decode case=%s va=0x%llx bytes=%llu access=0x%x non_pin=%u flags=0x%x mr=%p context=%p errno=%d\n",
    mode.c_str(), (unsigned long long)r.va, (unsigned long long)r.bytes,
    remote_cfg.flag.bs.access, remote_cfg.flag.bs.non_pin, remote_cfg.flag.value,
    (void*)r.remote, (void*)ctx, saved);
  if (!r.remote) { ReleaseAlias(r); errno = saved; return nullptr; }
  if (dual && mode != "dual-reverse" && !local_register()) {
    const int failure = errno;
    ReleaseAlias(r);
    if (urma_unregister_seg(r.remote)) Fatal("rollback remote MR");
    errno = failure; return nullptr;
  }
  if (!dual) r.local = r.remote;
  registrations.emplace(r.remote, r);
  return r.remote;
}
urma_status_t Unregister(urma_target_seg_t* mr) {
  auto it = registrations.find(mr);
  if (it != registrations.end()) {
    if (it->second.armed) Fatal("Decode released an armed DDR allocation before CLOSE");
    ReleaseAlias(it->second);
    registrations.erase(it);
  }
  const auto rc = urma_unregister_seg(mr);
  if (rc) Fatal("unregister exported MR");
  return rc;
}
}  // namespace probe

#define urma_register_seg ::probe::Register
#define urma_unregister_seg ::probe::Unregister
#include PROBE_CPU_SOURCE
#undef urma_register_seg
#undef urma_unregister_seg

// This harness selects the allocation through the v7 region descriptor.
// AICPU v6 also reserves gather control/staging bytes and needs its own adapter.
static_assert(kvaf::peer::kProtocolVersion == 7, "Use the shared-ddr v7 checkout and matching Decode build");

namespace probe {
// Deployed expanded CTP ABI, also used by KVAF CPU push transport.
struct TpQuery { std::uint32_t flag, mode; std::uint8_t local[16], peer[16]; };
struct TpInfo { std::uint64_t handle; };
struct RemoteJetty {
  urma_jetty_id_t id;
  std::uint32_t mode, policy, type, padding;
  std::uint64_t flag, stag, dtag;
  std::uint32_t tp_type, padding2;
};
struct ActiveTp { std::uint64_t local, peer, tag; std::uint32_t tx, rx; std::uint64_t reserved; };
#if defined(PROBE_EXTENDED_SGE)
// Explicit CPU-push reference layout; validate the deployed provider separately.
struct Sge { std::uint64_t addr; std::uint32_t len; urma_target_seg_t* tseg; void* user_tseg; };
static_assert(sizeof(Sge) == 32);
#else
using Sge = urma_sge_t;  // 24 B in br_UMDK_24.12.0_LTS.
#endif
static_assert(sizeof(urma_seg_t) == 48 && sizeof(urma_jetty_id_t) == 24);
static_assert(sizeof(TpQuery) == 40 && sizeof(RemoteJetty) == 72 && sizeof(ActiveTp) == 40);
static_assert(offsetof(Sge, tseg) == 16);
using GetTp = int (*)(urma_context_t*, TpQuery*, std::uint32_t*, TpInfo*);
using ImportJetty = urma_target_jetty_t* (*)(urma_context_t*, RemoteJetty*, urma_token_t*, ActiveTp*);

struct SourceDescriptor {
  urma_seg_t segment{};
  RemoteJetty remote{};
  TpQuery tp_query{};
};

SourceDescriptor DecodeSourceDescriptor(const Message& request, std::uint64_t expected_bytes,
                                        const urma_eid_t& local_eid) {
  Check(request.segment_bytes == sizeof(urma_seg_t) && request.key_bytes == 32 &&
    request.tp && request.psn && request.token, "invalid HBM descriptor/Jetty ABI");
  SourceDescriptor descriptor;
  std::memcpy(&descriptor.segment, request.segment, sizeof(descriptor.segment));
  const auto& segment = descriptor.segment;
  Check(request.bytes == expected_bytes && request.base <= UINT64_MAX - request.bytes &&
    request.base >= segment.ubva.va && request.bytes <= segment.len &&
    request.base - segment.ubva.va <= segment.len - request.bytes,
    "HBM range does not cover the full guarded allocation");
  std::memcpy(&descriptor.remote.id, request.key, sizeof(descriptor.remote.id));
  std::memcpy(&descriptor.remote.mode, request.key + sizeof(descriptor.remote.id),
    sizeof(descriptor.remote.mode));
  Check(descriptor.remote.mode == URMA_TM_RM, "source Jetty must use RM");
  Check(std::any_of(descriptor.remote.id.eid.raw, descriptor.remote.id.eid.raw + 16,
    [](std::uint8_t byte) { return byte != 0; }), "source Jetty EID must be nonzero");
  Check(std::memcmp(segment.ubva.eid.raw, descriptor.remote.id.eid.raw, 16) == 0,
    "HBM segment EID differs from exported Jetty EID");
  // As in CpuService::CreateChannel, the configured CTP route and the resource
  // identity in the Jetty key are separate inputs. Never rewrite either key.
  descriptor.tp_query.flag = 1;
  descriptor.tp_query.mode = URMA_TM_RM;
  Check(std::any_of(request.eid, request.eid + 16,
    [](std::uint8_t byte) { return byte != 0; }), "source route EID must be nonzero");
  std::memcpy(descriptor.tp_query.local, local_eid.raw, 16);
  std::memcpy(descriptor.tp_query.peer, request.eid, 16);
  return descriptor;
}

std::string SourceEidHex(const std::uint8_t* eid) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(32, '0');
  for (std::size_t i = 0; i < 16; ++i) {
    result[2*i] = digits[eid[i] >> 4];
    result[2*i+1] = digits[eid[i] & 15];
  }
  return result;
}

// Spare storage catches a larger deployed output layout. It is a diagnostic
// boundary, not an ABI adapter: any write past the declared value is rejected.
template <class T> struct NativeBuffer {
  alignas(T) std::uint8_t before[64];
  T value{};
  std::uint8_t after[4096];
  NativeBuffer() {
    static_assert(64 % alignof(T) == 0);
    static_assert(offsetof(NativeBuffer<T>, value) == 64);
    static_assert(offsetof(NativeBuffer<T>, after) == 64 + sizeof(T));
    std::fill(std::begin(before), std::end(before), 0xa7);
    std::fill(std::begin(after), std::end(after), 0xa7);
  }
};

template <class T> void CheckNativeBuffer(const char* operation, const NativeBuffer<T>& buffer,
                                         int rc, int error, bool log_healthy = true) {
  bool before_ok = true, after_ok = true;
  const char* side = "none";
  std::ptrdiff_t first_bad = 0;
  const volatile std::uint8_t* leading = buffer.before;
  const volatile std::uint8_t* trailing = buffer.after;
  for (std::size_t i = 0; i < sizeof(buffer.before); ++i) {
    if (leading[i] != 0xa7) {
      if (before_ok) { side = "before"; first_bad = static_cast<std::ptrdiff_t>(i) - 64; }
      before_ok = false;
    }
  }
  for (std::size_t i = 0; i < sizeof(buffer.after); ++i) {
    if (trailing[i] != 0xa7) {
      if (before_ok && after_ok) { side = "after"; first_bad = sizeof(T) + i; }
      after_ok = false;
    }
  }
  if (log_healthy || !before_ok || !after_ok)
    std::fprintf(stderr, "NATIVE_ABI_GUARD op=%s value=%p value_bytes=%zu before_ok=%u after_ok=%u "
      "bad_side=%s first_bad_offset=%td rc=%d errno=%d\n", operation, (const void*)&buffer.value,
      sizeof(T), unsigned(before_ok), unsigned(after_ok), side, first_bad, rc, error);
  if (!before_ok || !after_ok)
    throw std::runtime_error(std::string("native ABI output boundary overwritten at ") + operation);
}

void CheckPullQueryState(const char* stage, const TpQuery& actual, const TpQuery& expected) {
  const bool same = std::memcmp(&actual, &expected, sizeof(actual)) == 0;
  std::fprintf(stderr, "PULL_CTP_STATE stage=%s unchanged=%u query=%p flag=0x%x mode=%u "
    "local_eid=%s peer_eid=%s\n", stage, unsigned(same), (const void*)&actual, actual.flag,
    actual.mode, SourceEidHex(actual.local).c_str(), SourceEidHex(actual.peer).c_str());
  if (!same)
    throw std::runtime_error(std::string("pull CTP parameters overwritten at ") + stage);
}

struct PullCtpResult {
  int rc{};
  std::uint32_t count{};
  std::uint64_t handle{};
  int error{};
  bool ok() const { return rc == 0 && count != 0 && handle != 0; }
  const char* outcome() const {
    if (rc != 0) return "api_error";
    if (count == 0) return "empty_list";
    if (handle == 0) return "zero_handle";
    return "ok";
  }
};

PullCtpResult QueryPullCtp(GetTp query, urma_context_t* ctx, TpQuery config) {
  const auto input = std::make_unique<TpQuery>(config);
  std::fprintf(stderr, "PULL_CTP_QUERY_BEGIN ctx=%p flag=0x%x mode=%u local_eid=%s peer_eid=%s "
    "config_bytes=%zu count_bytes=%zu info_bytes=%zu\n", (void*)ctx, input->flag, input->mode,
    SourceEidHex(input->local).c_str(), SourceEidHex(input->peer).c_str(), sizeof(TpQuery),
    sizeof(std::uint32_t), sizeof(TpInfo));
  Check(input->flag == 1 && input->mode == URMA_TM_RM &&
    std::any_of(input->local, input->local + 16, [](std::uint8_t b) { return b != 0; }) &&
    std::any_of(input->peer, input->peer + 16, [](std::uint8_t b) { return b != 0; }),
    "invalid or overwritten pull CTP input (see PULL_CTP_QUERY_BEGIN)");
  auto query_config = std::make_unique<NativeBuffer<TpQuery>>();
  auto count = std::make_unique<NativeBuffer<std::uint32_t>>();
  auto info = std::make_unique<NativeBuffer<TpInfo>>();
  query_config->value = config;
  count->value = 1;
  errno = 0;
  const int rc = query(ctx, &query_config->value, &count->value, &info->value);
  const int saved_errno = errno;
  CheckNativeBuffer("urma_get_tp_list.config", *query_config, rc, saved_errno);
  CheckNativeBuffer("urma_get_tp_list.count", *count, rc, saved_errno);
  CheckNativeBuffer("urma_get_tp_list.info", *info, rc, saved_errno);
  const PullCtpResult result{rc, count->value, info->value.handle, saved_errno};
  const bool config_changed = std::memcmp(&query_config->value, input.get(), sizeof(TpQuery)) != 0;
  std::fprintf(stderr, "PULL_CTP_QUERY result=%s ctx=%p flag=0x%x mode=%u "
    "local_eid=%s peer_eid=%s capacity=1 rc=%d count=%u tp=0x%llx errno=%d config_changed=%u\n",
    result.outcome(), (void*)ctx, input->flag, input->mode,
    SourceEidHex(input->local).c_str(), SourceEidHex(input->peer).c_str(),
    result.rc, result.count, (unsigned long long)result.handle, result.error, unsigned(config_changed));
  errno = saved_errno;
  return result;
}

bool MrContains(const urma_target_seg_t& mr, std::uint64_t address, std::uint32_t length) {
  const auto base = mr.seg.ubva.va, bytes = mr.seg.len;
  return length && base <= UINT64_MAX - bytes && address >= base &&
    address <= UINT64_MAX - length && length <= bytes && address - base <= bytes - length;
}

void AuditReadMrs(const urma_target_seg_t* source, const urma_target_seg_t* destination,
                  urma_context_t* context, std::uint64_t src, std::uint64_t dst,
                  std::uint32_t length, const char* target, bool verbose) {
  Check(source && destination, "READ preflight: null MR");
  const bool src_range = MrContains(*source, src, length);
  const bool dst_range = MrContains(*destination, dst, length);
  const bool src_context = source->urma_ctx == context;
  const bool dst_context = destination->urma_ctx == context;
  if (verbose || !src_range || !dst_range || !src_context || !dst_context) {
    const auto print = [&](const char* role, const urma_target_seg_t& mr, std::uint64_t address) {
      std::fprintf(stderr, "READ_MR target=%s role=%s layout=compiled_header mr=%p context=%p "
        "expected_ctx=%p eid=%s uasid=0x%x base=0x%llx bytes=%llu attr=0x%x access=0x%x "
        "non_pin=%u mva=0x%llx token_id=0x%x token_handle=%p wr_va=0x%llx wr_bytes=%u\n",
        target, role, (const void*)&mr, (void*)mr.urma_ctx, (void*)context,
        SourceEidHex(mr.seg.ubva.eid.raw).c_str(), mr.seg.ubva.uasid,
        (unsigned long long)mr.seg.ubva.va, (unsigned long long)mr.seg.len, mr.seg.attr.value,
        mr.seg.attr.bs.access, mr.seg.attr.bs.non_pin, (unsigned long long)mr.mva,
        mr.seg.token_id, (void*)mr.token_id, (unsigned long long)address, length);
    };
    print("source", *source, src); print("destination", *destination, dst);
    std::fprintf(stderr, "READ_PREFLIGHT target=%s src_range_ok=%u dst_range_ok=%u "
      "src_context_ok=%u dst_context_ok=%u src_read_bit=%u dst_bit0=%u\n", target,
      unsigned(src_range), unsigned(dst_range), unsigned(src_context), unsigned(dst_context),
      unsigned((source->seg.attr.bs.access & 2U) != 0),
      unsigned((destination->seg.attr.bs.access & 1U) != 0));
  }
  // These checks validate the reported layout, not hardware translation or
  // authorization. Do not impose old LOCAL_WRITE semantics on a deployed MR.
  Check(src_range, "READ preflight: source outside reported MR range");
  Check(dst_range, "READ preflight: destination outside reported MR range");
  Check(src_context && dst_context, "READ preflight: MR context differs from pull context");
}

struct State {
  Registration* region{};
  urma_target_seg_t* source{};
  urma_jfc_t* cq{};
  urma_jfr_t* rq{};
  urma_jetty_t* jetty{};
  urma_target_jetty_t* peer{};
  std::uint64_t source_base{}, ready{}, acknowledged{}, sequence{};
  std::uint32_t max_bytes{};
  bool failed{}, closed{};
  bool independent_ddr_control{}, control_verified{};
  // Explicitly released only after successful queue teardown. Failed DMA or
  // validation retains these resources until the K process is stopped.
  void* control_memory{};
  urma_target_seg_t* control_mr{};
  urma_token_id_t* control_token{};
  std::string failure;

  void Connect(const Message& request, Message& reply) {
    Check(region && !source && !jetty, "ARM required, or source already connected");
    Check(max_bytes && max_bytes <= kMaxReadChunkBytes, "explicit READ chunk bytes required");
    auto* ctx = region->context;
    auto descriptor = DecodeSourceDescriptor(request, region->bytes, ctx->eid);
    const auto expected_query = std::make_unique<TpQuery>(descriptor.tp_query);
    const auto check_query = [&](const char* stage) {
      CheckPullQueryState(stage, descriptor.tp_query, *expected_query);
    };
    check_query("decoded");
    auto& segment = descriptor.segment;
    auto& remote = descriptor.remote;
    std::fprintf(stderr, "SOURCE_DESCRIPTOR route_eid=%s mr_eid=%s jetty_eid=%s source_tp=0x%llx\n",
      SourceEidHex(request.eid).c_str(), SourceEidHex(segment.ubva.eid.raw).c_str(),
      SourceEidHex(remote.id.eid.raw).c_str(), (unsigned long long)request.tp);
    urma_token_t token{}; token.token = request.token;
    urma_import_seg_flag_t import_flags{};
    import_flags.bs.access = 2;  // remote READ only, no bit 0
    import_flags.bs.mapping = URMA_SEG_NOMAP;
    std::fprintf(stderr, "SOURCE_SEGMENT base=0x%llx bytes=%llu attr=0x%x access=0x%x "
      "non_pin=%u eid=%s uasid=0x%x token_id=0x%x\n", (unsigned long long)segment.ubva.va,
      (unsigned long long)segment.len, segment.attr.value, segment.attr.bs.access,
      segment.attr.bs.non_pin, SourceEidHex(segment.ubva.eid.raw).c_str(),
      segment.ubva.uasid, segment.token_id);
    std::fprintf(stderr, "SOURCE_IMPORT context=%p access=0x%x flags=0x%x mapping=%u "
      "requested_addr=0x0 token_value=0x%x\n", (void*)ctx, import_flags.bs.access,
      import_flags.value, import_flags.bs.mapping, token.token);
    errno = 0;
    source = urma_import_seg(ctx, &segment, &token, 0, import_flags);
    const int import_error = errno;
    std::fprintf(stderr, "SOURCE_IMPORT_RESULT mr=%p errno=%d\n", (void*)source, import_error);
    check_query("after_import_seg");
    Check(source, "import HBM segment");
    source_base = request.base;
    urma_jfc_cfg_t cq_cfg{}; cq_cfg.depth = 16384;
    cq = urma_create_jfc(ctx, &cq_cfg);
    check_query("after_create_jfc"); Check(cq, "create pull CQ");
    urma_jfr_cfg_t rq_cfg{};
    rq_cfg.depth = 1024; rq_cfg.trans_mode = URMA_TM_RM; rq_cfg.max_sge = 1;
    rq_cfg.jfc = cq; rq_cfg.token_value.token = request.token;
    rq = urma_create_jfr(ctx, &rq_cfg);
    check_query("after_create_jfr"); Check(rq, "create pull RQ");
    urma_jetty_cfg_t cfg{};
    cfg.flag.bs.share_jfr = URMA_SHARE_JFR;
    cfg.jfs_cfg.depth = 8192; cfg.jfs_cfg.trans_mode = URMA_TM_RM;
    cfg.jfs_cfg.max_sge = 1; cfg.jfs_cfg.max_rsge = 1;
    cfg.jfs_cfg.priority = 15; cfg.jfs_cfg.rnr_retry = 7; cfg.jfs_cfg.err_timeout = 16;
    cfg.jfs_cfg.jfc = cq; cfg.shared.jfr = rq; cfg.shared.jfc = cq;
    jetty = urma_create_jetty(ctx, &cfg);
    check_query("after_create_jetty"); Check(jetty, "create pull Jetty");
    auto query = reinterpret_cast<GetTp>(dlsym(RTLD_DEFAULT, "urma_get_tp_list"));
    auto import = reinterpret_cast<ImportJetty>(dlsym(RTLD_DEFAULT, "urma_import_jetty_ex"));
    Check(query && import, "missing deployed expanded CTP APIs");
    check_query("before_ctp_query");
    const auto ctp = QueryPullCtp(query, ctx, descriptor.tp_query);
    Check(ctp.ok(), "query pull CTP (see PULL_CTP_QUERY)");
    remote.policy = URMA_JETTY_GRP_POLICY_RR; remote.type = URMA_JETTY; remote.tp_type = 1;
    urma_import_jetty_flag_t flag{}; flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    remote.flag = flag.value;
    ActiveTp active{}; active.local = ctp.handle; active.peer = request.tp;
    active.tx = 1; active.rx = request.psn;
    peer = import(ctx, &remote, &token, &active); Check(peer, "import source Jetty");
    reply.tp = ctp.handle; reply.psn = 1; reply.token = request.token;
    reply.key_bytes = 28;
    std::memcpy(reply.key, &jetty->jetty_id, 24);
    auto tm = jetty->jetty_cfg.jfs_cfg.trans_mode;
    std::memcpy(reply.key + 24, &tm, 4);
    std::fprintf(stderr, "SOURCE_CONNECTED local_mr=%p exported_mr=%p local_va=0x%llx exported_va=0x%llx bytes=%llu\n",
      (void*)region->local, (void*)region->remote, (unsigned long long)region->va,
      (unsigned long long)region->remote->seg.ubva.va, (unsigned long long)region->bytes);
    // Decode only the compiled layout for diagnosis, without treating it as
    // independent evidence that the deployed target-segment ABI matches.
    std::fprintf(stderr, "READ_LOCAL_MR layout=compiled_header expected_ctx=%p mr_ctx=%p "
      "mr_va=0x%llx mr_bytes=%llu attr=0x%x\n", (void*)ctx, (void*)region->local->urma_ctx,
      (unsigned long long)region->local->seg.ubva.va,
      (unsigned long long)region->local->seg.len, region->local->seg.attr.value);
  }
  void PostReadAndWait(std::uint64_t round, std::uint64_t offset, std::uint32_t length,
                       std::uint64_t destination, urma_target_seg_t* destination_mr,
                       const char* target) {
    auto completion = std::make_unique<NativeBuffer<urma_cr_t>>();
    Sge src{}; src.addr = source_base + kGuard + offset; src.len = length; src.tseg = source;
    Sge dst{}; dst.addr = destination; dst.len = length; dst.tseg = destination_mr;
    AuditReadMrs(source, destination_mr, region->context, src.addr, dst.addr,
      length, target, offset == 0);
    urma_jfs_wr_t wr{};
    wr.opcode = URMA_OPC_READ; wr.tjetty = peer; wr.user_ctx = ++sequence;
    wr.flag.bs.complete_enable = 1; wr.flag.bs.comp_order = 1;
    wr.flag.bs.place_order = URMA_STRONG_ORDER;
    wr.rw.src = {reinterpret_cast<urma_sge_t*>(&src), 1};
    wr.rw.dst = {reinterpret_cast<urma_sge_t*>(&dst), 1};
    urma_jfs_wr_t* bad{};
    errno = 0;
    const auto post_rc = urma_post_jetty_send_wr(jetty, &wr, &bad);
    const int post_error = errno;
    if (post_rc != 0) {
      std::fprintf(stderr, "READ_POST_FAILURE target=%s round=%llu offset=%llu bytes=%u context=%llu "
        "rc=%d bad_wr=%p errno=%d\n", target, (unsigned long long)round,
        (unsigned long long)offset, length, (unsigned long long)sequence,
        int(post_rc), (void*)bad, post_error);
      Check(false, "post READ (resources retained)");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    for (;;) {
      completion->value = {};
      errno = 0;
      const int n = urma_poll_jfc(cq, 1, &completion->value);
      const int poll_error = errno;
      CheckNativeBuffer("urma_poll_jfc.cr", *completion, n, poll_error, false);
      if (n < 0 || n > 1)
        std::fprintf(stderr, "READ_POLL_FAILURE target=%s round=%llu offset=%llu bytes=%u "
          "context=%llu capacity=1 rc=%d errno=%d\n", target, (unsigned long long)round,
          (unsigned long long)offset, length, (unsigned long long)sequence, n, poll_error);
      Check(n >= 0, "poll READ CQ");
      Check(n <= 1, "unexpected READ CQ count");
      if (n) {
        const auto& cr = completion->value;
        if (cr.status != URMA_CR_SUCCESS || cr.user_ctx != sequence) {
          const char* name = cr.status == URMA_CR_SUCCESS ? "URMA_CR_SUCCESS" :
            cr.status == URMA_CR_LOC_ACCESS_ERR ? "URMA_CR_LOC_ACCESS_ERR" : "OTHER";
          std::fprintf(stderr, "CQ_FAILURE status=%u status_name=%s status_ok=%u "
            "context=%llu expected=%llu context_match=%u round=%llu offset=%llu bytes=%u "
            "src_va=0x%llx dst_va=0x%llx src_mr=%p dst_mr=%p exported_mr=%p "
            "wire_sge_bytes=%zu native_cr_bytes=%zu target=%s errno=%d\n", unsigned(cr.status), name,
            unsigned(cr.status == URMA_CR_SUCCESS), (unsigned long long)cr.user_ctx,
            (unsigned long long)sequence, unsigned(cr.user_ctx == sequence),
            (unsigned long long)round, (unsigned long long)offset, length,
            (unsigned long long)src.addr, (unsigned long long)dst.addr, (void*)src.tseg,
            (void*)dst.tseg, (void*)region->remote, sizeof(Sge), sizeof(urma_cr_t), target, poll_error);
          Check(cr.status == URMA_CR_SUCCESS, "READ CQ returned error status");
          Check(false, "READ CQ user_ctx mismatch");
        }
        break;
      }
      Check(std::chrono::steady_clock::now() < deadline, "READ CQ timeout (resources retained)");
    }
  }
  void ReadIndependentDdr(std::uint64_t round, std::uint32_t length) {
    Check(!control_memory && !control_mr && !control_token, "fresh independent DDR control required");
    // Match the allocation size and first READ length; only the destination VA
    // and its single local registration change. Never copy this data to KV DDR.
    const int allocation_rc = posix_memalign(&control_memory, 4096, region->bytes);
    std::fprintf(stderr, "INDEPENDENT_DDR_ALLOCATE bytes=%llu alignment=4096 rc=%d va=%p\n",
      (unsigned long long)region->bytes, allocation_rc, control_memory);
    Check(allocation_rc == 0 && control_memory, "allocate independent DDR control");
    Check(reinterpret_cast<std::uint64_t>(control_memory) != region->va,
      "independent DDR must use a different allocation");
    auto* ctx = region->context;
    errno = 0;
    control_token = urma_alloc_token_id(ctx);
    Check(control_token, "allocate independent DDR token");
    urma_seg_cfg_t cfg{};
    cfg.va = reinterpret_cast<std::uint64_t>(control_memory); cfg.len = region->bytes;
    cfg.token_id = control_token; cfg.token_value.token = region->token_value;
    cfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    cfg.flag.bs.access = kLocalWrite; cfg.flag.bs.non_pin = 0; cfg.flag.bs.token_id_valid = 1;
    errno = 0;
    control_mr = urma_register_seg(ctx, &cfg);
    const int register_error = errno;
    std::fprintf(stderr, "REGISTER role=independent_ddr va=0x%llx bytes=%llu access=0x%x "
      "non_pin=%u flags=0x%x mr=%p context=%p errno=%d\n", (unsigned long long)cfg.va,
      (unsigned long long)cfg.len, cfg.flag.bs.access, cfg.flag.bs.non_pin, cfg.flag.value,
      (void*)control_mr, (void*)ctx, register_error);
    Check(control_mr, "register independent DDR control");
    // Like ARM for default KV DDR, initialize only after registration.
    std::memset(control_memory, 0xa5, region->bytes);
    auto* initial = static_cast<std::uint8_t*>(control_memory) + kGuard;
    for (std::uint32_t i = 0; i < length; ++i) initial[i] = Pattern(i, round) ^ 0xff;
    std::fprintf(stderr, "INDEPENDENT_DDR_READ_BEGIN round=%llu bytes=%u src_va=0x%llx "
      "dst_va=0x%llx shared_dst_va=0x%llx dst_mr=%p context=%p wire_sge_bytes=%zu\n",
      (unsigned long long)round, length, (unsigned long long)(source_base + kGuard),
      (unsigned long long)(cfg.va + kGuard), (unsigned long long)(region->va + kGuard),
      (void*)control_mr, (void*)ctx, sizeof(Sge));
    PostReadAndWait(round, 0, length, cfg.va + kGuard, control_mr, "independent_ddr");
    const auto* memory = static_cast<const volatile std::uint8_t*>(control_memory);
    for (std::uint64_t i = 0; i < region->bytes; ++i) {
      const auto expected = i >= kGuard && i - kGuard < length ? Pattern(i - kGuard, round) : 0xa5;
      if (memory[i] != expected) {
        std::fprintf(stderr, "INDEPENDENT_DDR_MISMATCH offset=%llu expected=%u actual=%u\n",
          (unsigned long long)i, unsigned(expected), unsigned(memory[i]));
        Check(false, "independent DDR data/guard validation failed");
      }
    }
    control_verified = true;
    std::fprintf(stderr, "INDEPENDENT_DDR_CONTROL_PASS round=%llu bytes=%u cpu=ok guards=ok "
      "scope=first_chunk_only shared_ddr_verified=0\n", (unsigned long long)round, length);
  }
  void ReleaseIndependentDdr() {
    if (control_mr) {
      Check(urma_unregister_seg(control_mr) == 0, "unregister independent DDR control");
      control_mr = nullptr;
    }
    if (control_token) {
      Check(urma_free_token_id(control_token) == 0, "free independent DDR token");
      control_token = nullptr;
    }
    std::free(control_memory); control_memory = nullptr;
  }
  void Read(std::uint64_t round) {
    Check(peer && !closed && round == ready + 1 && acknowledged == ready,
      "PULL requires connected source and previous Decode ACK");
    Check(max_bytes && max_bytes <= kMaxReadChunkBytes, "invalid explicit READ chunk bytes");
    const auto bytes = region->bytes - 2 * kGuard;
    if (independent_ddr_control && !control_verified)
      ReadIndependentDdr(round, static_cast<std::uint32_t>(std::min<std::uint64_t>(max_bytes, bytes)));
    for (std::uint64_t offset = 0; offset < bytes;) {
      const auto length = static_cast<std::uint32_t>(std::min<std::uint64_t>(max_bytes, bytes - offset));
      // The destination is the original KV allocation, never a staging buffer.
      PostReadAndWait(round, offset, length, region->va + kGuard + offset, region->local, "kv_pool");
      offset += length;
    }
    // Deliberately do not CPU-read/flush/copy the payload before Decode reads.
    ready = round;
    std::fprintf(stderr, "PULL_CQ_OK round=%llu bytes=%llu independent_ddr_control=%u\n",
      (unsigned long long)round, (unsigned long long)bytes, unsigned(independent_ddr_control));
  }
  void VerifyCpu(std::uint64_t round) {
    Check(round == ready && acknowledged + 1 == round, "ACK round mismatch");
    const auto* memory = reinterpret_cast<const volatile std::uint8_t*>(region->va);
    for (std::uint64_t i = 0; i < kGuard; ++i)
      Check(memory[i] == 0xa5 && memory[region->bytes - kGuard + i] == 0xa5, "K DDR guard overwritten");
    for (std::uint64_t i = 0; i < region->bytes - 2 * kGuard; ++i) {
      if (memory[kGuard + i] != Pattern(i, round)) {
        std::fprintf(stderr, "CPU_MISMATCH round=%llu offset=%llu expected=%u actual=%u\n",
          (unsigned long long)round, (unsigned long long)i, Pattern(i, round), memory[kGuard+i]);
        Check(false, "CPU visibility check failed after Decode check");
      }
    }
    acknowledged = round;
    std::fprintf(stderr, "ROUND_PASS round=%llu decode=ok cpu=ok guards=ok independent_ddr_control=%u\n",
      (unsigned long long)round, unsigned(independent_ddr_control));
  }
  void Shutdown() {
    Check(!failed && acknowledged == ready, "cannot CLOSE failed or unacknowledged transfer");
    if (peer) { Check(urma_unimport_jetty(peer) == 0, "unimport source Jetty"); peer = nullptr; }
    if (jetty) { Check(urma_delete_jetty(jetty) == 0, "destroy pull Jetty"); jetty = nullptr; }
    if (rq) { Check(urma_delete_jfr(rq) == 0, "destroy pull RQ"); rq = nullptr; }
    if (cq) { Check(urma_delete_jfc(cq) == 0, "destroy pull CQ"); cq = nullptr; }
    if (source) { Check(urma_unimport_seg(source) == 0, "unimport HBM segment"); source = nullptr; }
    ReleaseIndependentDdr();
    if (region) region->armed = false;
    region = nullptr;
    closed = true;
  }
  Message Handle(const Message& request) {
    Message reply; reply.op = request.op;
    try {
      Check(request.magic == kMagic && request.version == kVersion, "diagnostic wire version mismatch");
      Check(!failed, failure.c_str());
      if (request.op == Arm) {
        Check(!region && !closed, "fresh K service required for each case");
        Registration* found = nullptr;
        for (auto& entry : registrations) {
          auto& r = entry.second;
          if (!request.base || r.remote->seg.ubva.va == request.base) {
            Check(!found, "ambiguous KV region; select one CPU candidate"); found = &r;
          }
        }
        Check(found && found->bytes == request.bytes, "no registered KV allocation of expected size");
        region = found; region->armed = true;
        std::memset(reinterpret_cast<void*>(region->va), 0xa5, region->bytes);
        std::fprintf(stderr, "ARMED case=%s same_allocation=1 same_mr=%u header=%s\n",
          mode.c_str(), region->local == region->remote, kHeaderAccessModel);
      } else if (request.op == Export) Connect(request, reply);
      else if (request.op == Pull) Read(request.round);
      else if (request.op == Ack) VerifyCpu(request.round);
      else if (request.op == Close) Shutdown();
      else if (request.op == Fail) Check(false, "client reported validation failure; resources retained");
      else Check(request.op == Info, "unknown diagnostic operation");
      if (region) {
        reply.base = region->remote->seg.ubva.va; reply.bytes = region->bytes;
        std::memcpy(reply.eid, region->context->eid.raw, 16);
      }
      reply.round = ready; reply.ack = acknowledged;
      if (closed) reply.status = 1;
    } catch (const std::exception& e) {
      failed = true; failure = e.what(); reply.status = -1;
      std::snprintf(reply.error, sizeof(reply.error), "%s", e.what());
      std::fprintf(stderr, "PROBE_FAIL op=%u reason=%s errno=%d\n", request.op, e.what(), errno);
    }
    return reply;
  }
};
bool Io(int fd, void* buffer, std::size_t bytes, bool write) {
  auto* p = static_cast<char*>(buffer);
  while (bytes) {
    const auto n = write ? send(fd, p, bytes, MSG_NOSIGNAL) : recv(fd, p, bytes, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n; bytes -= n;
  }
  return true;
}
int Listen(const char* address, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0); Check(fd >= 0, "socket");
  int one = 1; setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons(port);
  Check(inet_pton(AF_INET, address, &addr.sin_addr) == 1, "listen address must be IPv4");
  Check(bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 && listen(fd, 16) == 0,
    "bind/listen");
  return fd;
}
}  // namespace probe

int main(int argc, char** argv) {
  using namespace probe;
  if ((argc != 11 && argc != 12) || std::string(argv[9]) != "--read-chunk-bytes" ||
      (argc == 12 && std::string(argv[11]) != "--independent-ddr-control")) {
    std::fprintf(stderr, "Usage: %s CASE UB_DEVICE EID_INDEX ALLOCATION_BYTES LISTEN_IPV4 KVAF_PORT PROBE_PORT PAGE_MODE --read-chunk-bytes N [--independent-ddr-control]\n"
      "CASE: single-current single-pinned single-plus-bit0 single-local-write-pinned dual dual-compatible dual-reverse dual-both-pinned dual-separate-context\n"
      "PAGE_MODE: default or thp\n"
      "READ CHUNK: required explicit candidate, software range 1..65536; hardware limits are NOT queried\n"
      "INDEPENDENT DDR CONTROL: optional first-chunk comparison, requires default page mode\n",
      argv[0]); return 2;
  }
  try {
    const auto read_chunk_bytes = ParseReadChunkBytes(argv[10]);
    mode = argv[1];
    const std::vector<std::string> cases{"single-current", "single-pinned", "single-plus-bit0",
      "single-local-write-pinned", "dual", "dual-compatible", "dual-reverse", "dual-both-pinned", "dual-separate-context"};
    Check(std::find(cases.begin(), cases.end(), mode) != cases.end(), "unknown case");
    allocation_bytes = std::stoull(argv[4]);
    Check(allocation_bytes > 2*kGuard && allocation_bytes <= (UINT64_C(1)<<30), "allocation must be >8192 and <=1GiB");
    kvaf::CpuRuntimeOptions options;
    options.ub_device = argv[2]; options.has_eid_index = true; options.eid_index = std::stoul(argv[3]);
    Check(std::string(argv[8]) == "default" || std::string(argv[8]) == "thp", "invalid page mode");
    Check(argc != 12 || std::string(argv[8]) == "default",
      "independent DDR control requires default page mode");
    options.memory_page_mode = std::string(argv[8]) == "thp" ? kvaf::CpuMemoryPageMode::kTransparentHugePage : kvaf::CpuMemoryPageMode::kDefault;
    kvaf::CpuRuntime runtime;
    Check(runtime.Initialize(options).ok(), "initialize real KVAF CPU runtime");
    int service = Listen(argv[5], std::stoi(argv[6]));
    int diagnostic = Listen(argv[5], std::stoi(argv[7]));
    State state;
    state.max_bytes = read_chunk_bytes;
    state.independent_ddr_control = argc == 12;
    std::fprintf(stderr, "PROBE_READY case=%s bytes=%llu header=%s native_sge_bytes=%zu wire_sge_bytes=%zu "
      "native_wr_bytes=%zu native_cr_bytes=%zu read_chunk_bytes=%u "
      "capability_source=explicit_candidate hardware_limits_queried=0 src_sge=1 dst_sge=1 independent_ddr_control=%u\n",
      mode.c_str(), (unsigned long long)allocation_bytes, kHeaderAccessModel, sizeof(urma_sge_t), sizeof(Sge),
      sizeof(urma_jfs_wr_t), sizeof(urma_cr_t), read_chunk_bytes, unsigned(state.independent_ddr_control));
    for (;;) {
      fd_set fds; FD_ZERO(&fds); FD_SET(service, &fds); FD_SET(diagnostic, &fds);
      const int rc = select(std::max(service, diagnostic)+1, &fds, nullptr, nullptr, nullptr);
      if (rc < 0 && errno == EINTR) continue;
      Check(rc > 0, "select");
      for (int listener : {service, diagnostic}) {
        if (!FD_ISSET(listener, &fds)) continue;
        int fd = accept(listener, nullptr, nullptr); Check(fd >= 0, "accept");
        timeval timeout{30, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        if (listener == service) {
          // Failed/armed resources must never enter the legacy service's
          // unconditional allocation teardown on a client error.
          if (!state.failed) Check(runtime.ServeConnection(fd).ok(), "KVAF control request");
        } else {
          Message request;
          if (Io(fd, &request, sizeof(request), false)) {
            auto reply = state.Handle(request);
            (void)Io(fd, &reply, sizeof(reply), true);
          }
        }
        close(fd);
      }
    }
  } catch (const std::exception& e) { Fatal(e.what()); }
}
