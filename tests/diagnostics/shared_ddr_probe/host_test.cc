// Runs the diagnostic's actual registration and READ orchestration against a
// fake provider. This tests the harness, NOT driver support or DMA coherence.
#define main probe_service_main
#include "k_service.cc"
#undef main
#include <iostream>

namespace {
int live_segments{}, live_tokens{}, live_contexts{};
bool reject_local{}, reject_post{}, error_cq{};
bool strict_24_12{};
std::uint64_t completed{};
std::vector<unsigned> access_order;
std::map<urma_target_seg_t*, std::pair<std::uint64_t, std::uint64_t>> ranges;
void Require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }

void CopyHex(const char* hex, std::uint8_t* bytes, std::size_t count) {
  Require(std::strlen(hex) == 2 * count, "bad fixture length");
  const auto digit = [](char c) -> unsigned {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    throw std::runtime_error("bad fixture hex");
  };
  for (std::size_t i = 0; i < count; ++i)
    bytes[i] = static_cast<std::uint8_t>((digit(hex[2*i]) << 4) | digit(hex[2*i+1]));
}

void ExpectSourceRejected(const probe::Message& request, const urma_eid_t& cpu_eid,
                          const char* reason) {
  bool rejected = false;
  try { (void)probe::DecodeSourceDescriptor(request, 2695168, cpu_eid); }
  catch (const std::runtime_error& error) {
    rejected = true;
    Require(std::strstr(error.what(), reason) != nullptr, "unexpected source rejection reason");
  }
  Require(rejected, "invalid source descriptor accepted");
}

void TestSourceDescriptor() {
  using namespace probe;
  // Actual exported keys from physical NPU 5. Its configured CLOS EID differs
  // from the matching MR/Jetty resource EID; this must not be rejected.
  Message request;
  request.op = Export;
  request.base = UINT64_C(0x12004c600000); request.bytes = 2695168;
  request.tp = 1; request.psn = 1; request.token = 0x51a7d001;
  request.segment_bytes = 48; request.key_bytes = 32;
  CopyHex("00000000003f030000100000df00ab00", request.eid, 16);
  CopyHex("000000000001030000100000df00a200000000000000604c00120000000000000020290000000000c129000000ff1000",
    request.segment, 48);
  CopyHex("000000000001030000100000df00a20000000000b02500000100000000000000", request.key, 32);
  urma_eid_t cpu_eid{};
  CopyHex("00000000007f020000100000df0a0b00", cpu_eid.raw, 16);
  const auto decoded = DecodeSourceDescriptor(request, request.bytes, cpu_eid);
  Require(std::memcmp(decoded.tp_query.peer, request.eid, 16) == 0,
    "CTP query lost configured route EID");
  Require(std::memcmp(decoded.tp_query.local, cpu_eid.raw, 16) == 0 &&
    decoded.tp_query.flag == 1 && decoded.tp_query.mode == URMA_TM_RM, "wrong CTP query");
  Require(std::memcmp(decoded.tp_query.peer, decoded.remote.id.eid.raw, 16) != 0,
    "fixture must distinguish route from resource identity");
  Require(std::memcmp(&decoded.segment, request.segment, 48) == 0 &&
    std::memcmp(&decoded.remote.id, request.key, sizeof(decoded.remote.id)) == 0,
    "exported resource identity was rewritten");
  Require(decoded.segment.ubva.va == request.base && decoded.segment.len == request.bytes &&
    decoded.remote.id.id == 0x25b0 && decoded.remote.mode == URMA_TM_RM, "wrong key decoding");

  auto same = request;
  std::memcpy(same.eid, request.key, 16);
  (void)DecodeSourceDescriptor(same, same.bytes, cpu_eid);

  auto bad = request;
  std::memcpy(bad.segment, bad.eid, 16);  // Old check accepted this MR != Jetty.
  ExpectSourceRejected(bad, cpu_eid, "HBM segment EID differs from exported Jetty EID");
  for (auto size : {0U, 24U, 31U, 33U, 64U}) {
    bad = request; bad.key_bytes = size;
    ExpectSourceRejected(bad, cpu_eid, "invalid HBM descriptor/Jetty ABI");
  }
  for (auto size : {0U, 47U, 49U, 64U}) {
    bad = request; bad.segment_bytes = size;
    ExpectSourceRejected(bad, cpu_eid, "invalid HBM descriptor/Jetty ABI");
  }
  bad = request;
  const std::uint32_t invalid_mode = UINT32_MAX;
  std::memcpy(bad.key + sizeof(urma_jetty_id_t), &invalid_mode, sizeof(invalid_mode));
  ExpectSourceRejected(bad, cpu_eid, "source Jetty must use RM");
  bad = request; ++bad.base;
  ExpectSourceRejected(bad, cpu_eid, "HBM range does not cover");
  bad = request; bad.base = UINT64_MAX - bad.bytes + 1;
  ExpectSourceRejected(bad, cpu_eid, "HBM range does not cover");
  bad = request; --bad.bytes;
  ExpectSourceRejected(bad, cpu_eid, "HBM range does not cover");
  bad = request; std::memset(bad.segment, 0, 16); std::memset(bad.key, 0, 16);
  ExpectSourceRejected(bad, cpu_eid, "source Jetty EID must be nonzero");
  bad = request; std::memset(bad.eid, 0, 16);
  ExpectSourceRejected(bad, cpu_eid, "source route EID must be nonzero");
  bad = request; bad.tp = 0;
  ExpectSourceRejected(bad, cpu_eid, "invalid HBM descriptor/Jetty ABI");
  bad = request; bad.psn = 0;
  ExpectSourceRejected(bad, cpu_eid, "invalid HBM descriptor/Jetty ABI");
  bad = request; bad.token = 0;
  ExpectSourceRejected(bad, cpu_eid, "invalid HBM descriptor/Jetty ABI");
  std::cerr << "SOURCE_DESCRIPTOR_TEST_PASS route_eid_preserved=1 resource_identity_checked=1\n";
}

struct FakePullQuery {
  urma_context_t* context{};
  probe::TpQuery config{};
  int rc{}, error{};
  std::uint32_t count{};
  std::uint64_t handle{};
  unsigned calls{};
} fake_pull_query;

int QueryPullTpFake(urma_context_t* context, probe::TpQuery* config,
                    std::uint32_t* count, probe::TpInfo* info) {
  auto& fake = fake_pull_query;
  ++fake.calls;
  Require(fake.calls == 1, "pull CTP query retried or fell back to another route");
  Require(errno == 0, "pull CTP query inherited stale errno");
  Require(context == fake.context && config && count && info, "wrong pull CTP query arguments");
  Require(config->flag == fake.config.flag && config->mode == fake.config.mode &&
    std::memcmp(config->local, fake.config.local, 16) == 0 &&
    std::memcmp(config->peer, fake.config.peer, 16) == 0,
    "pull CTP query rewrote its flags or endpoint EIDs");
  Require(*count == 1 && info->handle == 0, "pull CTP query outputs not initialized");
  *count = fake.count;
  info->handle = fake.handle;
  errno = fake.error;
  return fake.rc;
}

void TestPullCtpQuery() {
  urma_context_t context{};
  probe::TpQuery config{};
  config.flag = 1; config.mode = URMA_TM_RM;
  CopyHex("00000000007f020000100000df0a0b00", config.local, 16);
  CopyHex("00000000003f030000100000df00ab00", config.peer, 16);
  struct Scenario {
    int rc;
    std::uint32_t count;
    std::uint64_t handle;
    int error;
    bool ok;
    const char* outcome;
  };
  const Scenario scenarios[] = {
    {0, 1, 0x1234, 0, true, "ok"},
    {0, 1, 0x1234, EAGAIN, true, "ok"},
    {EINVAL, 1, 0x1234, EIO, false, "api_error"},
    {-1, 1, 0x1234, 0, false, "api_error"},
    {0, 0, 0, 0, false, "empty_list"},
    {0, 0, 0x1234, 0, false, "empty_list"},
    {0, 1, 0, 0, false, "zero_handle"},
  };
  for (const auto& scenario : scenarios) {
    fake_pull_query = {};
    fake_pull_query.context = &context;
    fake_pull_query.config = config;
    fake_pull_query.rc = scenario.rc;
    fake_pull_query.count = scenario.count;
    fake_pull_query.handle = scenario.handle;
    fake_pull_query.error = scenario.error;
    errno = EINVAL;  // Must be cleared even when the callback leaves errno zero.
    const auto result = probe::QueryPullCtp(QueryPullTpFake, &context, config);
    const int restored_errno = errno;
    Require(fake_pull_query.calls == 1, "pull CTP query did not issue exactly one call");
    Require(result.rc == scenario.rc && result.count == scenario.count &&
      result.handle == scenario.handle && result.error == scenario.error,
      "pull CTP query lost raw callback results");
    Require(restored_errno == scenario.error, "pull CTP query logging overwrote captured errno");
    Require(result.ok() == scenario.ok && std::strcmp(result.outcome(), scenario.outcome) == 0,
      "pull CTP query misclassified its result");
  }
  std::cerr << "PULL_CTP_QUERY_TEST_PASS single_call=1 endpoints_preserved=1 errno_captured=1\n";
}
}
extern "C" {
urma_target_seg_t* urma_register_seg(urma_context_t* ctx, urma_seg_cfg_t* cfg) {
  access_order.push_back(cfg->flag.bs.access);
  // Optional model of br_UMDK_24.12.0_LTS's core permission validation.
  // The plumbing profile deliberately makes no provider capability claims.
  const bool remote_write_or_atomic = cfg->flag.bs.access &
    (URMA_ACCESS_REMOTE_WRITE | URMA_ACCESS_REMOTE_ATOMIC);
  if ((cfg->flag.bs.access == kLocalWrite && reject_local) ||
      (strict_24_12 && remote_write_or_atomic && !(cfg->flag.bs.access & kLocalWrite))) {
    errno = EINVAL; return nullptr;
  }
  auto* seg = new urma_target_seg_t{};
  seg->seg.ubva.va = cfg->va; seg->seg.len = cfg->len; seg->urma_ctx = ctx;
  ranges[seg] = {cfg->va, cfg->len}; ++live_segments;
  return seg;
}
urma_status_t urma_unregister_seg(urma_target_seg_t* seg) {
  Require(ranges.erase(seg) == 1, "unregister unknown/double-freed MR");
  delete seg; --live_segments; return URMA_SUCCESS;
}
urma_token_id_t* urma_alloc_token_id(urma_context_t*) {
  ++live_tokens; return new urma_token_id_t{};
}
urma_status_t urma_free_token_id(urma_token_id_t* token) {
  delete token; --live_tokens; return URMA_SUCCESS;
}
urma_context_t* urma_create_context(urma_device_t* dev, std::uint32_t index) {
  auto* ctx = new urma_context_t{}; ctx->dev = dev; ctx->eid_index = index;
  ++live_contexts; return ctx;
}
urma_status_t urma_delete_context(urma_context_t* ctx) {
  delete ctx; --live_contexts; return URMA_SUCCESS;
}
urma_status_t urma_post_jetty_send_wr(urma_jetty_t*, urma_jfs_wr_t* wr, urma_jfs_wr_t**) {
  if (reject_post) return URMA_FAIL;
  Require(wr->opcode == URMA_OPC_READ && wr->rw.src.num_sge == 1 && wr->rw.dst.num_sge == 1,
    "must issue a READ with one source and destination SGE");
  auto& src = *reinterpret_cast<probe::Sge*>(wr->rw.src.sge);
  auto& dst = *reinterpret_cast<probe::Sge*>(wr->rw.dst.sge);
  const auto range = ranges.at(dst.tseg);
  Require(src.len == dst.len && dst.addr >= range.first &&
    dst.addr + dst.len <= range.first + range.second, "READ destination escapes original allocation");
  std::memcpy(reinterpret_cast<void*>(dst.addr), reinterpret_cast<void*>(src.addr), src.len);
  completed = wr->user_ctx; return URMA_SUCCESS;
}
int urma_poll_jfc(urma_jfc_t*, int, urma_cr_t* cr) {
  if (!completed) return 0;
  cr->user_ctx = completed; completed = 0;
  cr->status = error_cq ? URMA_CR_LOC_ACCESS_ERR : URMA_CR_SUCCESS;
  return 1;
}
}

int main() {
  using namespace probe;
  try {
    std::cerr << "HARNESS_UNIT_TEST fake_provider=1 profile=plumbing_only hardware_result=UNKNOWN\n";
    TestSourceDescriptor();
    TestPullCtpQuery();
    allocation_bytes = 3 * kGuard;
    std::vector<std::uint8_t> target(allocation_bytes), source(allocation_bytes);
    urma_context_t ctx{};
    urma_seg_cfg_t cfg{}; cfg.va = reinterpret_cast<std::uint64_t>(target.data());
    cfg.len = allocation_bytes; cfg.flag.bs.access = 14; cfg.flag.bs.non_pin = 1;
    for (const auto* name : {"single-current", "single-pinned", "dual", "dual-reverse",
                             "dual-both-pinned", "dual-separate-context", "single-plus-bit0",
                             "single-local-write-pinned", "dual-compatible"}) {
      mode = name; access_order.clear();
      auto* remote = Register(&ctx, &cfg); Require(remote, "register case");
      auto& r = registrations.at(remote);
      const bool dual = mode.rfind("dual", 0) == 0;
      Require((r.local != r.remote) == dual, "wrong number of MR views");
      Require(ranges.at(r.local) == ranges.at(r.remote), "MRs do not alias exactly the same allocation");
      if (mode == "dual-reverse") Require(access_order == std::vector<unsigned>{1,14}, "reverse registration order");
      if (mode == "dual") Require(access_order == std::vector<unsigned>{14,1}, "normal registration order");
      if (mode == "dual-separate-context") Require(r.context != &ctx, "separate context not used");
      State state;
      state.region = &r; state.peer = reinterpret_cast<urma_target_jetty_t*>(1);
      state.source_base = reinterpret_cast<std::uint64_t>(source.data());
      state.max_bytes = 997;  // Force unaligned chunk boundaries and a tail.
      std::fill(target.begin(), target.end(), 0xa5);
      for (std::uint64_t round = 1; round <= 3; ++round) {
        for (std::uint64_t i = 0; i < kGuard; ++i) source[kGuard+i] = Pattern(i, round);
        state.Read(round);
        Require(state.ready == round && state.acknowledged == round-1, "published before completion or auto-ACK");
        Require(std::equal(source.begin()+kGuard, source.end()-kGuard, target.begin()+kGuard),
          "READ did not write the original KV bytes");
        state.VerifyCpu(round);
      }
      reject_post = true;
      bool rejected = false;
      try { state.Read(4); } catch (const std::runtime_error&) { rejected = true; }
      reject_post = false;
      Require(rejected, "post error ignored");
      Require(state.ready == 3, "failed post published a round");
      error_cq = true;
      rejected = false;
      try { state.Read(4); } catch (const std::runtime_error&) { rejected = true; }
      error_cq = false;
      Require(rejected, "CQ error ignored");
      Require(state.ready == 3, "failed CQ published a round");
      Unregister(remote);
      Require(!live_segments && !live_tokens && !live_contexts && registrations.empty(), "case leaks resources");
    }
    mode = "dual"; reject_local = true;
    Require(!Register(&ctx, &cfg), "failed local registration accepted");
    reject_local = false;
    Require(!live_segments && !live_tokens && registrations.empty(), "rollback leaked remote MR/token");
    strict_24_12 = true;
    std::cerr << "HARNESS_UNIT_TEST profile=24.12_core_access_check hardware_result=UNKNOWN\n";
    for (const auto* name : {"single-current", "single-pinned", "dual", "dual-reverse"}) {
      mode = name;
      Require(!Register(&ctx, &cfg), "24.12 core must reject remote write/atomic without LOCAL_WRITE");
      Require(!live_segments && !live_tokens && registrations.empty(), "rejected case leaked resources");
    }
    for (const auto* name : {"single-plus-bit0", "single-local-write-pinned", "dual-compatible"}) {
      mode = name;
      auto* remote = Register(&ctx, &cfg);
      Require(remote, "24.12 core must accept combined access bits before provider validation");
      Unregister(remote);
      Require(!live_segments && !live_tokens && registrations.empty(), "compatible case leaked resources");
    }
    Require(cfg.flag.bs.access == 14 && cfg.flag.bs.non_pin == 1, "modified caller registration config");
    std::cout << "HOST_HARNESS_PASS (fake provider; hardware result UNKNOWN)\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
