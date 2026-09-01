#include "cpu_urma_transport.h"

#include <dlfcn.h>
#include <errno.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>

#include <urma_api.h>

#include "peer_link.h"

namespace kvaf {
namespace {

constexpr std::uint32_t kJfcDepth = 1024;
constexpr std::uint32_t kJettyDepth = 1024;
constexpr std::uint32_t kPageBytes = 4096;
constexpr std::uint32_t kKeyBytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
// AIV/Direct-HCCP and liburma expose the same routing EID prefix but may
// encode the provider/function suffix differently (for example df006b00 vs
// df080b00).  Keep the AIV-selected EID as the source of truth and use this
// prefix only to locate its liburma representation.
constexpr std::size_t kEidRoutingPrefixBytes = 13;
static_assert(kKeyBytes == 28, "unexpected URMA jetty key ABI");

// Some deployed UMDK headers omit the CTP-aware import declaration while the
// symbol is present in liburma. Keep this ABI-local adapter identical to the
// one already used by CpuRuntime.
struct ImportFlag { std::uint64_t value; std::uint64_t stag; std::uint64_t dtag; };
struct RemoteJettyEx {
  urma_jetty_id_t jetty_id;
  std::uint32_t trans_mode;
  std::uint32_t policy;
  std::uint32_t type;
  std::uint32_t reserved_before_flag;
  ImportFlag flag;
  std::uint32_t tp_type;
  std::uint32_t reserved_after_tp_type;
};
struct TpQuery { std::uint32_t flag; std::uint32_t trans_mode;
  std::uint8_t local_eid[16]; std::uint8_t peer_eid[16]; };
struct TpInfo { std::uint64_t tp_handle; };
struct ActiveTp { std::uint64_t tp_handle; std::uint64_t peer_tp_handle;
  std::uint64_t tag; std::uint32_t tx_psn; std::uint32_t rx_psn; std::uint64_t reserved; };
using GetTpList = int (*)(urma_context_t*, TpQuery*, std::uint32_t*, TpInfo*);
using ImportJettyEx = urma_target_jetty_t* (*)(urma_context_t*, RemoteJettyEx*, urma_token_t*, ActiveTp*);

Status Error(const char* operation) {
  return Status::Internal(std::string(operation) + " failed (errno=" + std::to_string(errno) + ")");
}

}  // namespace

struct CpuUrmaTransport::Impl {
  ~Impl() { (void)Shutdown(); }

  Status Initialize(const CpuUrmaTransportOptions& options) {
    if (options.peer_address.empty() || options.region_bytes == 0 ||
        options.region_bytes > std::numeric_limits<std::size_t>::max()) {
      return Status::InvalidArgument("CPU URMA transport requires peer_address and non-zero region_bytes");
    }
    options_ = options;
    // URMA is process-wide.  The ACL/HBM runtime may have initialized it
    // before this transport is constructed, in which case liburma returns
    // EEXIST and the existing instance must be reused.  Only the caller that
    // performed the successful initialization may uninitialize the library.
    urma_init_attr_t init{};
    const int init_rc = urma_init(&init);
    if (init_rc != URMA_SUCCESS && init_rc != EEXIST && init_rc != -EEXIST) {
      return Status::Internal("urma_init failed (rc=" + std::to_string(init_rc) +
                             ", errno=" + std::to_string(errno) + ")");
    }
    urma_initialized_ = true;
    urma_owned_ = (init_rc == URMA_SUCCESS);

    int count = 0;
    urma_device_t** devices = urma_get_device_list(&count);
    if (devices == nullptr) return Status::NotFound("no URMA devices");
    bool matched_eid = false;
    std::uint32_t ub_device_count = 0;
    std::uint32_t exposed_eid_count = 0;
    std::uint32_t matching_eid_count = 0;
    urma_device_t* compatible_device = nullptr;
    std::uint32_t compatible_eid_index = 0;
    const bool verbose_eid = std::getenv("KVAF_CPU_URMA_VERBOSE_EID") != nullptr;
    for (int i = 0; i < count && device_ == nullptr; ++i) {
      urma_device_t* candidate = devices[i];
      if (candidate == nullptr || candidate->type != URMA_TRANSPORT_UB ||
          (!options.ub_device.empty() && options.ub_device != candidate->name)) continue;
      ++ub_device_count;
      std::uint32_t eid_count = 0;
      urma_eid_info_t* eids = urma_get_eid_list(candidate, &eid_count);
      if (eids == nullptr) {
        std::fprintf(stderr,
                     "[KVAF_CPU_URMA] URMA_EID_ENUM_FAILED device=%s type=%u errno=%d\n",
                     candidate->name, static_cast<unsigned>(candidate->type), errno);
        continue;
      }
      exposed_eid_count += eid_count;
      if (options.has_preferred_eid && verbose_eid) {
        std::fprintf(stderr,
                     "[KVAF_CPU_URMA] URMA_DEVICE device=%s type=%u eid_count=%u preferred_eid=",
                     candidate->name, static_cast<unsigned>(candidate->type), eid_count);
        for (const std::uint8_t byte : options.preferred_eid) {
          std::fprintf(stderr, "%02x", byte);
        }
        std::fprintf(stderr, "\n");
      }
      for (std::uint32_t j = 0; j < eid_count; ++j) {
        if (options.has_preferred_eid && verbose_eid) {
          std::fprintf(stderr, "[KVAF_CPU_URMA] URMA_EID device=%s index=%u eid=",
                       candidate->name, eids[j].eid_index);
          for (std::size_t byte = 0; byte < 16; ++byte) {
            std::fprintf(stderr, "%02x", eids[j].eid.raw[byte]);
          }
          std::fprintf(stderr, " match=%s\n",
                       std::equal(options.preferred_eid.begin(), options.preferred_eid.end(),
                                  eids[j].eid.raw)
                           ? "yes" : "no");
        }
        const bool preferred_match =
            !options.has_preferred_eid ||
            std::equal(options.preferred_eid.begin(), options.preferred_eid.end(), eids[j].eid.raw);
        if (preferred_match) ++matching_eid_count;
        const bool usable =
            (!options.has_eid_index || eids[j].eid_index == options.eid_index) &&
            !std::all_of(eids[j].eid.raw, eids[j].eid.raw + 16,
                         [](std::uint8_t b) { return b == 0; });
        const bool routing_match =
            options.has_preferred_eid &&
            std::equal(options.preferred_eid.begin(),
                       options.preferred_eid.begin() + kEidRoutingPrefixBytes,
                       eids[j].eid.raw);
        if (usable && preferred_match) {
          device_ = candidate;
          eid_index_ = eids[j].eid_index;
          matched_eid = true;
          break;
        }
        // Exact byte equality is not available on some systems because the
        // AIV and URMA providers encode the EID suffix differently.  Retain
        // the first same-routing-prefix candidate as a fallback, but keep
        // scanning so an exact AIV EID match always wins.
        if (usable && routing_match && compatible_device == nullptr) {
          compatible_device = candidate;
          compatible_eid_index = eids[j].eid_index;
        }
      }
      urma_free_eid_list(eids);
    }
    urma_free_device_list(devices);
    if (device_ == nullptr && compatible_device != nullptr) {
      device_ = compatible_device;
      eid_index_ = compatible_eid_index;
      matched_eid = true;
      std::fprintf(stderr,
                   "[KVAF_CPU_URMA] EID_SCAN mapped_aiv_eid_to_urma_prefix eid_index=%u\n",
                   eid_index_);
    }
    if (device_ == nullptr) {
      if (options.has_preferred_eid && !matched_eid) {
        std::fprintf(stderr,
                     "[KVAF_CPU_URMA] EID_SCAN preferred_mismatch ub_devices=%u exposed_eids=%u exact_matches=%u\n",
                     ub_device_count, exposed_eid_count, matching_eid_count);
        return Status::NotFound("no usable UB EID matched preferred EID");
      }
      return Status::NotFound("no usable UB EID");
    }
    context_ = urma_create_context(device_, eid_index_);
    if (context_ == nullptr) return Error("urma_create_context");
    urma_jfc_cfg_t jfc_cfg{};
    jfc_cfg.depth = kJfcDepth;
    jfc_ = urma_create_jfc(context_, &jfc_cfg);
    if (jfc_ == nullptr) return Error("urma_create_jfc");
    get_tp_list_ = reinterpret_cast<GetTpList>(dlsym(RTLD_DEFAULT, "urma_get_tp_list"));
    import_jetty_ex_ = reinterpret_cast<ImportJettyEx>(dlsym(RTLD_DEFAULT, "urma_import_jetty_ex"));
    if (get_tp_list_ == nullptr || import_jetty_ex_ == nullptr) {
      return Status::NotImplemented("liburma lacks CTP-aware import APIs");
    }

    if (posix_memalign(&buffer_, kPageBytes, static_cast<std::size_t>(options.region_bytes)) != 0 ||
        buffer_ == nullptr) return Error("posix_memalign");
    std::memset(buffer_, 0, static_cast<std::size_t>(options.region_bytes));
    token_value_ = 1;
    token_id_ = urma_alloc_token_id(context_);
    if (token_id_ == nullptr) return Error("urma_alloc_token_id");
    urma_seg_cfg_t seg_cfg{};
    seg_cfg.va = reinterpret_cast<std::uint64_t>(buffer_);
    seg_cfg.len = options.region_bytes;
    seg_cfg.token_id = token_id_;
    seg_cfg.token_value.token = token_value_;
    seg_cfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    // UMDK releases use different names for local-only registration. Keep the
    // probe source-compatible with both the deployed CPU service headers and
    // older public UMDK headers.
#if defined(URMA_ACCESS_LOCAL_WRITE)
    seg_cfg.flag.bs.access = URMA_ACCESS_LOCAL_WRITE;
#elif defined(URMA_ACCESS_LOCAL_ONLY)
    seg_cfg.flag.bs.access = URMA_ACCESS_LOCAL_ONLY;
#else
    seg_cfg.flag.bs.access = 1U;
#endif
    seg_cfg.flag.bs.non_pin = 1;
    seg_cfg.flag.bs.token_id_valid = 1;
    local_seg_ = urma_register_seg(context_, &seg_cfg);
    if (local_seg_ == nullptr) return Error("urma_register_seg");

    link_ = std::make_unique<peer::PeerLink>(options.peer_address);
    peer::PeerRegion region{};
    Status status = Status::OK();
    if (options.has_external_region) {
      region = options.external_region;
      (void)link_->RememberRegion(region);
    } else {
      status = options.remote_hbm
                   ? link_->AllocateHbmRegion(options.region_bytes, options.region_bytes,
                                              options.remote_device_id,
                                              options.remote_eid_index, &region)
                   : link_->AllocateRegion(options.region_bytes, options.region_bytes,
                                           options.peer_candidate_id, &region);
      if (!status.ok()) return status;
      owns_remote_region_ = true;
    }
    remote_region_ = region;
    remote_seg_ = {};
    if (region.segment_key_bytes < sizeof(urma_seg_t)) {
      return Status::Internal("peer segment descriptor is shorter than urma_seg_t");
    }
    std::memcpy(&remote_seg_, region.segment_key.data(), sizeof(urma_seg_t));
    urma_token_t remote_token{};
    remote_token.token = region.token_value;
    urma_import_seg_flag_t import_flag{};
    import_flag.bs.cacheable = URMA_NON_CACHEABLE;
#if defined(URMA_ACCESS_REMOTE_READ) && defined(URMA_ACCESS_REMOTE_WRITE)
    import_flag.bs.access = URMA_ACCESS_REMOTE_READ | URMA_ACCESS_REMOTE_WRITE;
#elif defined(URMA_ACCESS_READ) && defined(URMA_ACCESS_WRITE)
    import_flag.bs.access = URMA_ACCESS_READ | URMA_ACCESS_WRITE;
#else
    import_flag.bs.access = (1U << 1U) | (1U << 2U);
#endif
    import_flag.bs.mapping = URMA_SEG_NOMAP;
    remote_seg_handle_ = urma_import_seg(context_, &remote_seg_, &remote_token, 0, import_flag);
    if (remote_seg_handle_ == nullptr) return Error("urma_import_seg");

    urma_jfr_cfg_t jfr_cfg{};
    jfr_cfg.depth = kJettyDepth;
    jfr_cfg.trans_mode = URMA_TM_RM;
    jfr_cfg.max_sge = 1;
    jfr_cfg.jfc = jfc_;
    jfr_cfg.token_value.token = region.token_value;
    jfr_ = urma_create_jfr(context_, &jfr_cfg);
    if (jfr_ == nullptr) return Error("urma_create_jfr");
    urma_jetty_cfg_t jetty_cfg{};
    jetty_cfg.flag.bs.share_jfr = URMA_SHARE_JFR;
    jetty_cfg.jfs_cfg.depth = kJettyDepth;
    jetty_cfg.jfs_cfg.trans_mode = URMA_TM_RM;
    jetty_cfg.jfs_cfg.max_sge = 1;
    jetty_cfg.jfs_cfg.max_rsge = 1;
    jetty_cfg.jfs_cfg.jfc = jfc_;
    jetty_cfg.shared.jfr = jfr_;
    jetty_cfg.shared.jfc = jfc_;
    local_jetty_ = urma_create_jetty(context_, &jetty_cfg);
    if (local_jetty_ == nullptr) return Error("urma_create_jetty");

    std::uint64_t local_tp = 0;
    TpQuery query{};
    query.flag = 1;
    query.trans_mode = static_cast<std::uint32_t>(URMA_TM_RM);
    std::memcpy(query.local_eid, context_->eid.raw, 16);
    std::memcpy(query.peer_eid, region.eid.data(), 16);
    TpInfo info{};
    std::uint32_t tp_count = 1;
    if (get_tp_list_(context_, &query, &tp_count, &info) != 0 || tp_count == 0 || info.tp_handle == 0) {
      return Error("urma_get_tp_list");
    }
    local_tp = info.tp_handle;
    std::array<std::uint8_t, peer::kQpKeyBytes> local_key{};
    std::memcpy(local_key.data(), &local_jetty_->jetty_id, sizeof(local_jetty_->jetty_id));
    const urma_transport_mode_t mode = local_jetty_->jetty_cfg.jfs_cfg.trans_mode;
    std::memcpy(local_key.data() + sizeof(local_jetty_->jetty_id), &mode, sizeof(mode));
    std::array<std::uint8_t, peer::kQpKeyBytes> peer_key{};
    std::uint32_t peer_key_bytes = 0;
    std::uint64_t peer_tp = 0;
    std::uint32_t peer_psn = 0;
    status = link_->CreateChannel(region.base, local_key.data(), kKeyBytes, context_->eid.raw,
                                  region.token_value, local_tp, 1, &peer_key, &peer_key_bytes,
                                  &peer_tp, &peer_psn);
    if (!status.ok()) return status;
    RemoteJettyEx remote{};
    std::memcpy(&remote.jetty_id, peer_key.data(), sizeof(remote.jetty_id));
    std::memcpy(&remote.trans_mode, peer_key.data() + sizeof(remote.jetty_id), sizeof(remote.trans_mode));
    remote.policy = static_cast<std::uint32_t>(URMA_JETTY_GRP_POLICY_RR);
    remote.type = static_cast<std::uint32_t>(URMA_JETTY);
    urma_import_jetty_flag_t jetty_flag{};
    jetty_flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    remote.flag.value = jetty_flag.value;
    // URMA_CTP is enum value 1 in the expanded import ABI.  Some installed
    // headers do not expose the enum symbol, while CpuRuntime already uses
    // this value for the same liburma entry point.
    remote.tp_type = 1U;
    ActiveTp active{};
    active.tp_handle = local_tp;
    active.peer_tp_handle = peer_tp;
    active.tx_psn = 1;
    active.rx_psn = peer_psn;
    urma_token_t peer_token{};
    peer_token.token = region.token_value;
    remote_jetty_ = import_jetty_ex_(context_, &remote, &peer_token, &active);
    if (remote_jetty_ == nullptr) return Error("urma_import_jetty_ex");
    initialized_ = true;
    return Status::OK();
  }

  Status Transfer(bool write, std::uint64_t offset, void* data, std::size_t bytes) {
    if (!initialized_) return Status::InvalidState("CPU URMA transport is not initialized");
    if (data == nullptr || bytes == 0 || offset > remote_region_.length || bytes > remote_region_.length - offset ||
        bytes > std::numeric_limits<std::uint32_t>::max()) return Status::InvalidArgument("invalid transfer range");
    // The probe owns one registered host segment.  User buffers are copied
    // through this registered bounce buffer; a production backend can replace
    // this with per-buffer registration or a registration cache.
    if (write) std::memcpy(buffer_, data, bytes);
    urma_sge_t local{};
    local.addr = reinterpret_cast<std::uint64_t>(buffer_);
    local.len = static_cast<std::uint32_t>(bytes);
    local.tseg = local_seg_;
    urma_sge_t remote{};
    remote.addr = remote_region_.base + offset;
    remote.len = static_cast<std::uint32_t>(bytes);
    remote.tseg = remote_seg_handle_;
    urma_jfs_wr_t wr{};
    wr.opcode = write ? URMA_OPC_WRITE : URMA_OPC_READ;
    wr.flag.bs.complete_enable = 1;
    wr.tjetty = remote_jetty_;
    wr.user_ctx = 0xC0DE;
    if (write) { wr.rw.src = {&local, 1}; wr.rw.dst = {&remote, 1}; }
    else { wr.rw.src = {&remote, 1}; wr.rw.dst = {&local, 1}; }
    urma_jfs_wr_t* bad = nullptr;
    if (urma_post_jetty_send_wr(local_jetty_, &wr, &bad) != URMA_SUCCESS) return Error("urma_post_jetty_send_wr");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      urma_cr_t cr{};
      const int polled = urma_poll_jfc(jfc_, 1, &cr);
      if (polled < 0) return Error("urma_poll_jfc");
      if (polled == 1) {
        if (cr.status != URMA_CR_SUCCESS || cr.user_ctx != 0xC0DE) return Status::Internal("URMA completion reported failure");
        if (!write) std::memcpy(data, buffer_, bytes);
        return Status::OK();
      }
      std::this_thread::yield();
    }
    return Status::Internal("timed out waiting for URMA completion");
  }

  Status TransferDeviceWrite(std::uint64_t offset, const void* device_source,
                             std::size_t bytes) {
    if (!initialized_) return Status::InvalidState("CPU URMA transport is not initialized");
    if (device_source == nullptr || bytes == 0 || offset > remote_region_.length ||
        bytes > remote_region_.length - offset ||
        bytes > std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument("invalid device transfer range");
    }

    // Try to register the caller-owned HBM/device VA directly. No CPU memcpy
    // is performed here. Whether this succeeds is provider/driver dependent;
    // this is precisely what the HBM probe is intended to establish.
    urma_token_id_t* device_token_id = urma_alloc_token_id(context_);
    if (device_token_id == nullptr) {
      return Status::Internal("HBM local segment token allocation failed; direct device-memory URMA is unavailable");
    }
    urma_seg_cfg_t device_seg_cfg{};
    device_seg_cfg.va = reinterpret_cast<std::uint64_t>(device_source);
    device_seg_cfg.len = bytes;
    device_seg_cfg.token_id = device_token_id;
    device_seg_cfg.token_value.token = 0xBEEF0001U;
    device_seg_cfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
#if defined(URMA_ACCESS_LOCAL_WRITE)
    device_seg_cfg.flag.bs.access = URMA_ACCESS_LOCAL_WRITE;
#elif defined(URMA_ACCESS_LOCAL_ONLY)
    device_seg_cfg.flag.bs.access = URMA_ACCESS_LOCAL_ONLY;
#else
    device_seg_cfg.flag.bs.access = 1U;
#endif
    device_seg_cfg.flag.bs.non_pin = 1;
    device_seg_cfg.flag.bs.token_id_valid = 1;
    urma_target_seg_t* device_seg = urma_register_seg(context_, &device_seg_cfg);
    if (device_seg == nullptr) {
      const int registration_errno = errno;
      (void)urma_free_token_id(device_token_id);
      errno = registration_errno;
      return Status::Internal(
          "HBM local segment registration failed; liburma/provider may not support device memory "
          "(errno=" + std::to_string(registration_errno) + ")");
    }

    urma_sge_t local{};
    local.addr = reinterpret_cast<std::uint64_t>(device_source);
    local.len = static_cast<std::uint32_t>(bytes);
    local.tseg = device_seg;
    urma_sge_t remote{};
    remote.addr = remote_region_.base + offset;
    remote.len = static_cast<std::uint32_t>(bytes);
    remote.tseg = remote_seg_handle_;
    urma_jfs_wr_t wr{};
    wr.opcode = URMA_OPC_WRITE;
    wr.flag.bs.complete_enable = 1;
    wr.tjetty = remote_jetty_;
    wr.user_ctx = 0xD3A1CEULL;
    wr.rw.src = {&local, 1};
    wr.rw.dst = {&remote, 1};
    urma_jfs_wr_t* bad = nullptr;
    Status result = Status::OK();
    if (urma_post_jetty_send_wr(local_jetty_, &wr, &bad) != URMA_SUCCESS) {
      result = Status::Internal("HBM direct URMA WRITE submission failed; device memory may not be DMA-addressable");
    } else {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      bool completed = false;
      while (std::chrono::steady_clock::now() < deadline) {
        urma_cr_t cr{};
        const int polled = urma_poll_jfc(jfc_, 1, &cr);
        if (polled < 0) {
          result = Error("urma_poll_jfc for HBM WRITE");
          completed = true;
          break;
        }
        if (polled == 1) {
          if (cr.status != URMA_CR_SUCCESS || cr.user_ctx != 0xD3A1CEULL) {
            result = Status::Internal("HBM direct URMA WRITE completion reported failure");
          }
          completed = true;
          break;
        }
        std::this_thread::yield();
      }
      if (!completed) result = Status::Internal("timed out waiting for HBM URMA WRITE completion");
    }
    (void)urma_unregister_seg(device_seg);
    (void)urma_free_token_id(device_token_id);
    return result;
  }

  Status Shutdown() {
    initialized_ = false;
    if (remote_jetty_ != nullptr) { (void)urma_unimport_jetty(remote_jetty_); remote_jetty_ = nullptr; }
    if (local_jetty_ != nullptr) { (void)urma_delete_jetty(local_jetty_); local_jetty_ = nullptr; }
    if (jfr_ != nullptr) { (void)urma_delete_jfr(jfr_); jfr_ = nullptr; }
    if (remote_seg_handle_ != nullptr) { (void)urma_unimport_seg(remote_seg_handle_); remote_seg_handle_ = nullptr; }
    if (local_seg_ != nullptr) { (void)urma_unregister_seg(local_seg_); local_seg_ = nullptr; }
    if (token_id_ != nullptr) { (void)urma_free_token_id(token_id_); token_id_ = nullptr; }
    if (owns_remote_region_ && link_ != nullptr && remote_region_.base != 0) {
      if (remote_region_.device_memory) (void)link_->ReleaseHbmRegion(remote_region_.session_id);
      else (void)link_->ReleaseRegion(remote_region_.base);
      remote_region_ = {};
    }
    owns_remote_region_ = false;
    if (buffer_ != nullptr) { free(buffer_); buffer_ = nullptr; }
    if (jfc_ != nullptr) { (void)urma_delete_jfc(jfc_); jfc_ = nullptr; }
    if (context_ != nullptr) { (void)urma_delete_context(context_); context_ = nullptr; }
    if (urma_initialized_ && urma_owned_) { (void)urma_uninit(); }
    urma_initialized_ = false;
    urma_owned_ = false;
    link_.reset();
    return Status::OK();
  }

  CpuUrmaTransportOptions options_{};
  std::unique_ptr<peer::PeerLink> link_;
  peer::PeerRegion remote_region_{};
  urma_device_t* device_{nullptr};
  urma_context_t* context_{nullptr};
  urma_jfc_t* jfc_{nullptr};
  urma_jfr_t* jfr_{nullptr};
  urma_jetty_t* local_jetty_{nullptr};
  urma_target_jetty_t* remote_jetty_{nullptr};
  urma_token_id_t* token_id_{nullptr};
  urma_target_seg_t* local_seg_{nullptr};
  urma_target_seg_t* remote_seg_handle_{nullptr};
  urma_seg_t remote_seg_{};
  void* buffer_{nullptr};
  std::uint32_t eid_index_{0};
  std::uint32_t token_value_{0};
  bool urma_initialized_{false};
  bool urma_owned_{false};
  bool initialized_{false};
  bool owns_remote_region_{false};
  GetTpList get_tp_list_{nullptr};
  ImportJettyEx import_jetty_ex_{nullptr};
};

CpuUrmaTransport::CpuUrmaTransport() : impl_(std::make_unique<Impl>()) {}
CpuUrmaTransport::~CpuUrmaTransport() = default;
Status CpuUrmaTransport::Initialize(const CpuUrmaTransportOptions& options) { return impl_->Initialize(options); }
Status CpuUrmaTransport::Write(std::uint64_t offset, const void* source, std::size_t bytes) {
  return impl_->Transfer(true, offset, const_cast<void*>(source), bytes);
}
Status CpuUrmaTransport::WriteDeviceMemory(std::uint64_t offset, const void* device_source,
                                           std::size_t bytes) {
  return impl_->TransferDeviceWrite(offset, device_source, bytes);
}
Status CpuUrmaTransport::Read(std::uint64_t offset, void* destination, std::size_t bytes) {
  return impl_->Transfer(false, offset, destination, bytes);
}
Status CpuUrmaTransport::Shutdown() { return impl_->Shutdown(); }
bool CpuUrmaTransport::initialized() const { return impl_->initialized_; }
std::uint64_t CpuUrmaTransport::region_bytes() const { return impl_->remote_region_.length; }

}  // namespace kvaf
