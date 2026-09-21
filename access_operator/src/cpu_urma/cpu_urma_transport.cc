#include "cpu_urma/cpu_urma_transport.h"
#include "cpu_urma/sgl_batch_plan.h"
#include "cpu_urma/jetty_batch_dispatch.h"

#include <dlfcn.h>
#include <errno.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>
#include <urma_api.h>

#include "peer_link.h"

namespace kvaf {
namespace {

constexpr std::uint32_t kJfcDepth = 16384;
constexpr std::uint32_t kJettyDepth = 8192;
constexpr std::uint32_t kMaxJettyCount = 16;
constexpr std::uint32_t kLocalSgeCount = 1;
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
// The deployed liburma uses the expanded SGE ABI (addr, len, tseg,
// user_tseg), which is 32 bytes. Older public headers omit user_tseg and
// declare urma_sge_t as 24 bytes. A one-entry SGL hides this stride mismatch,
// but every entry after the first is decoded at the wrong address. Keep SGL
// storage in the expanded wire layout and cast only at the API boundary.
struct SglSgeAbi {
  std::uint64_t addr;
  std::uint32_t len;
  urma_target_seg_t* tseg;
  void* user_tseg;
};
static_assert(sizeof(SglSgeAbi) == 32 && offsetof(SglSgeAbi, tseg) == 16 &&
                  offsetof(SglSgeAbi, user_tseg) == 24,
              "unexpected expanded URMA SGE ABI");
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
  struct SglBatchQueue {
    // Each WR has one contiguous local SGE and one contiguous remote SGE.
    std::vector<SglSgeAbi> local;
    std::vector<SglSgeAbi> remote;
    std::vector<urma_jfs_wr_t> wrs;
  };

  struct SglBatchJob {
    bool write{false};
    const CpuUrmaTransport::HostSge* buffers{nullptr};
    std::size_t remote_sgl_size{1U};
    const cpu_urma_detail::SglGroup* groups{nullptr};
    urma_target_seg_t* single_host_segment{nullptr};
    std::vector<SglBatchQueue> queues;
    const std::uint64_t* remote_offsets{nullptr};
    std::vector<std::uint64_t> generated_remote_offsets;
    std::vector<urma_target_seg_t*> host_segments;
    std::chrono::steady_clock::time_point deadline{};
  };

  struct WorkerResult {
    std::uint64_t prepare_ns{0};
    std::uint64_t post_ns{0};
    std::string error;
  };

  void PrepareSglBatchEntry(SglBatchJob& job, std::size_t wr_index) {
    const std::size_t jetty = wr_index % local_jettys_.size();
    const std::size_t position = wr_index / local_jettys_.size();
    const std::size_t first = job.groups != nullptr
        ? job.groups[wr_index].first - job.groups[0].first
        : wr_index;
    const bool coalesced = job.groups != nullptr;
    SglBatchQueue& queue = job.queues[jetty];
    SglSgeAbi* local = &queue.local[position];
    const std::uint64_t bytes = coalesced
        ? job.groups[wr_index].bytes : job.buffers[first].bytes;
    *local = {};
    local->addr = reinterpret_cast<std::uint64_t>(job.buffers[first].data);
    local->len = static_cast<std::uint32_t>(bytes);
    local->tseg = job.single_host_segment != nullptr
        ? job.single_host_segment : job.host_segments[first];
    SglSgeAbi* remote = &queue.remote[position];
    *remote = {};
    remote->addr = remote_region_.base + job.remote_offsets[first];
    remote->len = static_cast<std::uint32_t>(bytes);
    remote->tseg = remote_seg_handle_;
    urma_jfs_wr_t& wr = queue.wrs[position];
    wr = {};
    wr.opcode = job.write ? URMA_OPC_WRITE : URMA_OPC_READ;
    const bool completes_queue = position + 1U == queue.wrs.size();
    // The final CQE is the completion fence for the entire Jetty chain.
    // Completion ordering is required for both WRITE and READ, especially
    // when one WRITE coalesces several adjacent rows.
    wr.flag.bs.place_order =
        completes_queue ? URMA_STRONG_ORDER : URMA_RELAX_ORDER;
    wr.flag.bs.comp_order = 1U;
    wr.flag.bs.complete_enable = completes_queue ? 1U : 0U;
    wr.tjetty = remote_jettys_[jetty];
    wr.user_ctx = 0x5A1B0000ULL +
                  (static_cast<std::uint64_t>(jetty) << 32U) + position;
    urma_sg_t local_sg{};
    local_sg.sge = reinterpret_cast<urma_sge_t*>(local);
    local_sg.num_sge = kLocalSgeCount;
    urma_sg_t remote_sg{};
    remote_sg.sge = reinterpret_cast<urma_sge_t*>(remote);
    remote_sg.num_sge = 1U;
    if (job.write) {
      wr.rw.src = local_sg;
      wr.rw.dst = remote_sg;
    } else {
      wr.rw.src = remote_sg;
      wr.rw.dst = local_sg;
    }
    wr.next = position + 1U < queue.wrs.size()
                  ? &queue.wrs[position + 1U]
                  : nullptr;
  }

  std::string PrepareAndPostSglBatch(SglBatchJob& job,
                                     std::uint64_t* prepare_ns,
                                     std::uint64_t* post_ns) {
    std::array<WorkerResult, kMaxJettyCount> results{};
    const std::size_t chunk_size = wqes_per_doorbell_ == 0U
        ? kJettyDepth : wqes_per_doorbell_;
    // A worker owns the entire Jetty chain, SQ and doorbell. No other worker
    // writes or submits that queue, so no per-wave barrier is necessary.
#pragma omp parallel for schedule(static, 1) num_threads(static_cast<int>(job.queues.size()))
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(job.queues.size()); ++index) {
      const auto jetty = static_cast<std::size_t>(index);
      auto& queue = job.queues[jetty];
      auto& result = results[jetty];
      auto post_start = std::chrono::steady_clock::now();
      cpu_urma_detail::DispatchJettyChunks(queue.wrs.size(), chunk_size,
      [&](std::size_t begin, std::size_t end) {
        const auto prepare_start = std::chrono::steady_clock::now();
        for (std::size_t position = begin; position < end; ++position) {
          PrepareSglBatchEntry(job, position * job.queues.size() + jetty);
        }
        queue.wrs[end - 1U].next = nullptr;
        post_start = std::chrono::steady_clock::now();
        result.prepare_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(post_start - prepare_start).count());
      },
      [&](std::size_t begin, std::size_t) {
        urma_jfs_wr_t* bad = nullptr;
        const auto status = urma_post_jetty_send_wr(local_jettys_[jetty], &queue.wrs[begin], &bad);
        result.post_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - post_start).count());
        if (status != URMA_SUCCESS) {
          result.error = "SGL batch post failed (jetty=" + std::to_string(jetty) +
              ", status=" + std::to_string(static_cast<int>(status)) +
              ", bad_wr=" + std::to_string(reinterpret_cast<std::uintptr_t>(bad)) + ")";
          return false;
        }
        return true;
      });
    }
    // Keep phase timings additive and comparable to wall time: select one
    // critical worker instead of summing work performed concurrently.
    const auto result_end = results.begin() + job.queues.size();
    const auto critical = std::max_element(results.begin(), result_end,
        [](const WorkerResult& a, const WorkerResult& b) {
          return a.prepare_ns + a.post_ns < b.prepare_ns + b.post_ns;
        });
    *prepare_ns = critical->prepare_ns;
    *post_ns = critical->post_ns;
    for (auto result = results.begin(); result != result_end; ++result) {
      if (!result->error.empty()) return result->error;
    }
    return {};
  }

  std::string PollSglBatch(SglBatchJob& job) {
    for (std::size_t jetty = 0U; jetty < local_jettys_.size(); ++jetty) {
      const SglBatchQueue& queue = job.queues[jetty];
      if (queue.wrs.empty()) continue;
      const std::uint64_t expected_ctx = queue.wrs.back().user_ctx;
      bool completed = false;
      while (std::chrono::steady_clock::now() < job.deadline) {
        urma_cr_t cr{};
        const int polled = urma_poll_jfc(jfcs_[jetty], 1, &cr);
        if (polled < 0) {
          return "urma_poll_jfc for SGL batch failed (jetty=" +
                 std::to_string(jetty) + ")";
        }
        if (polled == 0) continue;
        if (cr.status != URMA_CR_SUCCESS || cr.user_ctx != expected_ctx) {
          return "SGL batch completion failed (jetty=" +
                 std::to_string(jetty) + ", status=" +
                 std::to_string(static_cast<int>(cr.status)) +
                 ", user_ctx=" +
                 std::to_string(static_cast<unsigned long long>(cr.user_ctx)) +
                 ", expected_ctx=" +
                 std::to_string(static_cast<unsigned long long>(expected_ctx)) +
                 ")";
        }
        completed = true;
        break;
      }
      if (!completed) {
        return "timed out waiting for SGL batch completion (jetty=" +
               std::to_string(jetty) + ")";
      }
    }
    return {};
  }

  ~Impl() { (void)Shutdown(); }

  Status Initialize(const CpuUrmaTransportOptions& options) {
    if (options.peer_address.empty() || options.region_bytes == 0 ||
        options.region_bytes > std::numeric_limits<std::size_t>::max() ||
        options.wqes_per_doorbell > kJettyDepth || options.remote_sgl_size == 0U ||
        options.jetty_count == 0U || options.jetty_count > kMaxJettyCount) {
      return Status::InvalidArgument(
          "CPU URMA transport requires peer_address, non-zero region_bytes, "
          "non-zero remote_sgl_size, jetty_count in [1,16] and "
          "wqes_per_doorbell in [0,8192]");
    }
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
    urma_device_attr_t device_attr{};
    max_message_bytes_ = std::numeric_limits<std::uint32_t>::max();
    if (urma_query_device(device_, &device_attr) == URMA_SUCCESS) {
      std::fprintf(stderr,
                   "[KVAF_CPU_URMA] DEVICE_CAP max_jfs=%u max_jfr=%u max_jetty=%u "
                   "max_jfc_depth=%u max_jfs_depth=%u max_jfr_depth=%u "
                   "max_jfs_sge=%u max_jfs_rsge=%u max_jfr_sge=%u "
                   "max_msg_size=%llu page_size_cap=0x%llx max_oor_cnt=%u mn=%u\n",
                   device_attr.dev_cap.max_jfs,
                   device_attr.dev_cap.max_jfr,
                   device_attr.dev_cap.max_jetty,
                   device_attr.dev_cap.max_jfc_depth,
                   device_attr.dev_cap.max_jfs_depth,
                   device_attr.dev_cap.max_jfr_depth,
                   device_attr.dev_cap.max_jfs_sge,
                   device_attr.dev_cap.max_jfs_rsge,
                   device_attr.dev_cap.max_jfr_sge,
                   static_cast<unsigned long long>(device_attr.dev_cap.max_msg_size),
                   static_cast<unsigned long long>(device_attr.dev_cap.page_size_cap),
                   device_attr.dev_cap.max_oor_cnt,
                   device_attr.dev_cap.mn);
      if (device_attr.dev_cap.max_msg_size != 0U) {
        max_message_bytes_ = std::min<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max(),
            device_attr.dev_cap.max_msg_size);
      }
    }
    // Coalescing still produces one contiguous remote SGE. max_jfs_rsge does
    // not limit how many adjacent logical rows can be represented by that SGE.
    batch_remote_sgl_size_ = options.remote_sgl_size;
    get_tp_list_ = reinterpret_cast<GetTpList>(dlsym(RTLD_DEFAULT, "urma_get_tp_list"));
    import_jetty_ex_ = reinterpret_cast<ImportJettyEx>(dlsym(RTLD_DEFAULT, "urma_import_jetty_ex"));
    if (get_tp_list_ == nullptr || import_jetty_ex_ == nullptr) {
      return Status::NotImplemented("liburma lacks CTP-aware import APIs");
    }

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

    jfcs_.reserve(options.jetty_count);
    jfrs_.reserve(options.jetty_count);
    local_jettys_.reserve(options.jetty_count);
    remote_jettys_.reserve(options.jetty_count);
    for (std::uint32_t jetty_index = 0U; jetty_index < options.jetty_count;
         ++jetty_index) {
      urma_jfc_cfg_t jfc_cfg{};
      jfc_cfg.depth = kJfcDepth;
      urma_jfc_t* jfc = urma_create_jfc(context_, &jfc_cfg);
      if (jfc == nullptr) return Error("urma_create_jfc");
      jfcs_.push_back(jfc);

      urma_jfr_cfg_t jfr_cfg{};
      jfr_cfg.depth = kJettyDepth;
      jfr_cfg.trans_mode = URMA_TM_RM;
      jfr_cfg.max_sge = 1;
      jfr_cfg.jfc = jfc;
      jfr_cfg.token_value.token = region.token_value;
      urma_jfr_t* jfr = urma_create_jfr(context_, &jfr_cfg);
      if (jfr == nullptr) return Error("urma_create_jfr");
      jfrs_.push_back(jfr);

      urma_jetty_cfg_t jetty_cfg{};
      jetty_cfg.flag.bs.share_jfr = URMA_SHARE_JFR;
      jetty_cfg.jfs_cfg.depth = kJettyDepth;
      jetty_cfg.jfs_cfg.trans_mode = URMA_TM_RM;
      jetty_cfg.jfs_cfg.priority = URMA_MAX_PRIORITY;
      jetty_cfg.jfs_cfg.max_sge = kLocalSgeCount;
      // RC WRITE uses one contiguous remote range. remote_sgl_size controls
      // how many adjacent rows may be coalesced into that range.
      jetty_cfg.jfs_cfg.max_rsge = 1U;
      jetty_cfg.jfs_cfg.rnr_retry = URMA_TYPICAL_RNR_RETRY;
      jetty_cfg.jfs_cfg.err_timeout = URMA_TYPICAL_ERR_TIMEOUT;
      jetty_cfg.jfs_cfg.jfc = jfc;
      jetty_cfg.shared.jfr = jfr;
      jetty_cfg.shared.jfc = jfc;
      urma_jetty_t* local_jetty = urma_create_jetty(context_, &jetty_cfg);
      if (local_jetty == nullptr) return Error("urma_create_jetty");
      local_jettys_.push_back(local_jetty);
      std::fprintf(stderr,
                   "[KVAF_CPU_URMA] JETTY index=%u trans_mode=%u max_sge=%u "
                   "max_rsge=%u depth=%u\n",
                   jetty_index,
                   static_cast<unsigned>(local_jetty->jetty_cfg.jfs_cfg.trans_mode),
                   static_cast<unsigned>(local_jetty->jetty_cfg.jfs_cfg.max_sge),
                   static_cast<unsigned>(local_jetty->jetty_cfg.jfs_cfg.max_rsge),
                   kJettyDepth);

      TpQuery query{};
      query.flag = 1;
      query.trans_mode = static_cast<std::uint32_t>(URMA_TM_RM);
      std::memcpy(query.local_eid, context_->eid.raw, 16);
      std::memcpy(query.peer_eid, region.eid.data(), 16);
      TpInfo info{};
      std::uint32_t tp_count = 1;
      if (get_tp_list_(context_, &query, &tp_count, &info) != 0 ||
          tp_count == 0 || info.tp_handle == 0) {
        return Error("urma_get_tp_list");
      }
      std::array<std::uint8_t, peer::kQpKeyBytes> local_key{};
      std::memcpy(local_key.data(), &local_jetty->jetty_id,
                  sizeof(local_jetty->jetty_id));
      const urma_transport_mode_t mode = local_jetty->jetty_cfg.jfs_cfg.trans_mode;
      std::memcpy(local_key.data() + sizeof(local_jetty->jetty_id), &mode,
                  sizeof(mode));
      std::array<std::uint8_t, peer::kQpKeyBytes> peer_key{};
      std::uint32_t peer_key_bytes = 0;
      std::uint64_t peer_tp = 0;
      std::uint32_t peer_psn = 0;
      status = link_->CreateChannel(region.base, local_key.data(), kKeyBytes,
                                    context_->eid.raw, region.token_value,
                                    info.tp_handle, 1, &peer_key,
                                    &peer_key_bytes, &peer_tp, &peer_psn);
      if (!status.ok()) return status;
      RemoteJettyEx remote{};
      std::memcpy(&remote.jetty_id, peer_key.data(), sizeof(remote.jetty_id));
      std::memcpy(&remote.trans_mode,
                  peer_key.data() + sizeof(remote.jetty_id),
                  sizeof(remote.trans_mode));
      remote.policy = static_cast<std::uint32_t>(URMA_JETTY_GRP_POLICY_RR);
      remote.type = static_cast<std::uint32_t>(URMA_JETTY);
      urma_import_jetty_flag_t jetty_flag{};
      jetty_flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
      remote.flag.value = jetty_flag.value;
      // URMA_CTP is enum value 1 in the expanded import ABI.
      remote.tp_type = 1U;
      ActiveTp active{};
      active.tp_handle = info.tp_handle;
      active.peer_tp_handle = peer_tp;
      active.tx_psn = 1;
      active.rx_psn = peer_psn;
      urma_token_t peer_token{};
      peer_token.token = region.token_value;
      urma_target_jetty_t* remote_jetty =
          import_jetty_ex_(context_, &remote, &peer_token, &active);
      if (remote_jetty == nullptr) return Error("urma_import_jetty_ex");
      remote_jettys_.push_back(remote_jetty);
    }
    if (local_jettys_.empty()) return Error("no CPU Jettys created");
    wqes_per_doorbell_ = options.wqes_per_doorbell;
    std::fprintf(stderr, "[KVAF_CPU_URMA] JETTYS_READY count=%zu omp_threads=%u wqes_per_doorbell=%zu local_sge=1 remote_sge=1 requested_remote_sgl_size=%u remote_sgl_size=%zu max_wr_bytes=%llu depth=%u\n",
                 local_jettys_.size(), options.jetty_count,
                 wqes_per_doorbell_,
                 options.remote_sgl_size, batch_remote_sgl_size_,
                 static_cast<unsigned long long>(max_message_bytes_), kJettyDepth);
    initialized_ = true;
    return Status::OK();
  }

  struct RegisteredHostBuffer {
    const void* address{nullptr};
    std::size_t bytes{0};
    urma_token_id_t* token_id{nullptr};
    urma_target_seg_t* segment{nullptr};
  };

  RegisteredHostBuffer* FindRegisteredHostBuffer(const void* address, std::size_t bytes) {
    const auto requested = reinterpret_cast<std::uintptr_t>(address);
    for (auto& entry : direct_host_buffers_) {
      const auto base = reinterpret_cast<std::uintptr_t>(entry.address);
      if (requested >= base && bytes <= entry.bytes && requested - base <= entry.bytes - bytes) {
        return &entry;
      }
    }
    return nullptr;
  }

  Status RegisterHostBuffer(const void* address, std::size_t bytes,
                            RegisteredHostBuffer** result) {
    if (result == nullptr || address == nullptr || bytes == 0) {
      return Status::InvalidArgument("invalid direct host buffer");
    }
    if (auto* existing = FindRegisteredHostBuffer(address, bytes); existing != nullptr) {
      *result = existing;
      return Status::OK();
    }
    const auto requested = reinterpret_cast<std::uintptr_t>(address);
    const auto base = requested & ~static_cast<std::uintptr_t>(kPageBytes - 1U);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - requested) {
      return Status::InvalidArgument("direct host buffer range overflow");
    }
    const auto end = requested + bytes;
    if (end > std::numeric_limits<std::uintptr_t>::max() - (kPageBytes - 1U)) {
      return Status::InvalidArgument("direct host buffer range overflow");
    }
    const auto rounded_end = (end + kPageBytes - 1U) &
                             ~static_cast<std::uintptr_t>(kPageBytes - 1U);
    if (rounded_end < base) return Status::InvalidArgument("direct host buffer range overflow");
    auto* token_id = urma_alloc_token_id(context_);
    if (token_id == nullptr) return Error("urma_alloc_token_id for direct host buffer");
    urma_seg_cfg_t cfg{};
    cfg.va = static_cast<std::uint64_t>(base);
    // UB providers register page-granular ranges. Direct callers must keep
    // the backing allocation valid through this rounded-up length.
    cfg.len = rounded_end - base;
    cfg.token_id = token_id;
    cfg.token_value.token = 0xD1CE0000U + static_cast<std::uint32_t>(direct_host_buffers_.size() + 1U);
    cfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
#if defined(URMA_ACCESS_LOCAL_WRITE)
    cfg.flag.bs.access = URMA_ACCESS_LOCAL_WRITE;
#elif defined(URMA_ACCESS_LOCAL_ONLY)
    cfg.flag.bs.access = URMA_ACCESS_LOCAL_ONLY;
#else
    cfg.flag.bs.access = 1U;
#endif
    cfg.flag.bs.non_pin = 0;
    cfg.flag.bs.token_id_valid = 1;
    auto* segment = urma_register_seg(context_, &cfg);
    if (segment == nullptr) {
      const int registration_errno = errno;
      (void)urma_free_token_id(token_id);
      errno = registration_errno;
      return Status::Internal("urma_register_seg for direct host buffer failed (va=0x" +
                             std::to_string(reinterpret_cast<std::uintptr_t>(address)) +
                             ", bytes=" + std::to_string(cfg.len) +
                             ", errno=" + std::to_string(registration_errno) + ")");
    }
    direct_host_buffers_.push_back(RegisteredHostBuffer{
        reinterpret_cast<const void*>(base), static_cast<std::size_t>(cfg.len), token_id, segment});
    *result = &direct_host_buffers_.back();
    return Status::OK();
  }

  Status RegisterDirectHostMemory(void* base, std::size_t bytes) {
    RegisteredHostBuffer* host = nullptr;
    Status status = RegisterHostBuffer(base, bytes, &host);
    if (status.ok()) explicit_host_region_ = *host;
    return status;
  }

  Status TransferSglBatchLocked(bool write, std::uint64_t offset,
                                const CpuUrmaTransport::HostSge* buffers,
                                const std::uint64_t* remote_offsets,
                                std::size_t count,
                                const cpu_urma_detail::SglGroup* groups = nullptr,
                                std::size_t group_count = 0U,
                                std::size_t remote_sgl_size = 1U,
                                bool rows_validated = false) {
    const auto setup_start = std::chrono::steady_clock::now();
    if (!initialized_) return Status::InvalidState("CPU URMA transport is not initialized");
    if (buffers == nullptr || count == 0U || offset > remote_region_.length ||
        remote_sgl_size == 0U || remote_sgl_size > batch_remote_sgl_size_ ||
        (remote_sgl_size > 1U && (!write || groups == nullptr || remote_offsets == nullptr))) {
      return Status::InvalidArgument("invalid SGL batch transfer");
    }
    const std::size_t wr_count = groups != nullptr
        ? group_count : count;
    if (wr_count == 0U || wr_count > kJettyDepth * local_jettys_.size()) {
      return Status::InvalidArgument("SGL batch exceeds Jetty queue capacity");
    }
    SglBatchJob& job = sgl_batch_workspace_;
    job.write = write;
    job.buffers = buffers;
    job.remote_sgl_size = remote_sgl_size;
    job.groups = groups;
    job.single_host_segment = nullptr;
    job.queues.resize(local_jettys_.size());
    for (std::size_t jetty = 0U; jetty < local_jettys_.size(); ++jetty) {
      const std::size_t queue_count = wr_count / local_jettys_.size() +
          (jetty < wr_count % local_jettys_.size() ? 1U : 0U);
      job.queues[jetty].local.resize(queue_count);
      job.queues[jetty].remote.resize(queue_count);
      job.queues[jetty].wrs.resize(queue_count);
    }
    const bool generate_remote_offsets = remote_offsets == nullptr;
    if (generate_remote_offsets) job.generated_remote_offsets.resize(count);
    // Explicit offsets remain valid for this synchronous transfer, so avoid
    // copying the caller's array on every batch.
    job.remote_offsets = generate_remote_offsets
        ? job.generated_remote_offsets.data() : remote_offsets;
    std::uint64_t total = 0U;
    bool use_explicit_host_region = explicit_host_region_.segment != nullptr;
    const auto host_span_is_registered = [&](const void* data, std::uint64_t bytes) {
      const auto requested = reinterpret_cast<std::uintptr_t>(data);
      const auto base = reinterpret_cast<std::uintptr_t>(explicit_host_region_.address);
      return requested >= base && bytes <= explicit_host_region_.bytes &&
          requested - base <= explicit_host_region_.bytes - bytes;
    };
    if (rows_validated && groups != nullptr) {
      // Planning already checked every row and made each group contiguous.
      // Check registered-memory coverage once per resulting WR instead of
      // walking all source rows again.
      for (std::size_t i = 0U; use_explicit_host_region && i < group_count; ++i) {
        const std::size_t first = groups[i].first - groups[0].first;
        use_explicit_host_region = host_span_is_registered(
            buffers[first].data, groups[i].bytes);
      }
    } else {
      for (std::size_t i = 0U; i < count; ++i) {
        const auto& buffer = buffers[i];
        if (generate_remote_offsets &&
            total > std::numeric_limits<std::uint64_t>::max() - offset) {
          return Status::InvalidArgument("invalid SGL batch entry");
        }
        const std::uint64_t remote_offset =
            generate_remote_offsets ? offset + total : remote_offsets[i];
        const bool invalid_entry = !rows_validated &&
            (buffer.data == nullptr || buffer.bytes == 0U ||
            buffer.bytes > std::numeric_limits<std::uint32_t>::max() ||
            buffer.bytes > std::numeric_limits<std::uintptr_t>::max() -
                reinterpret_cast<std::uintptr_t>(buffer.data) ||
            remote_offset > remote_region_.length ||
            buffer.bytes > remote_region_.length - remote_offset);
        const bool total_overflow = generate_remote_offsets &&
            buffer.bytes > std::numeric_limits<std::uint64_t>::max() - total;
        if (invalid_entry || total_overflow) {
          return Status::InvalidArgument("invalid SGL batch entry");
        }
        if (use_explicit_host_region &&
            !host_span_is_registered(buffer.data, buffer.bytes)) {
          use_explicit_host_region = false;
        }
        if (generate_remote_offsets) {
          job.generated_remote_offsets[i] = remote_offset;
          total += buffer.bytes;
        }
      }
    }
    if (use_explicit_host_region) {
      job.single_host_segment = explicit_host_region_.segment;
    } else if (remote_sgl_size > 1U) {
      job.host_segments.resize(count);
      // Register each entire contiguous source span: a single SGE must use
      // one segment covering all rows, even across page/registration boundaries.
      for (std::size_t i = 0U; i < group_count; ++i) {
        const std::size_t first = groups[i].first - groups[0].first;
        RegisteredHostBuffer* host = nullptr;
        const Status status = RegisterHostBuffer(
            buffers[first].data, static_cast<std::size_t>(groups[i].bytes), &host);
        if (!status.ok()) return status;
        job.host_segments[first] = host->segment;
      }
    } else {
      job.host_segments.resize(count);
      for (std::size_t i = 0U; i < count; ++i) {
        RegisteredHostBuffer* host = FindRegisteredHostBuffer(
            buffers[i].data, buffers[i].bytes);
        if (host == nullptr) {
          Status status = RegisterHostBuffer(buffers[i].data, buffers[i].bytes, &host);
          if (!status.ok()) return status;
        }
        job.host_segments[i] = host->segment;
      }
    }
    job.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    std::uint64_t prepare_ns = 0U;
    std::uint64_t post_ns = 0U;
    const auto dispatch_start = std::chrono::steady_clock::now();
    const auto setup_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            dispatch_start - setup_start).count());
    const auto post_error = PrepareAndPostSglBatch(job, &prepare_ns, &post_ns);
    const auto poll_start = std::chrono::steady_clock::now();
    const auto dispatch_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            poll_start - dispatch_start).count());
    const auto poll_error = post_error.empty() ? PollSglBatch(job) : post_error;
    const auto poll_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - poll_start).count());
    last_batch_timing_.prepare_ns = prepare_ns;
    last_batch_timing_.post_ns = post_ns;
    last_batch_timing_.poll_ns = poll_ns;
    last_batch_timing_.setup_ns = setup_ns;
    last_batch_timing_.dispatch_ns = dispatch_ns;
    last_batch_timing_.wr_count = wr_count;
    if (!post_error.empty()) return Status::Internal(post_error);
    if (!poll_error.empty()) return Status::Internal(poll_error);
    return Status::OK();
  }

  Status TransferSglBatch(bool write, std::uint64_t offset,
                          const CpuUrmaTransport::HostSge* buffers,
                          const std::uint64_t* remote_offsets,
                          std::size_t count) {
    std::unique_lock<std::mutex> submit_lock(batch_submit_mutex_);
    const auto start = std::chrono::steady_clock::now();
    last_batch_timing_ = {};
    const Status status = TransferSglBatchLocked(
        write, offset, buffers, remote_offsets, count);
    if (!status.ok()) {
      last_batch_timing_ = {};
      return status;
    }
    last_batch_timing_.total_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    return Status::OK();
  }

  Status TransferBatch(bool write, const std::uint64_t* remote_offsets,
                       const CpuUrmaTransport::HostSge* buffers,
                       std::size_t count) {
    if (!initialized_) {
      return Status::InvalidState("CPU URMA transport is not initialized");
    }
    if (remote_offsets == nullptr || buffers == nullptr || count == 0U) {
      return Status::InvalidArgument("invalid non-contiguous batch transfer");
    }
    std::unique_lock<std::mutex> submit_lock(batch_submit_mutex_);
    const auto start = std::chrono::steady_clock::now();
    last_batch_timing_ = {};
    const std::size_t batch_capacity = kJettyDepth * local_jettys_.size();
    const bool coalesce = write && batch_remote_sgl_size_ > 1U;
    const auto plan_start = std::chrono::steady_clock::now();
    const Status plan_status = coalesce
        ? cpu_urma_detail::PlanWriteCoalescedGroups(
              buffers, remote_offsets, count, batch_remote_sgl_size_,
              remote_region_.length, &batch_groups_, max_message_bytes_)
        : cpu_urma_detail::ValidateBatchRows(
              buffers, remote_offsets, count, remote_region_.length);
    if (!plan_status.ok()) return plan_status;
    CpuUrmaTransport::BatchTiming total_timing{};
    total_timing.setup_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - plan_start).count());
    const std::size_t wr_count = coalesce ? batch_groups_.size() : count;
    for (std::size_t begin = 0U; begin < wr_count; begin += batch_capacity) {
      const std::size_t chunk = std::min(batch_capacity, wr_count - begin);
      const std::size_t first_row = coalesce ? batch_groups_[begin].first : begin;
      const std::size_t row_count = coalesce
          ? batch_groups_[begin + chunk - 1U].first +
                batch_groups_[begin + chunk - 1U].count - first_row
          : chunk;
      const Status status = TransferSglBatchLocked(
          write, 0U, buffers + first_row, remote_offsets + first_row,
          row_count, coalesce ? batch_groups_.data() + begin : nullptr,
          coalesce ? chunk : 0U, coalesce ? batch_remote_sgl_size_ : 1U,
          true);
      if (!status.ok()) {
        last_batch_timing_ = {};
        return status;
      }
      total_timing.prepare_ns += last_batch_timing_.prepare_ns;
      total_timing.post_ns += last_batch_timing_.post_ns;
      total_timing.poll_ns += last_batch_timing_.poll_ns;
      total_timing.setup_ns += last_batch_timing_.setup_ns;
      total_timing.dispatch_ns += last_batch_timing_.dispatch_ns;
      total_timing.wr_count += last_batch_timing_.wr_count;
    }
    total_timing.total_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    last_batch_timing_ = total_timing;
    return Status::OK();
  }

  Status ValidateLink() {
    if (!initialized_) return Status::NotInitialized("CPU transport is not initialized");
    const std::size_t bytes = static_cast<std::size_t>(
        std::min<std::uint64_t>(64U, std::max<std::uint64_t>(
            1U, remote_region_.length / local_jettys_.size())));
    std::fill(probe_buffer_.begin(), probe_buffer_.end(), 0xA5U);
    std::fprintf(stderr, "[KVAF_CPU_URMA] LINK_PROBE stage=register_host\n");
    Status status = RegisterDirectHostMemory(probe_buffer_.data(), probe_buffer_.size());
    if (!status.ok()) return status;
    std::vector<std::uint64_t> offsets(local_jettys_.size());
    std::vector<CpuUrmaTransport::HostSge> sources(local_jettys_.size());
    for (std::size_t jetty = 0; jetty < local_jettys_.size(); ++jetty) {
      offsets[jetty] = remote_region_.length >= local_jettys_.size() ? jetty * bytes : 0U;
      // Reusing the source address prevents grouping: every Jetty is probed.
      sources[jetty] = {probe_buffer_.data(), bytes};
    }
    std::fprintf(stderr, "[KVAF_CPU_URMA] LINK_PROBE stage=transfer jettys=%zu bytes_per_wr=%zu\n",
                 sources.size(), bytes);
    status = TransferBatch(true, offsets.data(), sources.data(), sources.size());
    std::fprintf(stderr, "[KVAF_CPU_URMA] LINK_PROBE stage=done ok=%d\n", status.ok());
    last_batch_timing_ = {};
    return status;
  }

  Status Shutdown() {
    initialized_ = false;
    for (auto* remote_jetty : remote_jettys_) {
      if (remote_jetty != nullptr) (void)urma_unimport_jetty(remote_jetty);
    }
    remote_jettys_.clear();
    for (auto* local_jetty : local_jettys_) {
      if (local_jetty != nullptr) (void)urma_delete_jetty(local_jetty);
    }
    local_jettys_.clear();
    for (auto* jfr : jfrs_) {
      if (jfr != nullptr) (void)urma_delete_jfr(jfr);
    }
    jfrs_.clear();
    if (remote_seg_handle_ != nullptr) { (void)urma_unimport_seg(remote_seg_handle_); remote_seg_handle_ = nullptr; }
    for (auto& entry : direct_host_buffers_) {
      if (entry.segment != nullptr) (void)urma_unregister_seg(entry.segment);
      if (entry.token_id != nullptr) (void)urma_free_token_id(entry.token_id);
    }
    direct_host_buffers_.clear();
    explicit_host_region_ = {};
    if (owns_remote_region_ && link_ != nullptr && remote_region_.base != 0) {
      if (remote_region_.device_memory) (void)link_->ReleaseHbmRegion(remote_region_.session_id);
      else (void)link_->ReleaseRegion(remote_region_.base);
      remote_region_ = {};
    }
    owns_remote_region_ = false;
    for (auto* jfc : jfcs_) {
      if (jfc != nullptr) (void)urma_delete_jfc(jfc);
    }
    jfcs_.clear();
    if (context_ != nullptr) { (void)urma_delete_context(context_); context_ = nullptr; }
    if (urma_initialized_ && urma_owned_) { (void)urma_uninit(); }
    urma_initialized_ = false;
    urma_owned_ = false;
    link_.reset();
    return Status::OK();
  }

  std::unique_ptr<peer::PeerLink> link_;
  peer::PeerRegion remote_region_{};
  urma_device_t* device_{nullptr};
  urma_context_t* context_{nullptr};
  urma_target_seg_t* remote_seg_handle_{nullptr};
  urma_seg_t remote_seg_{};
  std::uint32_t eid_index_{0};
  std::uint64_t max_message_bytes_{std::numeric_limits<std::uint32_t>::max()};
  std::vector<urma_jfc_t*> jfcs_;
  std::vector<urma_jfr_t*> jfrs_;
  std::vector<urma_jetty_t*> local_jettys_;
  std::vector<urma_target_jetty_t*> remote_jettys_;
  std::mutex batch_submit_mutex_;
  SglBatchJob sgl_batch_workspace_;
  std::size_t wqes_per_doorbell_{0U};
  std::size_t batch_remote_sgl_size_{1U};
  std::vector<cpu_urma_detail::SglGroup> batch_groups_;
  CpuUrmaTransport::BatchTiming last_batch_timing_{};
  bool urma_initialized_{false};
  bool urma_owned_{false};
  // Registered memory must outlive all WRs and remain allocated until Shutdown.
  alignas(kPageBytes) std::array<std::uint8_t, kPageBytes> probe_buffer_{};
  bool initialized_{false};
  bool owns_remote_region_{false};
  std::vector<RegisteredHostBuffer> direct_host_buffers_;
  RegisteredHostBuffer explicit_host_region_{};
  GetTpList get_tp_list_{nullptr};
  ImportJettyEx import_jetty_ex_{nullptr};
};

CpuUrmaTransport::CpuUrmaTransport() : impl_(std::make_unique<Impl>()) {}
CpuUrmaTransport::~CpuUrmaTransport() = default;
Status CpuUrmaTransport::Initialize(const CpuUrmaTransportOptions& options) { return impl_->Initialize(options); }
Status CpuUrmaTransport::ValidateLink() { return impl_->ValidateLink(); }
Status CpuUrmaTransport::WriteSglBatch(std::uint64_t offset, const HostSge* sources,
                                       std::size_t count) {
  return impl_->TransferSglBatch(true, offset, sources, nullptr, count);
}
Status CpuUrmaTransport::WriteBatch(const std::uint64_t* remote_offsets,
                                    const HostSge* sources,
                                    std::size_t count) {
  return impl_->TransferBatch(true, remote_offsets, sources, count);
}
Status CpuUrmaTransport::RegisterDirectHostMemory(void* base, std::size_t bytes) {
  return impl_->RegisterDirectHostMemory(base, bytes);
}
Status CpuUrmaTransport::ReadSglBatch(std::uint64_t offset, const HostSge* destinations,
                                      std::size_t count) {
  return impl_->TransferSglBatch(false, offset, destinations, nullptr, count);
}
Status CpuUrmaTransport::ReadBatch(const std::uint64_t* remote_offsets,
                                   const HostSge* destinations,
                                   std::size_t count) {
  return impl_->TransferBatch(false, remote_offsets, destinations, count);
}
Status CpuUrmaTransport::Shutdown() { return impl_->Shutdown(); }
bool CpuUrmaTransport::initialized() const { return impl_->initialized_; }
std::uint64_t CpuUrmaTransport::region_bytes() const { return impl_->remote_region_.length; }
CpuUrmaTransport::BatchTiming CpuUrmaTransport::last_batch_timing() const {
  return impl_->last_batch_timing_;
}

}  // namespace kvaf
