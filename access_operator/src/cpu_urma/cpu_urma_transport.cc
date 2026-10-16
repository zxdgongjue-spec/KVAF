#include "cpu_urma/cpu_urma_transport.h"
#include "cpu_urma/batch_validation.h"
#include "cpu_urma/sgl_batch_builder.h"
#include "cpu_urma/jetty_batch_dispatch.h"
#include "cpu_urma/indexed_async_write_tracker.h"

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
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <urma_api.h>

#include "peer_link.h"

namespace kvaf {
namespace {

constexpr std::uint32_t kJfcDepth = 16384;
constexpr std::uint32_t kJettyDepth = 8192;
constexpr std::uint32_t kMaxJettyCount = 16;
constexpr std::uint32_t kMaxPrepareThreadCount = 64;
constexpr std::uint32_t kPageBytes = 4096;
constexpr std::size_t kMinRowsPerPrepareWorker = 64;

struct PreparedQueueRows { std::size_t first{}, count{}; };
inline PreparedQueueRows PreparedRowsForQueue(std::size_t rows, std::size_t queues,
    std::size_t queue) {
  if (!queues || queue >= queues) return {};
  const auto base = rows / queues, remainder = rows % queues;
  return {queue * base + std::min(queue, remainder), base + (queue < remainder ? 1U : 0U)};
}
constexpr std::size_t kParallelPrepareMinItems = 256;
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
    struct MissingHostSegment {
      std::size_t row{0U};
      std::size_t local_index{0U};
    };

    // WRITE WRs gather one or more local SGEs into one contiguous remote SGE.
    std::vector<SglSgeAbi> local;
    std::vector<SglSgeAbi> remote;
    std::vector<urma_jfs_wr_t> wrs;
    std::vector<MissingHostSegment> missing_host_segments;
    // Storage is retained at its high-water mark. Only the leading count
    // entries belong to the current batch.
    std::size_t count{0U};
  };

  struct SglBatchJob {
    bool write{false};
    std::uint64_t generation{0U};
    const CpuUrmaTransport::HostSge* buffers{nullptr};
    urma_target_seg_t* single_host_segment{nullptr};
    std::vector<SglBatchQueue> queues;
    const std::uint64_t* remote_offsets{nullptr};
    std::vector<std::uint64_t> generated_remote_offsets;
    std::vector<urma_target_seg_t*> host_segments;
    std::chrono::steady_clock::time_point deadline{};
  };

  struct WorkerResult {
    std::string post_error;
    std::string poll_error;
  };

  struct PreparedWrite {
    SglBatchJob job;
    std::array<cpu_urma_detail::IndexedAsyncWriteTracker::Fence, kMaxJettyCount> fences{};
    std::size_t fence_count{};
    cpu_urma_detail::IndexedAsyncWriteTracker::Handle handle;
    std::array<std::size_t, kMaxJettyCount> first_prefix{};
    std::uint64_t marker_wrs{};
    bool ordered_chain{};
  };

  static std::uint64_t MakeUserContext(std::uint64_t generation,
                                       std::size_t jetty,
                                       std::size_t wr_index) {
    constexpr std::uint64_t kGenerationMask = (1ULL << 39U) - 1U;
    return 0x5A00000000000000ULL |
        ((generation & kGenerationMask) << 17U) |
        ((static_cast<std::uint64_t>(jetty) & 0xFU) << 13U) |
        (static_cast<std::uint64_t>(wr_index) & 0x1FFFU);
  }

  static void EnsureSglQueueStorage(SglBatchQueue& queue,
                                    std::size_t wr_count,
                                    std::size_t local_sge_count) {
    if (queue.local.size() < local_sge_count) {
      queue.local.resize(local_sge_count);
    }
    if (queue.remote.size() < wr_count) queue.remote.resize(wr_count);
    if (queue.wrs.size() < wr_count) queue.wrs.resize(wr_count);
    queue.count = wr_count;
  }

  std::size_t PrepareWorkerCount(std::size_t item_count) const {
    if (item_count < kParallelPrepareMinItems) return 1U;
    const std::size_t useful_workers =
        (item_count + kMinRowsPerPrepareWorker - 1U) /
        kMinRowsPerPrepareWorker;
    return std::max<std::size_t>(
        1U, std::min({prepare_thread_count_, useful_workers, item_count}));
  }

  void PrepareSglBatchEntry(SglBatchJob& job, std::size_t wr_index) {
    const std::size_t jetty = wr_index % local_jettys_.size();
    const std::size_t position = wr_index / local_jettys_.size();
    const std::size_t first = wr_index;
    SglBatchQueue& queue = job.queues[jetty];
    SglSgeAbi* local = &queue.local[position];
    const auto& buffer = job.buffers[first];
    *local = {};
    local->addr = reinterpret_cast<std::uint64_t>(buffer.data);
    local->len = static_cast<std::uint32_t>(buffer.bytes);
    local->tseg = job.single_host_segment != nullptr
        ? job.single_host_segment : job.host_segments[first];
    SglSgeAbi* remote = &queue.remote[position];
    *remote = {};
    remote->addr = remote_region_.base + job.remote_offsets[first];
    remote->len = static_cast<std::uint32_t>(buffer.bytes);
    remote->tseg = remote_seg_handle_;
    urma_jfs_wr_t& wr = queue.wrs[position];
    wr = {};
    wr.opcode = job.write ? URMA_OPC_WRITE : URMA_OPC_READ;
    const bool completes_queue = position + 1U == queue.count;
    // The final CQE is the completion fence for the entire Jetty chain.
    // Completion ordering is required for both WRITE and READ, especially
    // when one WRITE gathers several local source SGEs.
    wr.flag.bs.place_order =
        completes_queue ? URMA_STRONG_ORDER : URMA_RELAX_ORDER;
    wr.flag.bs.comp_order = 1U;
    wr.flag.bs.complete_enable = completes_queue ? 1U : 0U;
    wr.tjetty = remote_jettys_[jetty];
    wr.user_ctx = MakeUserContext(job.generation, jetty, position);
    urma_sg_t local_sg{};
    local_sg.sge = reinterpret_cast<urma_sge_t*>(local);
    local_sg.num_sge = 1U;
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
    wr.next = position + 1U < queue.count
                  ? &queue.wrs[position + 1U]
                  : nullptr;
  }

  std::string PostSglQueue(SglBatchJob& job, std::size_t jetty) {
    SglBatchQueue& queue = job.queues[jetty];
    const std::size_t chunk_size = wqes_per_doorbell_ == 0U
        ? kJettyDepth : wqes_per_doorbell_;
    std::string error;
    cpu_urma_detail::DispatchJettyChunks(
        queue.count, chunk_size,
        [&](std::size_t, std::size_t end) {
          queue.wrs[end - 1U].next = nullptr;
        },
        [&](std::size_t begin, std::size_t) {
          urma_jfs_wr_t* bad = nullptr;
          const auto status = urma_post_jetty_send_wr(
              local_jettys_[jetty], &queue.wrs[begin], &bad);
          if (status == URMA_SUCCESS) return true;
          error = "SGL batch post failed (jetty=" +
                  std::to_string(jetty) + ", status=" +
                  std::to_string(static_cast<int>(status)) +
                  ", bad_wr=" +
                  std::to_string(reinterpret_cast<std::uintptr_t>(bad)) +
                  ")";
          return false;
        });
    return error;
  }

  std::string PollSglQueue(const SglBatchJob& job, std::size_t jetty) {
    const SglBatchQueue& queue = job.queues[jetty];
    if (queue.count == 0U) return {};
    const std::uint64_t expected_ctx = queue.wrs[queue.count - 1U].user_ctx;
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
      return {};
    }
    return "timed out waiting for SGL batch completion (jetty=" +
           std::to_string(jetty) + ")";
  }

  std::string PreparePostAndPollSglBatch(SglBatchJob& job,
                                         std::uint64_t* prepare_ns,
                                         std::uint64_t* post_ns,
                                         std::uint64_t* poll_ns) {
    std::array<WorkerResult, kMaxJettyCount> results{};
    std::size_t wr_count = 0U;
    for (const auto& queue : job.queues) wr_count += queue.count;
    // Single-queue control messages bypass the OpenMP team.
    std::size_t active = 0, sole = 0;
    for (std::size_t q = 0; q < job.queues.size(); ++q)
      if (job.queues[q].count) { ++active; sole = q; }
    if (active == 1) {
      const auto begin = std::chrono::steady_clock::now();
      for (std::size_t wr = 0; wr < wr_count; ++wr) PrepareSglBatchEntry(job, wr);
      const auto prepared = std::chrono::steady_clock::now();
      auto error = PostSglQueue(job, sole);
      const auto posted = std::chrono::steady_clock::now();
      if (error.empty()) error = PollSglQueue(job, sole);
      const auto finished = std::chrono::steady_clock::now();
      *prepare_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(prepared - begin).count();
      *post_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(posted - prepared).count();
      *poll_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(finished - posted).count();
      if (!error.empty()) batch_poisoned_ = true;
      return error;
    }
    const std::size_t prepare_tasks = PrepareWorkerCount(wr_count);
    const std::size_t team_threads =
        std::max(prepare_tasks, job.queues.size());
    (void)team_threads;
    const auto prepare_start = std::chrono::steady_clock::now();
    auto prepare_end = prepare_start;
    auto post_end = prepare_start;
    auto poll_end = prepare_start;

    // Preparation workers only write disjoint WR/SGL slots. After the barrier,
    // exactly one owner posts each Jetty and polls its matching JFC.
#pragma omp parallel num_threads(static_cast<int>(team_threads))
    {
#pragma omp for schedule(static, 1)
      for (std::int64_t task = 0;
           task < static_cast<std::int64_t>(prepare_tasks); ++task) {
        const std::size_t begin =
            wr_count * static_cast<std::size_t>(task) / prepare_tasks;
        const std::size_t end =
            wr_count * (static_cast<std::size_t>(task) + 1U) / prepare_tasks;
        for (std::size_t wr = begin; wr < end; ++wr) {
          PrepareSglBatchEntry(job, wr);
        }
      }
#pragma omp single
      { prepare_end = std::chrono::steady_clock::now(); }

#pragma omp for schedule(static, 1)
      for (std::int64_t index = 0;
           index < static_cast<std::int64_t>(job.queues.size()); ++index) {
        const auto jetty = static_cast<std::size_t>(index);
        results[jetty].post_error = PostSglQueue(job, jetty);
      }
#pragma omp single
      { post_end = std::chrono::steady_clock::now(); }

#pragma omp for schedule(static, 1)
      for (std::int64_t index = 0;
           index < static_cast<std::int64_t>(job.queues.size()); ++index) {
        const auto jetty = static_cast<std::size_t>(index);
        if (results[jetty].post_error.empty()) {
          results[jetty].poll_error = PollSglQueue(job, jetty);
        }
      }
#pragma omp single
      { poll_end = std::chrono::steady_clock::now(); }
    }

    *prepare_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            prepare_end - prepare_start).count());
    *post_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            post_end - prepare_end).count());
    *poll_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            poll_end - post_end).count());
    for (std::size_t jetty = 0U; jetty < job.queues.size(); ++jetty) {
      if (!results[jetty].post_error.empty()) {
        batch_poisoned_ = true;
        return results[jetty].post_error;
      }
    }
    for (std::size_t jetty = 0U; jetty < job.queues.size(); ++jetty) {
      if (!results[jetty].poll_error.empty()) {
        batch_poisoned_ = true;
        return results[jetty].poll_error;
      }
    }
    return {};
  }

  ~Impl() { (void)Shutdown(); }

  Status Initialize(const CpuUrmaTransportOptions& options) {
    if (options.peer_address.empty() || options.region_bytes == 0 ||
        options.region_bytes > std::numeric_limits<std::size_t>::max() ||
        options.wqes_per_doorbell > kJettyDepth || options.remote_sgl_size == 0U ||
        options.jetty_count == 0U || options.jetty_count > kMaxJettyCount ||
        options.prepare_thread_count == 0U ||
        options.prepare_thread_count > kMaxPrepareThreadCount ||
        !options.async_cq_batch || options.async_cq_batch > 64 ||
        !options.async_deadline_polls || options.async_deadline_polls > 1024) {
      return Status::InvalidArgument(
          "CPU URMA transport requires peer_address, non-zero region_bytes, "
          "non-zero remote_sgl_size, jetty_count in [1,16], "
          "prepare_thread_count in [1,64] and "
          "wqes_per_doorbell in [0,8192], async_cq_batch in [1,64] and "
          "async_deadline_polls in [1,1024]");
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
    max_local_sge_ = 1U;
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
      if (device_attr.dev_cap.max_jfs_sge != 0U) {
        max_local_sge_ = std::min<std::uint32_t>(
            std::numeric_limits<std::uint8_t>::max(),
            device_attr.dev_cap.max_jfs_sge);
      }
    }
    // remote_sgl_size is the retained public option name. It controls the
    // number of distinct local source SGEs in each WRITE WR.
    batch_local_sgl_size_ = std::min<std::size_t>(
        options.remote_sgl_size, max_local_sge_);
    prepare_thread_count_ = options.prepare_thread_count;
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
    if (region.base > std::numeric_limits<std::uint64_t>::max() -
                          region.length) {
      return Status::InvalidArgument("remote region address range overflows");
    }
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
      jetty_cfg.jfs_cfg.max_sge =
          static_cast<std::uint8_t>(batch_local_sgl_size_);
      // RC WRITE supports multiple local source SGEs but one remote destination.
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
      std::memcpy(query.peer_eid, region.RoutingEid().data(), 16);
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
    std::fprintf(stderr, "[KVAF_CPU_URMA] JETTYS_READY count=%zu jetty_owners=%zu prepare_threads=%zu write_team_threads=%zu wqes_per_doorbell=%zu requested_remote_sgl_size=%u local_sgl_size=%zu remote_sge=1 max_wr_bytes=%llu depth=%u\n",
                 local_jettys_.size(), local_jettys_.size(),
                 prepare_thread_count_,
                 std::min(prepare_thread_count_, local_jettys_.size()),
                 wqes_per_doorbell_,
                 options.remote_sgl_size, batch_local_sgl_size_,
                 static_cast<unsigned long long>(max_message_bytes_), kJettyDepth);
    batch_generation_ = 0U;
    batch_poisoned_ = false;
    indexed_tracker_ = cpu_urma_detail::IndexedAsyncWriteTracker(local_jettys_.size(), kJettyDepth);
    async_cq_batch_ = options.async_cq_batch;
    async_deadline_polls_ = options.async_deadline_polls;
    async_polls_since_deadline_ = next_async_poll_queue_ = 0;
    async_stats_ = {};
    initialized_ = true;
    return Status::OK();
  }

  struct RegisteredHostBuffer {
    const void* address{nullptr};
    std::size_t bytes{0};
    urma_token_id_t* token_id{nullptr};
    urma_target_seg_t* segment{nullptr};
  };

  static bool HostSpanFitsRegisteredBuffer(
      const RegisteredHostBuffer& region, const void* address,
      std::size_t bytes) {
    if (region.segment == nullptr || address == nullptr || bytes == 0U) {
      return false;
    }
    const auto requested = reinterpret_cast<std::uintptr_t>(address);
    const auto base = reinterpret_cast<std::uintptr_t>(region.address);
    return requested >= base && bytes <= region.bytes &&
        requested - base <= region.bytes - bytes;
  }

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
    std::unique_lock<std::mutex> submit_lock(batch_submit_mutex_);
    if (!initialized_) {
      return Status::InvalidState("CPU URMA transport is not initialized");
    }
    if (batch_poisoned_) {
      return Status::InvalidState(
          "CPU URMA transport requires reinitialization after a submission failure");
    }
    if (!indexed_tracker_.Empty())
      return Status::InvalidState("host registration while prepared WRITEs are in flight");
    RegisteredHostBuffer* host = nullptr;
    Status status = RegisterHostBuffer(base, bytes, &host);
    if (status.ok()) explicit_host_region_ = *host;
    return status;
  }

  Status ValidateAndRegisterHostBuffers(
      const CpuUrmaTransport::HostSge* buffers,
      const std::uint64_t* remote_offsets, std::size_t count) {
    for (std::size_t row = 0U; row < count; ++row) {
      const auto& buffer = buffers[row];
      const auto address = reinterpret_cast<std::uintptr_t>(buffer.data);
      const auto remote_offset = remote_offsets[row];
      if (address == 0U || buffer.bytes == 0U ||
          buffer.bytes > max_message_bytes_ ||
          buffer.bytes > std::numeric_limits<std::uint32_t>::max() ||
          buffer.bytes > std::numeric_limits<std::uintptr_t>::max() - address ||
          remote_offset > remote_region_.length ||
          buffer.bytes > remote_region_.length - remote_offset ||
          remote_region_.base >
              std::numeric_limits<std::uint64_t>::max() - remote_offset) {
        return Status::InvalidArgument("invalid WRITE gather row");
      }
      if (HostSpanFitsRegisteredBuffer(
              explicit_host_region_, buffer.data, buffer.bytes) ||
          FindRegisteredHostBuffer(buffer.data, buffer.bytes) != nullptr) {
        continue;
      }
      RegisteredHostBuffer* host = nullptr;
      const Status status = RegisterHostBuffer(
          buffer.data, buffer.bytes, &host);
      if (!status.ok()) return status;
    }
    return Status::OK();
  }

  Status BuildWriteGatherQueue(
      SglBatchJob& job, std::size_t jetty,
      const CpuUrmaTransport::HostSge* buffers,
      const std::uint64_t* remote_offsets, std::size_t row_count,
      std::size_t first, std::size_t count) {
    SglBatchQueue& queue = job.queues[jetty];
    queue.count = 0U;
    queue.missing_host_segments.clear();
    if (count == 0U) return Status::OK();

    std::size_t wr_count = 0U;
    const Status status = cpu_urma_detail::BuildWriteGatherShard(
        buffers, remote_offsets, row_count, first, count,
        batch_local_sgl_size_, remote_region_.length, max_message_bytes_,
        [&](std::size_t local_index, std::size_t row,
            const CpuUrmaTransport::HostSge& buffer) {
          SglSgeAbi& local = queue.local[local_index];
          local = {};
          local.addr = reinterpret_cast<std::uint64_t>(buffer.data);
          local.len = static_cast<std::uint32_t>(buffer.bytes);

          if (HostSpanFitsRegisteredBuffer(
                  explicit_host_region_, buffer.data, buffer.bytes)) {
            local.tseg = explicit_host_region_.segment;
          } else if (RegisteredHostBuffer* host =
                         FindRegisteredHostBuffer(buffer.data, buffer.bytes);
                     host != nullptr) {
            local.tseg = host->segment;
          } else {
            queue.missing_host_segments.push_back({row, local_index});
          }
        },
        [&](std::size_t wr_index,
            const cpu_urma_detail::WriteGatherGroup& group) {
          SglSgeAbi& remote = queue.remote[wr_index];
          remote = {};
          remote.addr = remote_region_.base + remote_offsets[group.first];
          remote.len = static_cast<std::uint32_t>(group.bytes);
          remote.tseg = remote_seg_handle_;

          urma_jfs_wr_t& wr = queue.wrs[wr_index];
          wr = {};
          wr.opcode = URMA_OPC_WRITE;
          wr.flag.bs.place_order = URMA_RELAX_ORDER;
          wr.flag.bs.comp_order = 1U;
          wr.flag.bs.complete_enable = 0U;
          wr.tjetty = remote_jettys_[jetty];
          wr.user_ctx = MakeUserContext(
              job.generation, jetty, wr_index);

          const std::size_t local_first = group.first - first;
          urma_sg_t local_sg{};
          local_sg.sge = reinterpret_cast<urma_sge_t*>(
              queue.local.data() + local_first);
          local_sg.num_sge = static_cast<std::uint32_t>(group.count);
          urma_sg_t remote_sg{};
          remote_sg.sge = reinterpret_cast<urma_sge_t*>(&remote);
          remote_sg.num_sge = 1U;
          wr.rw.src = local_sg;
          wr.rw.dst = remote_sg;
          wr.next = nullptr;
          if (wr_index != 0U) {
            queue.wrs[wr_index - 1U].next = &wr;
          }
        },
        &wr_count);
    if (!status.ok()) {
      queue.missing_host_segments.clear();
      return status;
    }

    queue.count = wr_count;
    urma_jfs_wr_t& tail = queue.wrs[wr_count - 1U];
    tail.flag.bs.place_order = URMA_STRONG_ORDER;
    tail.flag.bs.complete_enable = 1U;
    return Status::OK();
  }

  Status BuildPostAndPollWriteWave(
      const CpuUrmaTransport::HostSge* buffers,
      const std::uint64_t* remote_offsets, std::size_t row_count,
      std::size_t first, std::size_t count,
      CpuUrmaTransport::BatchTiming* timing) {
    const auto setup_start = std::chrono::steady_clock::now();
    SglBatchJob& job = sgl_batch_workspace_;
    job.write = true;
    job.generation = ++batch_generation_;
    job.queues.resize(local_jettys_.size());

    const std::size_t base = count / job.queues.size();
    const std::size_t remainder = count % job.queues.size();
    for (std::size_t jetty = 0U; jetty < job.queues.size(); ++jetty) {
      const std::size_t shard_count =
          base + (jetty < remainder ? 1U : 0U);
      EnsureSglQueueStorage(job.queues[jetty], shard_count, shard_count);
      job.queues[jetty].count = 0U;
      job.queues[jetty].missing_host_segments.clear();
    }

    const auto prepare_start = std::chrono::steady_clock::now();
    timing->setup_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            prepare_start - setup_start).count());
    std::array<Status, kMaxJettyCount> build_status{};
    std::array<WorkerResult, kMaxJettyCount> results{};
    Status shared_status = Status::OK();
    auto prepare_end = prepare_start;
    auto post_end = prepare_start;
    auto poll_end = prepare_start;
    std::array<std::chrono::steady_clock::time_point, kMaxJettyCount>
        post_started{};
    std::array<std::chrono::steady_clock::time_point, kMaxJettyCount>
        post_finished{};
    std::array<std::chrono::steady_clock::time_point, kMaxJettyCount>
        poll_finished{};

    const std::size_t team_threads = std::min(
        prepare_thread_count_, local_jettys_.size());
    (void)team_threads;
#pragma omp parallel num_threads(static_cast<int>(team_threads))
    {
#pragma omp for schedule(static, 1)
      for (std::int64_t index = 0;
           index < static_cast<std::int64_t>(job.queues.size()); ++index) {
        const std::size_t jetty = static_cast<std::size_t>(index);
        const std::size_t shard_count =
            base + (jetty < remainder ? 1U : 0U);
        const std::size_t shard_first = first + jetty * base +
            std::min(jetty, remainder);
        build_status[jetty] = shard_count == 0U
            ? Status::OK()
            : BuildWriteGatherQueue(job, jetty, buffers, remote_offsets,
                                    row_count, shard_first, shard_count);
      }

#pragma omp master
      {
        for (std::size_t jetty = 0U; jetty < job.queues.size(); ++jetty) {
          if (!build_status[jetty].ok()) {
            shared_status = build_status[jetty];
            break;
          }
        }
        if (shared_status.ok()) {
          for (std::size_t jetty = 0U;
               jetty < job.queues.size() && shared_status.ok(); ++jetty) {
            SglBatchQueue& queue = job.queues[jetty];
            for (const auto& missing : queue.missing_host_segments) {
              RegisteredHostBuffer* host = nullptr;
              const auto& buffer = buffers[missing.row];
              shared_status = RegisterHostBuffer(
                  buffer.data, buffer.bytes, &host);
              if (!shared_status.ok()) break;
              queue.local[missing.local_index].tseg = host->segment;
            }
          }
        }
        job.deadline = std::chrono::steady_clock::now() +
            std::chrono::seconds(5);
      }
#pragma omp barrier

#pragma omp for schedule(static, 1) nowait
      for (std::int64_t index = 0;
           index < static_cast<std::int64_t>(job.queues.size()); ++index) {
        const auto jetty = static_cast<std::size_t>(index);
        post_started[jetty] = std::chrono::steady_clock::now();
        if (shared_status.ok()) {
          results[jetty].post_error = PostSglQueue(job, jetty);
        }
        post_finished[jetty] = std::chrono::steady_clock::now();
      }

#pragma omp for schedule(static, 1)
      for (std::int64_t index = 0;
           index < static_cast<std::int64_t>(job.queues.size()); ++index) {
        const auto jetty = static_cast<std::size_t>(index);
        if (shared_status.ok() && results[jetty].post_error.empty()) {
          results[jetty].poll_error = PollSglQueue(job, jetty);
        }
        poll_finished[jetty] = std::chrono::steady_clock::now();
      }
#pragma omp master
      {
        prepare_end = *std::min_element(
            post_started.begin(),
            post_started.begin() + job.queues.size());
        post_end = *std::max_element(
            post_finished.begin(),
            post_finished.begin() + job.queues.size());
        poll_end = *std::max_element(
            poll_finished.begin(),
            poll_finished.begin() + job.queues.size());
      }
    }

    timing->prepare_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            prepare_end - prepare_start).count());
    timing->post_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            post_end - prepare_end).count());
    timing->poll_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            poll_end - post_end).count());
    timing->dispatch_ns = timing->prepare_ns + timing->post_ns;
    for (const auto& queue : job.queues) timing->wr_count += queue.count;

    if (!shared_status.ok()) return shared_status;
    for (std::size_t jetty = 0U; jetty < job.queues.size(); ++jetty) {
      if (!results[jetty].post_error.empty()) {
        batch_poisoned_ = true;
        return Status::Internal(results[jetty].post_error);
      }
    }
    for (std::size_t jetty = 0U; jetty < job.queues.size(); ++jetty) {
      if (!results[jetty].poll_error.empty()) {
        batch_poisoned_ = true;
        return Status::Internal(results[jetty].poll_error);
      }
    }
    return Status::OK();
  }

  Status TransferWriteBatchLocked(
      const std::uint64_t* remote_offsets,
      const CpuUrmaTransport::HostSge* buffers, std::size_t count) {
    if (!initialized_) {
      return Status::InvalidState("CPU URMA transport is not initialized");
    }
    if (batch_poisoned_) {
      return Status::InvalidState(
          "CPU URMA transport requires reinitialization after a submission failure");
    }
    if (!indexed_tracker_.Empty())
      return Status::InvalidState("synchronous transfer while prepared WRITEs are in flight");
    if (remote_offsets == nullptr || buffers == nullptr || count == 0U) {
      return Status::InvalidArgument("invalid non-contiguous batch transfer");
    }

    const auto start = std::chrono::steady_clock::now();
    CpuUrmaTransport::BatchTiming total_timing{};
    const std::size_t wave_rows = kJettyDepth * local_jettys_.size();
    if (count > wave_rows) {
      const auto validate_start = std::chrono::steady_clock::now();
      const Status status = ValidateAndRegisterHostBuffers(
          buffers, remote_offsets, count);
      total_timing.plan_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - validate_start).count());
      if (!status.ok()) return status;
    }

    for (std::size_t first = 0U; first < count;) {
      const std::size_t wave_count = std::min(wave_rows, count - first);
      CpuUrmaTransport::BatchTiming wave_timing{};
      const Status status = BuildPostAndPollWriteWave(
          buffers, remote_offsets, count, first, wave_count, &wave_timing);
      if (!status.ok()) return status;
      total_timing.prepare_ns += wave_timing.prepare_ns;
      total_timing.post_ns += wave_timing.post_ns;
      total_timing.poll_ns += wave_timing.poll_ns;
      total_timing.setup_ns += wave_timing.setup_ns;
      total_timing.dispatch_ns += wave_timing.dispatch_ns;
      total_timing.wr_count += wave_timing.wr_count;
      first += wave_count;
    }
    total_timing.total_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
    last_batch_timing_ = total_timing;
    return Status::OK();
  }

  Status TransferSglBatchLocked(bool write, std::uint64_t offset,
                                const CpuUrmaTransport::HostSge* buffers,
                                const std::uint64_t* remote_offsets,
                                std::size_t count,
                                bool rows_validated = false) {
    const auto setup_start = std::chrono::steady_clock::now();
    if (!initialized_) return Status::InvalidState("CPU URMA transport is not initialized");
    if (batch_poisoned_) {
      return Status::InvalidState(
          "CPU URMA transport requires reinitialization after a submission failure");
    }
    if (!indexed_tracker_.Empty())
      return Status::InvalidState("synchronous transfer while prepared WRITEs are in flight");
    if (buffers == nullptr || count == 0U ||
        offset > remote_region_.length) {
      return Status::InvalidArgument("invalid SGL batch transfer");
    }
    const std::size_t wr_count = count;
    if (wr_count == 0U || wr_count > kJettyDepth * local_jettys_.size()) {
      return Status::InvalidArgument("SGL batch exceeds Jetty queue capacity");
    }
    SglBatchJob& job = sgl_batch_workspace_;
    job.write = write;
    job.generation = ++batch_generation_;
    job.buffers = buffers;
    job.single_host_segment = nullptr;
    job.queues.resize(local_jettys_.size());
    for (std::size_t jetty = 0U; jetty < local_jettys_.size(); ++jetty) {
      const std::size_t queue_count = wr_count / local_jettys_.size() +
          (jetty < wr_count % local_jettys_.size() ? 1U : 0U);
      EnsureSglQueueStorage(
          job.queues[jetty], queue_count, queue_count);
    }
    const bool generate_remote_offsets = remote_offsets == nullptr;
    if (generate_remote_offsets) job.generated_remote_offsets.resize(count);
    // Explicit offsets remain valid for this synchronous transfer, so avoid
    // copying the caller's array on every batch.
    job.remote_offsets = generate_remote_offsets
        ? job.generated_remote_offsets.data() : remote_offsets;
    std::uint64_t total = 0U;
    bool use_explicit_host_region = explicit_host_region_.segment != nullptr;
    const auto host_span_is_registered = [&](const void* data, std::size_t bytes) {
      const auto requested = reinterpret_cast<std::uintptr_t>(data);
      const auto base = reinterpret_cast<std::uintptr_t>(explicit_host_region_.address);
      return requested >= base && bytes <= explicit_host_region_.bytes &&
          requested - base <= explicit_host_region_.bytes - bytes;
    };
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
          buffer.bytes > max_message_bytes_ ||
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
    if (use_explicit_host_region) {
      job.single_host_segment = explicit_host_region_.segment;
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
    std::uint64_t poll_ns = 0U;
    const auto dispatch_start = std::chrono::steady_clock::now();
    const auto setup_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            dispatch_start - setup_start).count());
    const auto dispatch_error = PreparePostAndPollSglBatch(
        job, &prepare_ns, &post_ns, &poll_ns);
    const std::uint64_t dispatch_ns = prepare_ns + post_ns;
    last_batch_timing_.prepare_ns = prepare_ns;
    last_batch_timing_.post_ns = post_ns;
    last_batch_timing_.poll_ns = poll_ns;
    last_batch_timing_.setup_ns = setup_ns;
    last_batch_timing_.dispatch_ns = dispatch_ns;
    last_batch_timing_.wr_count = wr_count;
    if (!dispatch_error.empty()) return Status::Internal(dispatch_error);
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
    std::unique_lock<std::mutex> submit_lock(batch_submit_mutex_);
    if (!initialized_) {
      return Status::InvalidState("CPU URMA transport is not initialized");
    }
    if (!indexed_tracker_.Empty())
      return Status::InvalidState("synchronous transfer while prepared WRITEs are in flight");
    if (remote_offsets == nullptr || buffers == nullptr || count == 0U) {
      return Status::InvalidArgument("invalid non-contiguous batch transfer");
    }
    last_batch_timing_ = {};
    if (write) {
      const Status status = TransferWriteBatchLocked(
          remote_offsets, buffers, count);
      if (!status.ok()) last_batch_timing_ = {};
      return status;
    }

    const auto start = std::chrono::steady_clock::now();
    const std::size_t batch_capacity = kJettyDepth * local_jettys_.size();
    const auto plan_start = std::chrono::steady_clock::now();
    const std::size_t planner_count = PrepareWorkerCount(count);
    Status plan_status = Status::OK();
    if (planner_count == 1U) {
      plan_status = cpu_urma_detail::ValidateBatchRows(
          buffers, remote_offsets, count, remote_region_.length,
          max_message_bytes_);
    } else {
      plan_status = cpu_urma_detail::PrepareBatchValidationSlices(
          count, planner_count, &batch_validation_slices_);
      if (plan_status.ok()) {
#pragma omp parallel for schedule(static, 1) num_threads(static_cast<int>(planner_count))
        for (std::int64_t index = 0;
             index < static_cast<std::int64_t>(batch_validation_slices_.size());
             ++index) {
          auto& slice =
              batch_validation_slices_[static_cast<std::size_t>(index)];
          slice.status = cpu_urma_detail::ValidateBatchRows(
              buffers + slice.first, remote_offsets + slice.first,
              slice.count, remote_region_.length, max_message_bytes_);
        }
        for (const auto& slice : batch_validation_slices_) {
          if (!slice.status.ok()) {
            plan_status = slice.status;
            break;
          }
        }
      }
    }
    if (!plan_status.ok()) return plan_status;
    CpuUrmaTransport::BatchTiming total_timing{};
    total_timing.plan_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - plan_start).count());
    const std::size_t wr_count = count;
    for (std::size_t begin = 0U; begin < wr_count; begin += batch_capacity) {
      const std::size_t chunk = std::min(batch_capacity, wr_count - begin);
      const Status status = TransferSglBatchLocked(
          false, 0U, buffers + begin, remote_offsets + begin,
          chunk, true);
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
    std::unique_lock<std::mutex> submit_lock(batch_submit_mutex_);
    if (!initialized_) return Status::NotInitialized("CPU transport is not initialized");
    if (!indexed_tracker_.Empty())
      return Status::InvalidState("link validation while prepared WRITEs are in flight");
    if (batch_poisoned_) {
      return Status::InvalidState(
          "CPU URMA transport requires reinitialization after a submission failure");
    }
    const std::size_t bytes = static_cast<std::size_t>(
        std::min<std::uint64_t>(64U, std::max<std::uint64_t>(
            1U, remote_region_.length / local_jettys_.size())));
    std::fill(probe_buffer_.begin(), probe_buffer_.end(), 0xA5U);
    std::fprintf(stderr, "[KVAF_CPU_URMA] LINK_PROBE stage=register_host\n");
    RegisteredHostBuffer* probe_host = nullptr;
    Status status = RegisterHostBuffer(
        probe_buffer_.data(), probe_buffer_.size(), &probe_host);
    if (!status.ok()) return status;
    explicit_host_region_ = *probe_host;
    std::vector<std::uint64_t> offsets(local_jettys_.size());
    std::vector<CpuUrmaTransport::HostSge> sources(local_jettys_.size());
    for (std::size_t jetty = 0; jetty < local_jettys_.size(); ++jetty) {
      offsets[jetty] = remote_region_.length >= local_jettys_.size() ? jetty * bytes : 0U;
      // The probe uses one WR per row so round-robin assignment reaches every
      // Jetty regardless of the configured gather size.
      sources[jetty] = {probe_buffer_.data(), bytes};
    }
    std::fprintf(stderr, "[KVAF_CPU_URMA] LINK_PROBE stage=transfer jettys=%zu bytes_per_wr=%zu\n",
                 sources.size(), bytes);
    status = TransferSglBatchLocked(
        true, 0U, sources.data(), offsets.data(), sources.size());
    std::fprintf(stderr, "[KVAF_CPU_URMA] LINK_PROBE stage=done ok=%d\n", status.ok());
    last_batch_timing_ = {};
    return status;
  }

  Status PrepareWriteBatch(const std::uint64_t* offsets,
                           const CpuUrmaTransport::HostSge* sources,
                           std::size_t count, CpuUrmaTransport::PreparedWriteId* id) {
    std::unique_lock<std::mutex> lock(batch_submit_mutex_);
    if (!initialized_ || batch_poisoned_ || !indexed_tracker_.Empty())
      return Status::InvalidState("prepare WRITE requires an initialized, idle transport");
    if (!id || !offsets || !sources || !count ||
        count > kJettyDepth * local_jettys_.size())
      return Status::InvalidArgument("prepared WRITE must fit one Jetty wave");
    auto status = ValidateAndRegisterHostBuffers(sources, offsets, count);
    if (!status.ok()) return status;
    auto plan = std::make_unique<PreparedWrite>();
    auto& job = plan->job;
    job.write = true;
    job.queues.resize(local_jettys_.size());
    for (std::size_t q = 0; q < job.queues.size(); ++q) {
      const auto range = PreparedRowsForQueue(count, job.queues.size(), q);
      const auto rows = range.count;
      EnsureSglQueueStorage(job.queues[q], rows, rows);
      if (!rows) continue;
      status = BuildWriteGatherQueue(job, q, sources, offsets, count, range.first, rows);
      if (!status.ok()) return status;
      if (!job.queues[q].missing_host_segments.empty())
        return Status::Internal("prepared WRITE has an unresolved host segment");
      plan->fences[plan->fence_count++] = {q, job.queues[q].count, 0};
    }
    prepared_writes_.push_back(std::move(plan));
    *id = prepared_writes_.size();
    return Status::OK();
  }

  Status PrepareWriteChain(const CpuUrmaTransport::WriteChainChunk* chunks, std::size_t count,
      CpuUrmaTransport::PreparedWriteId* id) {
    std::unique_lock<std::mutex> lock(batch_submit_mutex_);
    if (!initialized_ || batch_poisoned_ || !indexed_tracker_.Empty())
      return Status::InvalidState("prepare chain requires an initialized, idle transport");
    if (!chunks || !count || !id || count > kJettyDepth)
      return Status::InvalidArgument("invalid ordered WRITE chain");
    auto plan = std::make_unique<PreparedWrite>();
    auto& job = plan->job; job.write = true; job.queues.resize(local_jettys_.size());
    plan->ordered_chain = true;
    // Allocate worst-case storage before binding any descriptor pointers.
    std::array<std::size_t, kMaxJettyCount> capacity{};
    for (std::size_t c = 0; c < count; ++c) {
      const auto& chunk = chunks[c];
      if (!chunk.count || !chunk.sources || !chunk.remote_offsets || !chunk.marker_offsets ||
          !chunk.marker_value || reinterpret_cast<std::uintptr_t>(chunk.marker_value) % 8 ||
          chunk.count > kJettyDepth * local_jettys_.size())
        return Status::InvalidArgument("invalid chain chunk/source/marker alignment");
      auto status = ValidateAndRegisterHostBuffers(chunk.sources, chunk.remote_offsets, chunk.count);
      if (!status.ok()) return status;
      for (std::size_t q = 0; q < job.queues.size(); ++q) {
        const auto offset = chunk.marker_offsets[q];
        if (offset % 8 || offset > remote_region_.length || remote_region_.length - offset < 8)
          return Status::InvalidArgument("chain marker is outside remote region or unaligned");
        capacity[q] += PreparedRowsForQueue(chunk.count, job.queues.size(), q).count + 1;
        if (capacity[q] > kJettyDepth)
          return Status::InvalidArgument("WRITE chain source/marker budget exceeds one SQ wave; reduce rows/chunks");
      }
      // Each chunk shares one local marker across Jettys.
      const CpuUrmaTransport::HostSge marker{const_cast<std::uint64_t*>(chunk.marker_value), 8};
      status = ValidateAndRegisterHostBuffers(&marker, chunk.marker_offsets, 1);
      if (!status.ok()) return status;
    }
    for (std::size_t q = 0; q < job.queues.size(); ++q) {
      auto& queue = job.queues[q];
      queue.local.resize(capacity[q]); queue.remote.resize(capacity[q]);
      queue.wrs.resize(capacity[q]);
    }
    SglBatchJob temp; temp.write = true; temp.queues.resize(job.queues.size());
    std::array<std::size_t, kMaxJettyCount> locals{};
    for (std::size_t c = 0; c < count; ++c) {
      const auto& chunk = chunks[c];
      auto* marker_segment = explicit_host_region_.segment;
      if (!HostSpanFitsRegisteredBuffer(explicit_host_region_, chunk.marker_value, 8)) {
        const auto* host = FindRegisteredHostBuffer(chunk.marker_value, 8);
        marker_segment = host ? host->segment : nullptr;
      }
      if (!marker_segment) return Status::Internal("unresolved chain marker segment");
      for (std::size_t q = 0; q < job.queues.size(); ++q) {
        const auto range = PreparedRowsForQueue(chunk.count, job.queues.size(), q);
        auto& dst = job.queues[q]; auto& src = temp.queues[q];
        EnsureSglQueueStorage(src, range.count, range.count);
        const auto status = BuildWriteGatherQueue(temp, q, chunk.sources, chunk.remote_offsets,
                                                chunk.count, range.first, range.count);
        if (!status.ok()) return status;
        if (!src.missing_host_segments.empty()) return Status::Internal("unresolved chain host segment");
        const auto local_base = locals[q];
        // The gather builder emits exactly one local SGE per source row.
        std::copy_n(src.local.data(), range.count, dst.local.data() + local_base);
        locals[q] += range.count;
        for (std::size_t w = 0; w < src.count; ++w) {
          const auto index = dst.count++;
          dst.wrs[index] = src.wrs[w];
          auto& wr = dst.wrs[index];
          wr.rw.src.sge = reinterpret_cast<urma_sge_t*>(dst.local.data() + local_base +
              (reinterpret_cast<SglSgeAbi*>(src.wrs[w].rw.src.sge) - src.local.data()));
          dst.remote[index] = src.remote[w];
          wr.rw.dst.sge = reinterpret_cast<urma_sge_t*>(&dst.remote[index]);
          wr.flag.bs.complete_enable = 0;
          wr.flag.bs.place_order = URMA_RELAX_ORDER;
        }
        // Every queue publishes its own progress, including empty shards.
        const auto index = dst.count++, local = locals[q]++;
        dst.local[local] = {reinterpret_cast<std::uint64_t>(chunk.marker_value), 8, marker_segment, nullptr};
        dst.remote[index] = {remote_region_.base + chunk.marker_offsets[q], 8, remote_seg_handle_, nullptr};
        auto& wr = dst.wrs[index]; wr = {}; wr.opcode = URMA_OPC_WRITE;
        wr.flag.bs.place_order = URMA_STRONG_ORDER; wr.flag.bs.comp_order = 1;
        wr.tjetty = remote_jettys_[q];
        wr.rw.src.sge = reinterpret_cast<urma_sge_t*>(&dst.local[local]); wr.rw.src.num_sge = 1;
        wr.rw.dst.sge = reinterpret_cast<urma_sge_t*>(&dst.remote[index]); wr.rw.dst.num_sge = 1;
        ++plan->marker_wrs;
        if (!c) plan->first_prefix[q] = dst.count;
      }
    }
    for (std::size_t q = 0; q < job.queues.size(); ++q) {
      auto& queue = job.queues[q];
      if (queue.count > kJettyDepth) return Status::InvalidArgument("WRITE chain exceeds one SQ wave; reduce rows/chunks");
      for (std::size_t w = 0; w < queue.count; ++w)
        queue.wrs[w].next = w + 1 < queue.count ? &queue.wrs[w + 1] : nullptr;
      // Permanent cuts agree with first-prefix/suffix submission and doorbells.
      for (unsigned phase = 0; phase < 2; ++phase)
        cpu_urma_detail::ForEachChainPost(phase ? plan->first_prefix[q] : 0,
            phase ? queue.count : plan->first_prefix[q], wqes_per_doorbell_,
            [&](std::size_t, std::size_t end) { queue.wrs[end - 1].next = nullptr; });
      auto& tail = queue.wrs[queue.count - 1]; tail.flag.bs.complete_enable = 1;
      tail.flag.bs.place_order = URMA_STRONG_ORDER; tail.flag.bs.comp_order = 1;
      plan->fences[plan->fence_count++] = {q, queue.count, 0};
    }
    prepared_writes_.push_back(std::move(plan)); *id = prepared_writes_.size();
    return Status::OK();
  }

  Status DescribePreparedWrite(CpuUrmaTransport::PreparedWriteId id,
                              CpuUrmaTransport::PreparedWriteInfo* info) {
    std::unique_lock<std::mutex> lock(batch_submit_mutex_);
    if (!info || !id || id > prepared_writes_.size())
      return Status::InvalidArgument("unknown prepared WRITE or missing info");
    *info = {};
    const auto& plan = *prepared_writes_[id - 1];
    info->marker_wrs = plan.marker_wrs;
    for (const auto& q : plan.job.queues) if (q.count) {
      info->wr_count += q.count;
      ++info->tail_cqes;
    }
    return Status::OK();
  }

  Status SubmitPreparedWriteLocked(CpuUrmaTransport::PreparedWriteId id, bool* submitted) {
    if (!submitted) return Status::InvalidArgument("missing submission result");
    *submitted = false;
    if (!initialized_ || batch_poisoned_)
      return Status::InvalidState("prepared WRITE transport is unavailable");
    if (!id || id > prepared_writes_.size()) return Status::InvalidArgument("unknown prepared WRITE");
    auto& plan = *prepared_writes_[id - 1];
    auto& job = plan.job;
    // Reject duplicate submissions before modifying live WRs.
    if (plan.handle.token) return Status::InvalidState("prepared WRITE is already in flight");
    const auto start = std::chrono::steady_clock::now();
    auto status = indexed_tracker_.TryBegin(id, plan.handle, plan.fences.data(), plan.fence_count, submitted);
    if (!status.ok() || !*submitted) return status;
    job.deadline = start + std::chrono::seconds(5);
    for (std::size_t i = 0; i < plan.fence_count; ++i) {
      const auto& f = plan.fences[i];
      job.queues[f.queue].wrs[f.wrs - 1].user_ctx = f.context;
      const auto error = plan.ordered_chain ? std::string{} : PostSglQueue(job, f.queue);
      if (!error.empty()) {
        // A failed post may leave live WRs; retain sources and descriptors until teardown.
        batch_poisoned_ = true;
        return Status::Internal(error);
      }
    }
    if (plan.ordered_chain) {
      std::string error;
      const bool ok = cpu_urma_detail::DispatchWriteChain(job.queues.size(),
          [&](std::size_t q) { return plan.first_prefix[q]; },
          [&](std::size_t q) { return job.queues[q].count; },
          [&](std::size_t q, std::size_t begin, std::size_t end) {
            bool success = true;
            cpu_urma_detail::ForEachChainPost(begin, end, wqes_per_doorbell_,
                [&](std::size_t first, std::size_t) {
                  if (!success) return;
                  urma_jfs_wr_t* bad = nullptr;
                  const auto rc = urma_post_jetty_send_wr(local_jettys_[q], &job.queues[q].wrs[first], &bad);
                  if (rc != URMA_SUCCESS) {
                    success = false;
                    error = "WRITE chain post failed (jetty=" + std::to_string(q) +
                        ", status=" + std::to_string(static_cast<int>(rc)) + ")";
                  }
                });
            return success;
          });
      if (!ok) { batch_poisoned_ = true; return Status::Internal(error); }
    }
    return Status::OK();
  }

  Status PollPreparedWritesLocked(CpuUrmaTransport::PreparedWriteCompletion* completed,
                                 std::size_t capacity, std::size_t* count) {
    if (!completed || !capacity || !count) return Status::InvalidArgument("invalid async completion buffer");
    *count = 0;
    if (!initialized_ || batch_poisoned_) return Status::InvalidState("async CQ transport is unavailable");
    std::array<urma_cr_t, 64> crs;
    std::size_t output_count = 0;
    auto sweep = cpu_urma_detail::SweepAsyncCqs(jfcs_.size(), next_async_poll_queue_,
        async_cq_batch_, capacity, count,
        [&](std::size_t q) { return indexed_tracker_.Pending(q); },
        [&](std::size_t q, std::size_t limit, std::size_t* polled) -> Status {
          const auto n = urma_poll_jfc(jfcs_[q], static_cast<int>(limit), crs.data());
          if (n < 0) return Status::Internal("prepared WRITE CQ poll failed (queue=" + std::to_string(q) + ")");
          *polled = static_cast<std::size_t>(n);
          async_stats_.tail_cqes += static_cast<std::uint64_t>(n);
          return Status::OK();
        },
        [&](std::size_t q, std::size_t index, bool* finished) -> Status {
          const auto& cr = crs[index];
          if (cr.status != URMA_CR_SUCCESS)
            return Status::Internal("prepared WRITE CQ failed (queue=" + std::to_string(q) + ")");
          std::uint64_t id = 0;
          auto status = indexed_tracker_.Complete(q, cr.user_ctx, &id);
          if (!status.ok()) return status;
          if (!id) return Status::OK();
          completed[output_count++] = {id}; *finished = true;
          return Status::OK();
        });
    if (!sweep.ok()) { batch_poisoned_ = true; return sweep; }
    if (++next_async_poll_queue_ == jfcs_.size()) next_async_poll_queue_ = 0;
    ++async_polls_since_deadline_;
    return CheckPreparedDeadlines();
  }

  Status CheckPreparedDeadlines() {
    bool expired = false;
    // Check all outstanding deadlines once per configured poll interval.
    if (async_polls_since_deadline_ >= async_deadline_polls_) {
      async_polls_since_deadline_ = 0;
      const auto now = std::chrono::steady_clock::now();
      indexed_tracker_.ForEachActive([&](std::uint64_t id) { expired |= now >= prepared_writes_[id - 1]->job.deadline; });
    }
    if (expired) {
      batch_poisoned_ = true;
      return Status::Internal("prepared WRITE completion timed out");
    }
    return Status::OK();
  }

  Status ReadPreparedAsyncStats(CpuUrmaTransport::PreparedAsyncStats* stats, bool reset) {
    if (!stats) return Status::InvalidArgument("missing prepared async stats");
    std::unique_lock<std::mutex> lock(batch_submit_mutex_);
    *stats = async_stats_;
    if (reset) { async_stats_ = {}; async_polls_since_deadline_ = 0; }
    return Status::OK();
  }

  Status Shutdown() {
    std::unique_lock<std::mutex> submit_lock(batch_submit_mutex_);
    initialized_ = false;
    // Stop local queues before releasing imported targets or registered sources.
    bool delete_failed = false;
    for (auto*& local_jetty : local_jettys_) {
      if (local_jetty == nullptr) continue;
      if (urma_delete_jetty(local_jetty) == URMA_SUCCESS) local_jetty = nullptr;
      else delete_failed = true;
    }
    if (delete_failed)
      return Status::Internal("Jetty teardown failed; retain outstanding WRITE sources and descriptors");
    local_jettys_.clear();
    for (auto* remote_jetty : remote_jettys_) {
      if (remote_jetty != nullptr) (void)urma_unimport_jetty(remote_jetty);
    }
    remote_jettys_.clear();
    prepared_writes_.clear();
    indexed_tracker_ = cpu_urma_detail::IndexedAsyncWriteTracker();
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
    batch_generation_ = 0U;
    batch_poisoned_ = false;
    return Status::OK();
  }

  std::unique_ptr<peer::PeerLink> link_;
  peer::PeerRegion remote_region_{};
  urma_device_t* device_{nullptr};
  urma_context_t* context_{nullptr};
  urma_target_seg_t* remote_seg_handle_{nullptr};
  urma_seg_t remote_seg_{};
  std::uint32_t eid_index_{0};
  std::uint32_t max_local_sge_{1U};
  std::uint64_t max_message_bytes_{std::numeric_limits<std::uint32_t>::max()};
  std::vector<urma_jfc_t*> jfcs_;
  std::vector<urma_jfr_t*> jfrs_;
  std::vector<urma_jetty_t*> local_jettys_;
  std::vector<urma_target_jetty_t*> remote_jettys_;
  std::mutex batch_submit_mutex_;
  SglBatchJob sgl_batch_workspace_;
  std::vector<std::unique_ptr<PreparedWrite>> prepared_writes_;
  cpu_urma_detail::IndexedAsyncWriteTracker indexed_tracker_;
  std::size_t async_cq_batch_{1}, async_deadline_polls_{1}, async_polls_since_deadline_{};
  std::size_t next_async_poll_queue_{};
  CpuUrmaTransport::PreparedAsyncStats async_stats_{};
  std::size_t wqes_per_doorbell_{0U};
  std::size_t batch_local_sgl_size_{1U};
  std::size_t prepare_thread_count_{4U};
  std::vector<cpu_urma_detail::BatchValidationSlice>
      batch_validation_slices_;
  CpuUrmaTransport::BatchTiming last_batch_timing_{};
  std::uint64_t batch_generation_{0U};
  bool batch_poisoned_{false};
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

struct CpuUrmaTransport::PreparedWriteSession::State {
  explicit State(Impl* transport) : impl(transport), lock(transport->batch_submit_mutex_) {}
  Impl* impl;
  std::unique_lock<std::mutex> lock;
};
CpuUrmaTransport::PreparedWriteSession::PreparedWriteSession(std::unique_ptr<State> state)
    : state_(std::move(state)) {}
CpuUrmaTransport::PreparedWriteSession::~PreparedWriteSession() = default;
Status CpuUrmaTransport::PreparedWriteSession::Submit(PreparedWriteId id, bool* submitted) {
  return state_->impl->SubmitPreparedWriteLocked(id, submitted);
}
Status CpuUrmaTransport::PreparedWriteSession::Poll(PreparedWriteCompletion* completed,
                                                  std::size_t capacity, std::size_t* count) {
  return state_->impl->PollPreparedWritesLocked(completed, capacity, count);
}
Status CpuUrmaTransport::BeginPreparedWriteSession(std::unique_ptr<PreparedWriteSession>* session) {
  if (!session || *session) return Status::InvalidArgument("missing/nonempty prepared WRITE session");
  auto state = std::make_unique<PreparedWriteSession::State>(impl_.get());
  if (!impl_->initialized_ || impl_->batch_poisoned_ || !impl_->indexed_tracker_.Empty())
    return Status::InvalidState("prepared WRITE session requires an initialized, idle transport");
  session->reset(new PreparedWriteSession(std::move(state)));
  return Status::OK();
}

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
Status CpuUrmaTransport::PrepareWriteBatch(const std::uint64_t* offsets,
    const HostSge* sources, std::size_t count, PreparedWriteId* id) {
  return impl_->PrepareWriteBatch(offsets, sources, count, id);
}
Status CpuUrmaTransport::PrepareWriteChain(const WriteChainChunk* chunks, std::size_t count,
    PreparedWriteId* id) {
  return impl_->PrepareWriteChain(chunks, count, id);
}
Status CpuUrmaTransport::DescribePreparedWrite(PreparedWriteId id, PreparedWriteInfo* info) const {
  return impl_->DescribePreparedWrite(id, info);
}
Status CpuUrmaTransport::ReadPreparedAsyncStats(PreparedAsyncStats* stats, bool reset) const {
  return impl_->ReadPreparedAsyncStats(stats, reset);
}
std::uint64_t CpuUrmaTransport::max_write_bytes() const { return impl_->max_message_bytes_; }
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
