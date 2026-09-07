#include "cpu_urma_transport.h"

#include <dlfcn.h>
#include <errno.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <omp.h>
#include <urma_api.h>

#include "peer_link.h"

namespace kvaf {
namespace {

constexpr std::uint32_t kJfcDepth = 1024;
constexpr std::uint32_t kJettyDepth = 1024;
constexpr std::uint32_t kMaxJettyCount = 16;
constexpr std::uint32_t kMaxSge = 16;
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
  struct BatchQueue {
    std::vector<urma_sge_t> local;
    std::vector<urma_sge_t> remote;
    std::vector<urma_jfs_wr_t> wrs;
  };

  struct BatchJob {
    bool write{false};
    std::vector<BatchQueue> queues;
    const CpuUrmaTransport::HostSge* buffers{nullptr};
    std::size_t count{0U};
    urma_target_seg_t* single_host_segment{nullptr};
    std::vector<std::uint64_t> remote_offsets;
    std::vector<urma_target_seg_t*> host_segments;
    std::chrono::steady_clock::time_point deadline{};
  };

  // A pipeline slot contains one doorbell-sized wave.  Two slots are enough
  // to let the OpenMP producer prepare wave N+1 while the submitter posts
  // wave N.  A slot is released only after its fence CQEs have been polled.
  struct PipelineSlot {
    std::vector<BatchQueue> queues;
    std::vector<std::uint64_t> expected_ctx;
    std::size_t ready_workers{0U};
    std::size_t free_epoch{0U};
    std::mutex state_mutex;
    std::condition_variable state_cv;
  };

  void PrepareBatchEntry(BatchJob& job, std::size_t index) {
    const std::size_t jetty = index % local_jettys_.size();
    const std::size_t position = index / local_jettys_.size();
    BatchQueue& queue = job.queues[jetty];
    const auto& buffer = job.buffers[index];
    urma_target_seg_t* host_segment = job.single_host_segment != nullptr
        ? job.single_host_segment : job.host_segments[index];
    queue.local[position] = {
        reinterpret_cast<std::uint64_t>(buffer.data),
        static_cast<std::uint32_t>(buffer.bytes), host_segment};
    queue.remote[position] = {
        remote_region_.base + job.remote_offsets[index],
        static_cast<std::uint32_t>(buffer.bytes), remote_seg_handle_};
    queue.wrs[position] = {};
    queue.wrs[position].opcode = job.write ? URMA_OPC_WRITE : URMA_OPC_READ;
    queue.wrs[position].flag.bs.complete_enable =
        (position + 1U == queue.wrs.size()) ? 1U : 0U;
    queue.wrs[position].tjetty = remote_jettys_[jetty];
    queue.wrs[position].user_ctx =
        0xBA7C0000ULL + (static_cast<std::uint64_t>(jetty) << 32U) + position;
    if (job.write) {
      queue.wrs[position].rw.src = {&queue.local[position], 1U};
      queue.wrs[position].rw.dst = {&queue.remote[position], 1U};
    } else {
      queue.wrs[position].rw.src = {&queue.remote[position], 1U};
      queue.wrs[position].rw.dst = {&queue.local[position], 1U};
    }
    queue.wrs[position].next =
        position + 1U < queue.wrs.size() ? &queue.wrs[position + 1U] : nullptr;
  }

  void PrepareBatch(BatchJob& job) {
#pragma omp parallel for schedule(static) num_threads(openmp_thread_count_)
    for (std::int64_t request = 0; request < static_cast<std::int64_t>(job.count); ++request) {
      PrepareBatchEntry(job, static_cast<std::size_t>(request));
    }
  }

  std::string PostBatchChunk(BatchJob& job, std::size_t begin,
                             std::size_t chunk_size) {
    for (std::size_t jetty = 0U; jetty < local_jettys_.size(); ++jetty) {
      BatchQueue& queue = job.queues[jetty];
      if (begin >= queue.wrs.size()) continue;
      const std::size_t end = std::min(begin + chunk_size, queue.wrs.size());
      // URMA follows next pointers, so terminate this chunk before posting.
      // The next batch rebuilds the links; no provider-visible state depends
      // on the host-side next pointer after the call returns.
      queue.wrs[end - 1U].next = nullptr;
      urma_jfs_wr_t* bad = nullptr;
      const urma_status_t post_status =
          urma_post_jetty_send_wr(local_jettys_[jetty], &queue.wrs[begin], &bad);
      if (post_status != URMA_SUCCESS) {
        return "batch post failed (jetty=" + std::to_string(jetty) +
               ", status=" + std::to_string(static_cast<int>(post_status)) +
               ", bad_index=" +
               std::to_string(bad == nullptr
                                  ? end
                                  : begin + static_cast<std::size_t>(bad - &queue.wrs[begin])) +
               ", count=" + std::to_string(queue.wrs.size()) + ")";
      }
    }
    return {};
  }

  std::string PostBatch(BatchJob& job) {
    std::size_t max_queue_size = 0U;
    for (const auto& queue : job.queues) {
      max_queue_size = std::max(max_queue_size, queue.wrs.size());
    }
    return PostBatchChunk(job, 0U, max_queue_size);
  }

  void PreparePipelineEntry(const BatchJob& plan, PipelineSlot& slot,
                            std::size_t jetty, std::size_t position,
                            std::size_t global_index, std::size_t wave) {
    BatchQueue& queue = slot.queues[jetty];
    const auto& buffer = plan.buffers[global_index];
    urma_target_seg_t* host_segment = plan.single_host_segment != nullptr
        ? plan.single_host_segment : plan.host_segments[global_index];
    queue.local[position] = {
        reinterpret_cast<std::uint64_t>(buffer.data),
        static_cast<std::uint32_t>(buffer.bytes), host_segment};
    queue.remote[position] = {
        remote_region_.base + plan.remote_offsets[global_index],
        static_cast<std::uint32_t>(buffer.bytes), remote_seg_handle_};
    queue.wrs[position] = {};
    queue.wrs[position].opcode = plan.write ? URMA_OPC_WRITE : URMA_OPC_READ;
    queue.wrs[position].flag.bs.complete_enable = 0U;
    queue.wrs[position].tjetty = remote_jettys_[jetty];
    queue.wrs[position].user_ctx =
        0xBA7C0000ULL + (static_cast<std::uint64_t>(wave) << 16U) +
        (static_cast<std::uint64_t>(jetty) << 8U) + position;
    if (plan.write) {
      queue.wrs[position].rw.src = {&queue.local[position], 1U};
      queue.wrs[position].rw.dst = {&queue.remote[position], 1U};
    } else {
      queue.wrs[position].rw.src = {&queue.remote[position], 1U};
      queue.wrs[position].rw.dst = {&queue.local[position], 1U};
    }
    queue.wrs[position].next =
        position + 1U < queue.wrs.size() ? &queue.wrs[position + 1U] : nullptr;
  }

  void ResizePipelineSlot(const BatchJob& plan, PipelineSlot& slot,
                          std::size_t begin) {
    const std::size_t chunk_size = wqes_per_doorbell_;
    slot.expected_ctx.assign(local_jettys_.size(), 0U);
    for (std::size_t jetty = 0U; jetty < local_jettys_.size(); ++jetty) {
      const std::size_t total = plan.queues[jetty].wrs.size();
      const std::size_t remaining = begin < total ? total - begin : 0U;
      const std::size_t entries = std::min(chunk_size, remaining);
      slot.queues[jetty].local.resize(entries);
      slot.queues[jetty].remote.resize(entries);
      slot.queues[jetty].wrs.resize(entries);
    }
  }

  std::string PostPipelineSlot(PipelineSlot& slot) {
    for (std::size_t jetty = 0U; jetty < local_jettys_.size(); ++jetty) {
      BatchQueue& queue = slot.queues[jetty];
      if (queue.wrs.empty()) continue;
      urma_jfs_wr_t* bad = nullptr;
      const urma_status_t post_status =
          urma_post_jetty_send_wr(local_jettys_[jetty], &queue.wrs.front(), &bad);
      if (post_status != URMA_SUCCESS) {
        return "pipeline post failed (jetty=" + std::to_string(jetty) +
               ", status=" + std::to_string(static_cast<int>(post_status)) +
               ", bad_wr=" + std::to_string(reinterpret_cast<std::uintptr_t>(bad)) +
               ")";
      }
    }
    return {};
  }

  std::string PollPipelineSlot(PipelineSlot& slot,
                               std::chrono::steady_clock::time_point deadline) {
    for (std::size_t jetty = 0U; jetty < local_jettys_.size(); ++jetty) {
      if (slot.queues[jetty].wrs.empty()) continue;
      bool completed = false;
      while (std::chrono::steady_clock::now() < deadline) {
        urma_cr_t cr{};
        const int polled = urma_poll_jfc(jfcs_[jetty], 1, &cr);
        if (polled < 0) return "urma_poll_jfc for pipeline slot";
        if (polled != 1) continue;
        if (cr.status != URMA_CR_SUCCESS ||
            cr.user_ctx != slot.expected_ctx[jetty]) {
          return "pipeline completion failed (jetty=" + std::to_string(jetty) +
                 ", status=" + std::to_string(static_cast<int>(cr.status)) +
                 ", user_ctx=" +
                 std::to_string(static_cast<unsigned long long>(cr.user_ctx)) +
                 ", expected_ctx=" +
                 std::to_string(static_cast<unsigned long long>(slot.expected_ctx[jetty])) +
                 ")";
        }
        completed = true;
        break;
      }
      if (!completed) return "timed out waiting for pipeline completion";
    }
    return {};
  }

  std::string PrepareAndPostDoubleBuffered(
      BatchJob& plan, std::uint64_t* prepare_ns, std::uint64_t* post_ns,
      std::uint64_t* poll_ns) {
    const std::size_t chunk_size = wqes_per_doorbell_;
    const std::size_t jetty_count = local_jettys_.size();
    std::size_t max_queue_size = 0U;
    for (const auto& queue : plan.queues) {
      max_queue_size = std::max(max_queue_size, queue.wrs.size());
    }
    const std::size_t waves = (max_queue_size + chunk_size - 1U) / chunk_size;
    if (waves == 0U) {
      *prepare_ns = *post_ns = *poll_ns = 0U;
      return {};
    }
    PipelineSlot& slot0 = pipeline_slots_[0];
    PipelineSlot& slot1 = pipeline_slots_[1];
    {
      std::lock_guard<std::mutex> state_lock(slot0.state_mutex);
      slot0.free_epoch = 0U;
      slot0.ready_workers = 0U;
    }
    {
      std::lock_guard<std::mutex> state_lock(slot1.state_mutex);
      slot1.free_epoch = 1U;
      slot1.ready_workers = 0U;
    }
    const std::size_t workers = openmp_thread_count_ > 1U
        ? openmp_thread_count_ - 1U : 0U;
    std::atomic<bool> failed{false};
    std::string error;
    std::uint64_t prepare_total = 0U;
    std::uint64_t post_total = 0U;
    std::uint64_t poll_total = 0U;

    // Publish the initial two generations before workers enter the parallel
    // region.  Later generations are resized by the submitter immediately
    // after polling the slot that is about to be reused.
    ResizePipelineSlot(plan, slot0, 0U);
    if (waves > 1U) ResizePipelineSlot(plan, slot1, chunk_size);

    if (workers == 0U) {
      // A single OpenMP thread cannot overlap producer and submitter. Keep a
      // correct sequential fallback for --omp-threads 1.
      for (std::size_t wave = 0U; wave < waves; ++wave) {
        PipelineSlot& slot = (wave & 1U) == 0U ? slot0 : slot1;
        const std::size_t begin = wave * chunk_size;
        const auto prepare_start = std::chrono::steady_clock::now();
        ResizePipelineSlot(plan, slot, begin);
        for (std::size_t jetty = 0U; jetty < jetty_count; ++jetty) {
          BatchQueue& queue = slot.queues[jetty];
          for (std::size_t position = 0U; position < queue.wrs.size(); ++position) {
            const std::size_t global_index = (begin + position) * jetty_count + jetty;
            PreparePipelineEntry(plan, slot, jetty, position, global_index, wave);
          }
          if (!queue.wrs.empty()) {
            queue.wrs.back().flag.bs.complete_enable = 1U;
            queue.wrs.back().next = nullptr;
            slot.expected_ctx[jetty] = queue.wrs.back().user_ctx;
          }
        }
        const auto prepare_end = std::chrono::steady_clock::now();
        const auto post_start = prepare_end;
        error = PostPipelineSlot(slot);
        const auto post_end = std::chrono::steady_clock::now();
        const auto poll_start = post_end;
        if (error.empty()) error = PollPipelineSlot(slot, plan.deadline);
        const auto poll_end = std::chrono::steady_clock::now();
        prepare_total += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(prepare_end - prepare_start).count());
        post_total += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(post_end - post_start).count());
        poll_total += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(poll_end - poll_start).count());
        if (!error.empty()) break;
      }
      *prepare_ns = prepare_total;
      *post_ns = post_total;
      *poll_ns = poll_total;
      return error;
    }

#pragma omp parallel num_threads(openmp_thread_count_)
    {
      const int tid = omp_get_thread_num();
      if (tid == 0) {
        for (std::size_t wave = 0U; wave < waves && !failed.load(); ++wave) {
          PipelineSlot& slot = (wave & 1U) == 0U ? slot0 : slot1;
          const auto prepare_start = std::chrono::steady_clock::now();
          {
            std::unique_lock<std::mutex> state_lock(slot.state_mutex);
            slot.state_cv.wait(state_lock, [&] {
              return slot.ready_workers >= workers ||
                     failed.load(std::memory_order_acquire);
            });
          }
          prepare_total += static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - prepare_start).count());
          if (failed.load(std::memory_order_acquire)) break;
          for (std::size_t jetty = 0U; jetty < jetty_count; ++jetty) {
            BatchQueue& queue = slot.queues[jetty];
            if (!queue.wrs.empty()) {
              queue.wrs.back().flag.bs.complete_enable = 1U;
              queue.wrs.back().next = nullptr;
              slot.expected_ctx[jetty] = queue.wrs.back().user_ctx;
            }
          }
          const auto post_start = std::chrono::steady_clock::now();
          error = PostPipelineSlot(slot);
          post_total += static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - post_start).count());
          if (!error.empty()) {
            failed.store(true, std::memory_order_release);
            slot0.state_cv.notify_all();
            slot1.state_cv.notify_all();
            break;
          }
          if (wave > 0U) {
            PipelineSlot& old = (wave & 1U) == 0U ? slot1 : slot0;
            const auto poll_start = std::chrono::steady_clock::now();
            error = PollPipelineSlot(old, plan.deadline);
            poll_total += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - poll_start).count());
            if (!error.empty()) {
              failed.store(true, std::memory_order_release);
              slot0.state_cv.notify_all();
              slot1.state_cv.notify_all();
              break;
            }
            if (wave + 1U < waves) {
              ResizePipelineSlot(plan, old, (wave + 1U) * chunk_size);
              {
                std::lock_guard<std::mutex> state_lock(old.state_mutex);
                old.ready_workers = 0U;
                old.free_epoch = wave + 1U;
              }
              old.state_cv.notify_all();
            }
          }
        }
        if (!failed.load(std::memory_order_acquire)) {
          PipelineSlot& last = ((waves - 1U) & 1U) == 0U ? slot0 : slot1;
          const auto poll_start = std::chrono::steady_clock::now();
          error = PollPipelineSlot(last, plan.deadline);
          poll_total += static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - poll_start).count());
          if (!error.empty()) {
            failed.store(true, std::memory_order_release);
            slot0.state_cv.notify_all();
            slot1.state_cv.notify_all();
          }
        }
      } else {
        const std::size_t worker = static_cast<std::size_t>(tid - 1);
        for (std::size_t wave = 0U; wave < waves; ++wave) {
          PipelineSlot& slot = (wave & 1U) == 0U ? slot0 : slot1;
          {
            std::unique_lock<std::mutex> state_lock(slot.state_mutex);
            slot.state_cv.wait(state_lock, [&] {
              return slot.free_epoch == wave ||
                     failed.load(std::memory_order_acquire);
            });
          }
          if (failed.load(std::memory_order_acquire)) break;
          const std::size_t begin = wave * chunk_size;
          for (std::size_t jetty = 0U; jetty < jetty_count; ++jetty) {
            const std::size_t entries = slot.queues[jetty].wrs.size();
            for (std::size_t position = worker; position < entries;
                 position += workers) {
              const std::size_t global_index =
                  (begin + position) * jetty_count + jetty;
              PreparePipelineEntry(plan, slot, jetty, position,
                                   global_index, wave);
            }
          }
          {
            std::lock_guard<std::mutex> state_lock(slot.state_mutex);
            ++slot.ready_workers;
          }
          slot.state_cv.notify_one();
          if (failed.load(std::memory_order_acquire)) break;
        }
      }
    }
    *prepare_ns = prepare_total;
    *post_ns = post_total;
    *poll_ns = poll_total;
    return error;
  }

  std::string PollBatch(BatchJob& job) {
    // Post every queue before draining one fence CQE from each used queue.
    for (std::size_t jetty = 0U; jetty < local_jettys_.size(); ++jetty) {
      BatchQueue& queue = job.queues[jetty];
      if (queue.wrs.empty()) continue;
      const std::uint64_t expected_ctx = queue.wrs.back().user_ctx;
      bool completed = false;
      while (std::chrono::steady_clock::now() < job.deadline) {
        urma_cr_t cr{};
        const int polled = urma_poll_jfc(jfcs_[jetty], 1, &cr);
        if (polled < 0) return "urma_poll_jfc for batch (jetty=" + std::to_string(jetty) + ")";
        if (polled != 1) continue;
        if (cr.status != URMA_CR_SUCCESS || cr.user_ctx != expected_ctx) {
          return "batch completion failed (jetty=" + std::to_string(jetty) +
                 ", status=" + std::to_string(static_cast<int>(cr.status)) +
                 ", user_ctx=" +
                 std::to_string(static_cast<unsigned long long>(cr.user_ctx)) +
                 ", expected_ctx=" +
                 std::to_string(static_cast<unsigned long long>(expected_ctx)) + ")";
        }
        completed = true;
        break;
      }
      if (completed) continue;
      urma_jetty_cfg_t observed_cfg{};
      urma_jetty_attr_t observed_attr{};
      observed_attr.mask = JETTY_STATE;
      const urma_status_t query_status =
          urma_query_jetty(local_jettys_[jetty], &observed_cfg, &observed_attr);
      return "timed out waiting for batch completion (jetty=" +
             std::to_string(jetty) + ", jetty_query_status=" +
             std::to_string(static_cast<int>(query_status)) + ", jetty_state=" +
             std::to_string(static_cast<unsigned>(observed_attr.state)) + ")";
    }
    return {};
  }

  ~Impl() { (void)Shutdown(); }

  Status Initialize(const CpuUrmaTransportOptions& options) {
    if (options.peer_address.empty() || options.region_bytes == 0 ||
        options.region_bytes > std::numeric_limits<std::size_t>::max() ||
        options.jetty_count == 0U || options.jetty_count > kMaxJettyCount ||
        options.omp_thread_count == 0U ||
        options.omp_thread_count > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        options.wqes_per_doorbell > kJettyDepth) {
      return Status::InvalidArgument(
          "CPU URMA transport requires peer_address, non-zero region_bytes, "
          "jetty_count in [1,16], and omp_thread_count in [1,INT_MAX]");
    }
    options_ = options;
    // The pipeline uses one OpenMP submitter plus (N-1) producers.  Disable
    // dynamic teams so the requested team size is honored and the producer
    // completion counters cannot wait for threads that were never created.
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(options.omp_thread_count));
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
    if (urma_query_device(device_, &device_attr) == URMA_SUCCESS) {
      std::fprintf(stderr,
                   "[KVAF_CPU_URMA] DEVICE_CAP max_jfs_sge=%u max_jfs_rsge=%u "
                   "max_jfr_sge=%u\n",
                   device_attr.dev_cap.max_jfs_sge,
                   device_attr.dev_cap.max_jfs_rsge,
                   device_attr.dev_cap.max_jfr_sge);
      if (device_attr.dev_cap.max_jfs_sge != 0U) {
        max_sge_ = std::min<std::uint32_t>(kMaxSge, device_attr.dev_cap.max_jfs_sge);
      }
    }
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
      jetty_cfg.jfs_cfg.max_sge = static_cast<std::uint8_t>(max_sge_);
      jetty_cfg.jfs_cfg.max_rsge = 1;
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
    jfc_ = jfcs_[0];
    jfr_ = jfrs_[0];
    local_jetty_ = local_jettys_[0];
    remote_jetty_ = remote_jettys_[0];
    openmp_thread_count_ = static_cast<std::uint32_t>(omp_get_max_threads());
    wqes_per_doorbell_ = options.wqes_per_doorbell;
    std::fprintf(stderr, "[KVAF_CPU_URMA] JETTYS_READY count=%zu omp_threads=%u wqes_per_doorbell=%zu depth=%u\n",
                 local_jettys_.size(), openmp_thread_count_,
                 wqes_per_doorbell_, kJettyDepth);
    batch_workspace_.queues.resize(local_jettys_.size());
    const std::size_t queue_capacity =
        (kJettyDepth + local_jettys_.size() - 1U) / local_jettys_.size();
    for (auto& queue : batch_workspace_.queues) {
      queue.local.reserve(queue_capacity);
      queue.remote.reserve(queue_capacity);
      queue.wrs.reserve(queue_capacity);
    }
    batch_workspace_.remote_offsets.reserve(kJettyDepth);
    batch_workspace_.host_segments.reserve(kJettyDepth);
    for (auto& slot : pipeline_slots_) {
      slot.queues.resize(local_jettys_.size());
      slot.expected_ctx.resize(local_jettys_.size());
      for (auto& queue : slot.queues) {
        queue.local.reserve(wqes_per_doorbell_ == 0U ? queue_capacity
                                                    : wqes_per_doorbell_);
        queue.remote.reserve(wqes_per_doorbell_ == 0U ? queue_capacity
                                                      : wqes_per_doorbell_);
        queue.wrs.reserve(wqes_per_doorbell_ == 0U ? queue_capacity
                                                   : wqes_per_doorbell_);
      }
    }
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
    cfg.flag.bs.non_pin = 1;
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

  Status TransferSgl(bool write, std::uint64_t offset,
                     const CpuUrmaTransport::HostSge* buffers, std::size_t count) {
    if (!initialized_) return Status::InvalidState("CPU URMA transport is not initialized");
    if (buffers == nullptr || count == 0U || count > max_sge_ ||
        offset > remote_region_.length) {
      return Status::InvalidArgument("invalid SGL transfer");
    }
    std::vector<urma_sge_t> local(count);
    std::uint64_t total = 0U;
    for (std::size_t i = 0; i < count; ++i) {
      if (buffers[i].data == nullptr || buffers[i].bytes == 0U ||
          buffers[i].bytes > std::numeric_limits<std::uint32_t>::max() ||
          total > std::numeric_limits<std::uint64_t>::max() - buffers[i].bytes) {
        return Status::InvalidArgument("invalid SGL entry");
      }
      total += buffers[i].bytes;
      RegisteredHostBuffer* host = FindRegisteredHostBuffer(buffers[i].data, buffers[i].bytes);
      if (host == nullptr) {
        Status status = RegisterHostBuffer(buffers[i].data, buffers[i].bytes, &host);
        if (!status.ok()) return status;
      }
      local[i].addr = reinterpret_cast<std::uint64_t>(buffers[i].data);
      local[i].len = static_cast<std::uint32_t>(buffers[i].bytes);
      local[i].tseg = host->segment;
    }
    if (total == 0U || total > remote_region_.length - offset ||
        total > std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument("SGL transfer exceeds remote region");
    }
    urma_sge_t remote{};
    remote.addr = remote_region_.base + offset;
    remote.len = static_cast<std::uint32_t>(total);
    remote.tseg = remote_seg_handle_;
    urma_sg_t local_sg{local.data(), static_cast<std::uint32_t>(count)};
    urma_sg_t remote_sg{&remote, 1U};
    urma_jfs_wr_t wr{};
    wr.opcode = write ? URMA_OPC_WRITE : URMA_OPC_READ;
    wr.flag.bs.complete_enable = 1;
    wr.tjetty = remote_jetty_;
    wr.user_ctx = 0xD1CFULL;
    if (write) {
      wr.rw.src = local_sg;
      wr.rw.dst = remote_sg;
    } else {
      wr.rw.src = remote_sg;
      wr.rw.dst = local_sg;
    }
    urma_jfs_wr_t* bad = nullptr;
    const urma_status_t post_status = urma_post_jetty_send_wr(local_jetty_, &wr, &bad);
    if (post_status != URMA_SUCCESS) {
      return Status::Internal("urma_post_jetty_send_wr for SGL failed (status=" +
                             std::to_string(static_cast<int>(post_status)) +
                             ", bad_wr=" +
                             std::to_string(reinterpret_cast<std::uintptr_t>(bad)) +
                             ", num_sge=" + std::to_string(count) + ", errno=" +
                             std::to_string(errno) + ")");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      urma_cr_t cr{};
      const int polled = urma_poll_jfc(jfc_, 1, &cr);
      if (polled < 0) return Error("urma_poll_jfc for SGL");
      if (polled == 1) {
        if (cr.status != URMA_CR_SUCCESS || cr.user_ctx != 0xD1CFULL) {
          std::fprintf(stderr,
                       "[KVAF_CPU_URMA] SGL_CQE_ERROR op=%s num_sge=%zu status=%d "
                       "completion_len=%u user_ctx=0x%llx\n",
                       write ? "WRITE" : "READ", count, static_cast<int>(cr.status),
                       cr.completion_len,
                       static_cast<unsigned long long>(cr.user_ctx));
          for (std::size_t i = 0; i < local.size(); ++i) {
            std::fprintf(stderr,
                         "[KVAF_CPU_URMA] SGL_ENTRY index=%zu addr=0x%llx len=%u "
                         "token_id=%u\n",
                         i, static_cast<unsigned long long>(local[i].addr), local[i].len,
                         local[i].tseg == nullptr ? 0U : local[i].tseg->seg.token_id);
          }
          return Status::Internal("SGL URMA completion reported failure (status=" +
                                 std::to_string(static_cast<int>(cr.status)) +
                                 ", completion_len=" + std::to_string(cr.completion_len) +
                                 ", user_ctx=0x" +
                                 std::to_string(static_cast<unsigned long long>(cr.user_ctx)) +
                                 ", num_sge=" + std::to_string(count) + ")");
        }
        return Status::OK();
      }
      std::this_thread::yield();
    }
    urma_jetty_cfg_t observed_cfg{};
    urma_jetty_attr_t observed_attr{};
    observed_attr.mask = JETTY_STATE;
    const urma_status_t query_status =
        urma_query_jetty(local_jetty_, &observed_cfg, &observed_attr);
    urma_cr_t flushed{};
    const int flushed_count = urma_flush_jetty(local_jetty_, 1, &flushed);
    std::string flush_info = ", flush_count=" + std::to_string(flushed_count);
    if (flushed_count > 0) {
      flush_info += ", flush_status=" + std::to_string(static_cast<int>(flushed.status)) +
                     ", flush_ctx=0x" +
                     std::to_string(static_cast<unsigned long long>(flushed.user_ctx));
    }
    return Status::Internal("timed out waiting for SGL completion (op=" +
                           std::string(write ? "WRITE" : "READ") +
                           ", num_sge=" + std::to_string(count) +
                           ", jetty_query_status=" +
                           std::to_string(static_cast<int>(query_status)) +
                           ", jetty_state=" +
                           std::to_string(static_cast<unsigned>(observed_attr.state)) +
                           flush_info + ")");
  }

  Status TransferDirect(bool write, std::uint64_t offset, void* data, std::size_t bytes) {
    if (!initialized_) return Status::InvalidState("CPU URMA transport is not initialized");
    if (data == nullptr || bytes == 0 || offset > remote_region_.length ||
        bytes > remote_region_.length - offset || bytes > std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument("invalid direct transfer range");
    }
    RegisteredHostBuffer* host = nullptr;
    host = FindRegisteredHostBuffer(data, bytes);
    if (host == nullptr) {
      Status status = RegisterHostBuffer(data, bytes, &host);
      if (!status.ok()) return status;
    }
    urma_sge_t local{};
    local.addr = reinterpret_cast<std::uint64_t>(data);
    local.len = static_cast<std::uint32_t>(bytes);
    local.tseg = host->segment;
    urma_sge_t remote{};
    remote.addr = remote_region_.base + offset;
    remote.len = static_cast<std::uint32_t>(bytes);
    remote.tseg = remote_seg_handle_;
    urma_jfs_wr_t wr{};
    wr.opcode = write ? URMA_OPC_WRITE : URMA_OPC_READ;
    wr.flag.bs.complete_enable = 1;
    wr.tjetty = remote_jetty_;
    wr.user_ctx = 0xD1CE;
    if (write) {
      wr.rw.src = {&local, 1};
      wr.rw.dst = {&remote, 1};
    } else {
      wr.rw.src = {&remote, 1};
      wr.rw.dst = {&local, 1};
    }
    urma_jfs_wr_t* bad = nullptr;
    const urma_status_t post_status = urma_post_jetty_send_wr(local_jetty_, &wr, &bad);
    if (post_status != URMA_SUCCESS) {
      return Status::Internal("urma_post_jetty_send_wr for direct host buffer failed (status=" +
                             std::to_string(static_cast<int>(post_status)) +
                             ", bad_wr=" +
                             std::to_string(reinterpret_cast<std::uintptr_t>(bad)) +
                             ", errno=" + std::to_string(errno) + ")");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      urma_cr_t cr{};
      const int polled = urma_poll_jfc(jfc_, 1, &cr);
      if (polled < 0) return Error("urma_poll_jfc for direct host buffer");
      if (polled == 1) {
        if (cr.status != URMA_CR_SUCCESS || cr.user_ctx != 0xD1CE) {
          return Status::Internal("direct host buffer URMA completion reported failure (status=" +
                                 std::to_string(static_cast<int>(cr.status)) +
                                 ", completion_len=" + std::to_string(cr.completion_len) +
                                 ", user_ctx=0x" +
                                 std::to_string(static_cast<unsigned long long>(cr.user_ctx)) +
                                 ")");
        }
        return Status::OK();
      }
      std::this_thread::yield();
    }
    urma_jetty_cfg_t observed_cfg{};
    urma_jetty_attr_t observed_attr{};
    observed_attr.mask = JETTY_STATE;
    const urma_status_t query_status =
        urma_query_jetty(local_jetty_, &observed_cfg, &observed_attr);
    return Status::Internal(
        std::string("timed out waiting for direct host buffer completion (op=") +
        (write ? "WRITE" : "READ") + ", offset=" + std::to_string(offset) +
        ", bytes=" + std::to_string(bytes) + ", jetty_query_status=" +
        std::to_string(static_cast<int>(query_status)) + ", jetty_state=" +
        std::to_string(static_cast<unsigned>(observed_attr.state)) + ")");
  }

  Status TransferBatch(bool write, std::uint64_t offset,
                       const CpuUrmaTransport::HostSge* buffers, std::size_t count) {
    if (!initialized_) return Status::InvalidState("CPU URMA transport is not initialized");
    std::unique_lock<std::mutex> submit_lock(batch_submit_mutex_);
    const std::size_t jetty_count = local_jettys_.size();
    // The NPU side currently exposes one target Jetty with a 1024-entry RQ.
    // Multiple CPU initiator Jettys parallelize that wave; they do not enlarge
    // the single remote receive queue.
    if (buffers == nullptr || count == 0U || offset > remote_region_.length ||
        jetty_count == 0U || count > kJettyDepth) {
      return Status::InvalidArgument("invalid batch transfer");
    }

    std::vector<std::size_t> queue_counts(jetty_count, 0U);
    for (std::size_t i = 0U; i < count; ++i) ++queue_counts[i % jetty_count];
    BatchJob& job = batch_workspace_;
    job.write = write;
    job.buffers = buffers;
    job.count = count;
    job.remote_offsets.resize(count);
    job.queues.resize(jetty_count);
    job.single_host_segment = nullptr;
    for (std::size_t jetty = 0U; jetty < jetty_count; ++jetty) {
      job.queues[jetty].local.resize(queue_counts[jetty]);
      job.queues[jetty].remote.resize(queue_counts[jetty]);
      job.queues[jetty].wrs.resize(queue_counts[jetty]);
    }

    std::uint64_t total = 0U;
    bool use_explicit_host_region = explicit_host_region_.segment != nullptr;
    for (std::size_t i = 0; i < count; ++i) {
      if (buffers[i].data == nullptr || buffers[i].bytes == 0U ||
          buffers[i].bytes > std::numeric_limits<std::uint32_t>::max() ||
          total > remote_region_.length - offset ||
          buffers[i].bytes > remote_region_.length - offset - total) {
        return Status::InvalidArgument("invalid batch entry");
      }
      if (use_explicit_host_region) {
        const auto& host = explicit_host_region_;
        const auto requested = reinterpret_cast<std::uintptr_t>(buffers[i].data);
        const auto base = reinterpret_cast<std::uintptr_t>(host.address);
        if (requested < base || buffers[i].bytes > host.bytes ||
            requested - base > host.bytes - buffers[i].bytes) {
          use_explicit_host_region = false;
        }
      }
      job.remote_offsets[i] = offset + total;
      total += buffers[i].bytes;
    }
    if (use_explicit_host_region) {
      job.single_host_segment = explicit_host_region_.segment;
    } else {
      job.host_segments.resize(count);
      for (std::size_t i = 0; i < count; ++i) {
        RegisteredHostBuffer* host = FindRegisteredHostBuffer(buffers[i].data, buffers[i].bytes);
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
    std::string post_error;
    if (wqes_per_doorbell_ != 0U) {
      // Double-buffered mode overlaps OpenMP WQE preparation for the next
      // doorbell wave with submission of the current wave.  A slot is reused
      // only after its completion CQEs have been drained.
      std::uint64_t pipeline_poll_ns = 0U;
      post_error = PrepareAndPostDoubleBuffered(
          job, &prepare_ns, &post_ns, &pipeline_poll_ns);
      // Polling is performed inside the pipeline helper and is accounted for
      // separately below; preserve the phase value for the caller.
      last_batch_timing_.poll_ns = pipeline_poll_ns;
    } else {
      const auto prepare_start = std::chrono::steady_clock::now();
      PrepareBatch(job);
      const auto prepare_end = std::chrono::steady_clock::now();
      const auto post_start = prepare_end;
      post_error = PostBatch(job);
      const auto post_end = std::chrono::steady_clock::now();
      prepare_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(prepare_end - prepare_start).count());
      post_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(post_end - post_start).count());
    }
    const auto poll_start = std::chrono::steady_clock::now();
    const std::string poll_error =
        (post_error.empty() && wqes_per_doorbell_ == 0U) ? PollBatch(job) : post_error;
    const auto poll_end = std::chrono::steady_clock::now();
    last_batch_timing_.prepare_ns = prepare_ns;
    last_batch_timing_.post_ns = post_ns;
    if (wqes_per_doorbell_ == 0U) {
      last_batch_timing_.poll_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(poll_end - poll_start).count());
    }
    const std::string transport_error = post_error.empty() ? poll_error : post_error;
    if (!transport_error.empty()) return Status::Internal(transport_error);
    return Status::OK();
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
    const urma_status_t post_status = urma_post_jetty_send_wr(local_jetty_, &wr, &bad);
    if (post_status != URMA_SUCCESS) {
      result = Status::Internal(
          "HBM direct URMA WRITE submission failed (status=" +
          std::to_string(static_cast<int>(post_status)) + ", bad_wr=" +
          std::to_string(reinterpret_cast<std::uintptr_t>(bad)) + ", errno=" +
          std::to_string(errno) + "); device memory may not be DMA-addressable");
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
            result = Status::Internal(
                "HBM direct URMA WRITE completion reported failure (status=" +
                std::to_string(static_cast<int>(cr.status)) +
                ", completion_len=" + std::to_string(cr.completion_len) +
                ", user_ctx=0x" +
                std::to_string(static_cast<unsigned long long>(cr.user_ctx)) + ")");
          }
          completed = true;
          break;
        }
        std::this_thread::yield();
      }
      if (!completed) {
        urma_jetty_cfg_t observed_cfg{};
        urma_jetty_attr_t observed_attr{};
        observed_attr.mask = JETTY_STATE;
        const urma_status_t query_status =
            urma_query_jetty(local_jetty_, &observed_cfg, &observed_attr);
        result = Status::Internal(
            "timed out waiting for HBM URMA WRITE completion (jetty_query_status=" +
            std::to_string(static_cast<int>(query_status)) + ", jetty_state=" +
            std::to_string(static_cast<unsigned>(observed_attr.state)) + ")");
      }
    }
    (void)urma_unregister_seg(device_seg);
    (void)urma_free_token_id(device_token_id);
    return result;
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
    jfc_ = nullptr;
    jfr_ = nullptr;
    local_jetty_ = nullptr;
    remote_jetty_ = nullptr;
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
  urma_target_seg_t* remote_seg_handle_{nullptr};
  urma_seg_t remote_seg_{};
  std::uint32_t eid_index_{0};
  std::uint32_t max_sge_{1};
  std::vector<urma_jfc_t*> jfcs_;
  std::vector<urma_jfr_t*> jfrs_;
  std::vector<urma_jetty_t*> local_jettys_;
  std::vector<urma_target_jetty_t*> remote_jettys_;
  std::mutex batch_submit_mutex_;
  BatchJob batch_workspace_;
  std::array<PipelineSlot, 2> pipeline_slots_{};
  std::uint32_t openmp_thread_count_{1U};
  std::size_t wqes_per_doorbell_{0U};
  CpuUrmaTransport::BatchTiming last_batch_timing_{};
  bool urma_initialized_{false};
  bool urma_owned_{false};
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
Status CpuUrmaTransport::Write(std::uint64_t offset, const void* source, std::size_t bytes) {
  return impl_->TransferDirect(true, offset, const_cast<void*>(source), bytes);
}
Status CpuUrmaTransport::WriteSgl(std::uint64_t offset, const HostSge* sources,
                                  std::size_t count) {
  return impl_->TransferSgl(true, offset, sources, count);
}
Status CpuUrmaTransport::RegisterDirectHostMemory(void* base, std::size_t bytes) {
  return impl_->RegisterDirectHostMemory(base, bytes);
}
Status CpuUrmaTransport::WriteDeviceMemory(std::uint64_t offset, const void* device_source,
                                           std::size_t bytes) {
  return impl_->TransferDeviceWrite(offset, device_source, bytes);
}
Status CpuUrmaTransport::Read(std::uint64_t offset, void* destination, std::size_t bytes) {
  return impl_->TransferDirect(false, offset, destination, bytes);
}
Status CpuUrmaTransport::WriteBatch(std::uint64_t offset, const HostSge* sources,
                                    std::size_t count) {
  return impl_->TransferBatch(true, offset, sources, count);
}
Status CpuUrmaTransport::ReadBatch(std::uint64_t offset, const HostSge* destinations,
                                   std::size_t count) {
  return impl_->TransferBatch(false, offset, destinations, count);
}
Status CpuUrmaTransport::ReadSgl(std::uint64_t offset, const HostSge* destinations,
                                 std::size_t count) {
  return impl_->TransferSgl(false, offset, destinations, count);
}
Status CpuUrmaTransport::Shutdown() { return impl_->Shutdown(); }
bool CpuUrmaTransport::initialized() const { return impl_->initialized_; }
std::uint64_t CpuUrmaTransport::region_bytes() const { return impl_->remote_region_.length; }
std::uint32_t CpuUrmaTransport::max_sge() const { return impl_->max_sge_; }
CpuUrmaTransport::BatchTiming CpuUrmaTransport::last_batch_timing() const {
  return impl_->last_batch_timing_;
}

}  // namespace kvaf
