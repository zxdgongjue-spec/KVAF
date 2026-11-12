#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu_gather_executor.h"
#include "cpu_push_get_layout.h"
#include "cpu_urma_channel.h"
#include "cpu_urma/indexed_async_write_tracker.h"
#include "peer_protocol.h"
#include "status.h"

namespace kvaf::cpu_push::aiv {
// One service thread owns and submits the four prepared Jetty chains.
class Transport {
 public:
  ~Transport();
  Status ConfigureGather(void* memory, std::uint64_t base, std::size_t threads,
                         std::size_t min_bytes, bool affinity, peer::GatherWaitMode mode);
  void BindWorker() const { if (gather_) gather_->BindWorker(0); }
  Status Initialize(urma_context_t*, urma_target_seg_t* source,
                    const peer::OpenCpuPushGetRequest&, urma::GetTpList,
                    urma::ImportJettyEx, std::uint32_t psn, peer::OpenAivCpuPushGetResponse*);
  Status ReadRequest(std::uint64_t* sequence);
  Status ReadDescriptor(std::uint64_t sequence);
  Status WriteBatch(const std::atomic<bool>* stop = nullptr);
  Status Complete(std::uint64_t sequence, Result result);
  void PrintProfile() const;
  Status Shutdown();
  const Request& request() const { return *reinterpret_cast<const Request*>(memory_); }
  const Layout& layout() const { return layout_; }
  std::uint64_t session() const { return session_; }
 private:
  friend struct TransportTest;
  struct Profile {
    bool enabled{};
    std::uint64_t header_read_ns{}, address_read_ns{}, write_ns{}, post_ns{}, cq_wait_ns{}, done_ns{};
    std::uint64_t first_post_ns{}, last_cqe_ns{};
    std::uint64_t gather_prepare_ns{}, gather_copy_ns{}, suffix_wait_ns{};
    std::uint64_t gather_join_wait_ns{}, slowest_elapsed_ns{}, slowest_start_delay_ns{};
    std::uint64_t slowest_ready_delay_ns{};
    peer::CpuGatherThreadProfile slowest_thread{};
    std::uint32_t slowest_worker{};
    std::uint32_t data_wrs{}, ready_wrs{}, post_calls{}, waves{}, tail_cqes{}, cpu_threads{};
    bool plan_reused{};
  } profile_;
  struct GatherProfileSample {
    std::uint64_t sequence;
    std::uint32_t rows;
    bool pipeline, gather_pipeline, interleaved;
    Profile profile;
  };
  std::vector<GatherProfileSample> gather_profile_samples_;
  std::size_t gather_profile_dropped_{};
  struct Queue {
    cpu_runtime_detail::UrmaChannel channel;
    bool poisoned{};
  };
  using Tracker = cpu_urma_detail::IndexedAsyncWriteTracker;
  struct PreparedQueue {
    struct Source { std::uint32_t local, row, offset; };
    struct SourceRun { std::uint32_t local, row, count; };
    struct Marker { std::uint64_t* value; std::uint32_t progress, local; };
    struct Post { std::uint32_t wr, first_sge, end_sge; };
    std::vector<urma::Sge> local, remote;
    std::vector<urma_jfs_wr_t> wrs;
    std::vector<Source> sources;
    // One sequential refresh per contiguous payload span; Ready SGEs are
    // deliberately excluded. Fragmented tokens retain the exact Source map.
    std::vector<SourceRun> source_runs;
    std::vector<Marker> markers;
    std::vector<Post> posts;
    // Whole-batch, unsplit tokens have one SGE per contiguous descriptor row.
    std::uint32_t first_row{UINT32_MAX};
  };
  struct PreparedWrite {
    std::array<PreparedQueue, kJettyCount> queues;
    std::array<Tracker::Fence, kJettyCount> fences;
    std::size_t fence_count{};
    Tracker::Handle handle;
    std::size_t ready_waves{};
    bool suffix{}, gather_before_post{};
  };
  Status EnsureGatherStaging(std::uint32_t count);
  std::uint32_t FirstGatherChunkRows(std::uint32_t count) const;
  Status GatherRows(std::uint32_t count, const peer::CpuGatherExecutor::WaveReady& on_wave = {},
                    const peer::CpuGatherExecutor::FixedSuffix* suffix = nullptr);
  void FlushGatherProfiles();
  Status PrepareWrites(std::uint32_t count);
  void PrepareCompletions();
  void AppendPrepared(PreparedQueue&, std::uint32_t queue, std::uint64_t remote,
                      std::uint32_t bytes, std::uint32_t sges, bool ordered = false);
  void SealPrepared(PreparedWrite&);
  Status BeginPrepared(PreparedWrite&, std::uint64_t id, Profile* drain_profile = nullptr,
                       bool drain_for_credits = false);
  Status PostPrepared(PreparedWrite&, bool refresh_sources = false,
                      std::uint64_t* first_post_ns = nullptr);
  Status PollPrepared(std::uint64_t id, Profile* profile = nullptr);
  Status DrainPrepared(Profile* profile = nullptr);
  Status WaitSuffix(const std::atomic<bool>* stop);
  void RefreshPrepared(PreparedQueue&, const PreparedQueue::Post&);
  std::array<Queue, kJettyCount> queues_;
  std::vector<std::unique_ptr<PreparedWrite>> prepared_;
  std::vector<GatherWave> gather_waves_;
  std::size_t gather_suffix_wave_{};
  PreparedWrite done_;
  PreparedWrite profiled_done_;
  Tracker tracker_{kJettyCount, 8192};
  std::uint32_t prepared_count_{UINT32_MAX}, next_poll_queue_{};
  bool prepared_batch_done_only_{};
  bool gather_interleave_{}, prepared_interleaved_{};
  std::uint32_t prepared_head_rows_{};
  urma_context_t* context_{};
  urma_target_seg_t* source_{};
  urma_target_seg_t* remote_{};
  urma_target_seg_t* control_{};
  urma_token_id_t* control_token_{};
  std::uint8_t* memory_{};
  std::unique_ptr<peer::CpuGatherExecutor> gather_;
  urma_target_seg_t* gather_segment_{};
  urma_token_id_t* gather_token_{};
  std::uint8_t* gather_staging_{};
  std::uint64_t gather_staging_bytes_{}, source_bytes_{};
  std::uint64_t hbm_base_{}, session_{};
  std::uint32_t max_sge_{}, max_bytes_{}, max_gather_write_bytes_{}, workers_{}, token_bytes_{};
  std::uint32_t token_value_{};
  std::uint32_t post_batch_wrs_{128};
  Layout layout_;
};
}  // namespace kvaf::cpu_push::aiv
