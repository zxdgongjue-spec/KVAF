#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu_push_get_layout.h"
#include "cpu_service/urma/cpu_urma_channel.h"
#include "cpu_urma/indexed_async_write_tracker.h"
#include "peer_protocol.h"
#include "status.h"

namespace kvaf::cpu_push {
// One service thread owns and submits the four prepared Jetty chains.
class Transport {
 public:
  ~Transport();
  Status Initialize(urma_context_t*, urma_target_seg_t* source, const peer::OpenCpuPushGetRequest&,
                    urma::GetTpList, urma::ImportJettyEx, std::uint32_t psn,
                    peer::OpenCpuPushGetResponse*, std::uint64_t canonical_base = 0,
                    const peer::OpenCpuSglPushRequest* push = nullptr, peer::OpenCpuSglPushResponse* mailbox = nullptr);
  Status ReadRequest(std::uint64_t* sequence);
  Status ReadDescriptor(std::uint64_t sequence);
  Status WriteBatch(const std::atomic<bool>* stop = nullptr);
  Status Complete(std::uint64_t sequence, Result result);
  void PrintProfile() const;
  Status Shutdown();
  const Request& request() const { return *reinterpret_cast<const Request*>(memory_); }
  const Layout& layout() const { return layout_; }
  Result batch_result() const { return batch_result_; }
 private:
  struct Profile {
    bool enabled{};
    std::uint64_t header_read_ns{}, address_read_ns{}, write_ns{}, post_ns{}, cq_wait_ns{}, done_ns{};
    std::uint64_t first_post_ns{}, last_cqe_ns{};
    // CPU-only payload diagnostics; these do not change the device/wire ABI.
    std::uint64_t notify_read_ns{}, cpu_start_ns{}, cpu_total_ns{};
    std::uint64_t address_list_read_ns{}, address_validate_ns{}, output_desc_read_ns{}, output_import_ns{};
    std::uint64_t prepare_writes_ns{}, target_update_ns{}, source_sge_update_ns{}, data_begin_ns{};
    std::uint64_t payload_post_ns{}, payload_cq_wait_ns{};
    std::uint64_t output_unimport_ns{}, done_post_ns{}, done_cq_wait_ns{};
    std::uint64_t head_submit_ns{}, head_cq_wait_ns{}, suffix_wait_ns{}, suffix_read_ns{};
    std::uint64_t suffix_submit_ns{}, suffix_cq_wait_ns{};
    std::uint64_t chunk_ready_ns{}, chunk_read_ns{}, chunk_submit_ns{}, chunk_cq_wait_ns{};
    std::uint64_t no_ready_inflight_ns{}, no_ready_idle_ns{};
    std::uint32_t chunks{}, progress_reads{}, max_inflight{};
    std::uint32_t data_wrs{}, ready_wrs{}, post_calls{}, waves{}, tail_cqes{}, cpu_threads{};
    bool plan_reused{};
  } profile_;
  struct Queue {
    cpu_runtime_detail::UrmaChannel channel;
    std::vector<urma::Sge> local, remote;
    std::vector<urma_jfs_wr_t> wrs;
    std::uint64_t sequence{}, pending{};
    bool poisoned{};
  };
  using Tracker = cpu_urma_detail::IndexedAsyncWriteTracker;
  struct PreparedQueue {
    struct Post { std::uint32_t wr, first_sge, end_sge; };
    struct Source { std::uint32_t local, row, offset; };
    struct SourceRun { std::uint32_t local, row, count; };
    struct Marker { std::uint64_t* value; std::uint32_t progress, local; };
    std::vector<urma::Sge> local, remote;
    std::vector<urma_jfs_wr_t> wrs;
    std::vector<Post> posts;
    struct Destination { std::uint32_t wr; std::uint64_t offset; };
    std::vector<Destination> destinations;
    std::vector<Source> sources;
    std::vector<SourceRun> source_runs;
    std::vector<Marker> markers;
  };
  struct PreparedWrite {
    std::array<PreparedQueue, kJettyCount> queues;
    std::array<Tracker::Fence, kJettyCount> fences;
    std::size_t fence_count{};
    Tracker::Handle handle;
  };
  Status PrepareWrites(std::uint32_t count);
  void RetargetWrites();
  void AppendPrepared(PreparedQueue&, std::uint32_t queue, std::uint64_t remote,
                      std::uint32_t bytes, std::uint32_t sges, bool ordered = false);
  void SealPrepared(PreparedWrite&);
  Status BeginPrepared(PreparedWrite&, std::uint64_t id, bool* submitted = nullptr);
  Status PostPrepared(PreparedWrite&, bool refresh_sources = false, Profile* profile = nullptr,
                      std::uint64_t* first_post_ns = nullptr);
  Status WritePayloadOnly();
  Status WritePipeline();
  Status WriteChunks(const std::atomic<bool>* stop = nullptr);
  Status WritePushed(const std::atomic<bool>* stop);
  Status ValidateSources(std::uint32_t begin, std::uint32_t end);
  Status BeginControlRead(std::uint64_t offset, std::uint64_t bytes);
  Status PollPrepared(std::uint64_t id, Profile* profile = nullptr);
  Status PollAnyPrepared(std::uint64_t* completed, Profile* profile, bool wait);
  void RefreshPrepared(PreparedQueue&, const PreparedQueue::Post&);
  Status Transfer(bool write, std::uint64_t offset, std::uint64_t bytes);
  Status Post(Queue&);
  Status Wait(Queue&);
  void Add(Queue&, urma_opcode_t, std::uint64_t remote, std::uint32_t bytes,
           std::uint32_t first_sge, std::uint32_t sges);
  void ResetRequests(Queue&);
  std::array<Queue, kJettyCount> queues_;
  Queue control_queue_;
  std::vector<std::unique_ptr<PreparedWrite>> prepared_;
  std::vector<Range> chunk_plans_;
  PreparedWrite done_;
  Tracker tracker_{kJettyCount, 8192};
  std::uint32_t prepared_count_{UINT32_MAX}, next_poll_queue_{};
  std::uint32_t head_count_{}, head_waves_{}, head_rows_{kChunkRows};
  urma_context_t* context_{};
  urma_target_seg_t* source_{};
  urma_target_seg_t* remote_{};
  urma_target_seg_t* output_remote_{};
  OutputTarget output_;
  bool device_push_{}, cooperative_{};
  bool direct_output_{};
  bool payload_only_{};
  bool pipeline_{};
  bool chunked_{};
  Result batch_result_{Result::kOk};
  CpuSglWriteSample payload_sample_{};
  urma_target_seg_t* control_{};
  urma_token_id_t* control_token_{};
  std::uint8_t* memory_{};
  std::uint64_t hbm_base_{}, source_base_{}, source_bytes_{};
  std::uint32_t max_sge_{}, max_bytes_{}, workers_{}, token_bytes_{};
  Layout layout_;
};
}  // namespace kvaf::cpu_push
