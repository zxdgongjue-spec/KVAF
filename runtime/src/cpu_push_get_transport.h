#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu_push_get_layout.h"
#include "cpu_urma_channel.h"
#include "cpu_urma/indexed_async_write_tracker.h"
#include "peer_protocol.h"
#include "status.h"

namespace kvaf::cpu_push {
// One service thread owns and submits the four prepared Jetty chains.
class Transport {
 public:
  ~Transport();
  Status Initialize(urma_context_t*, urma_target_seg_t* source,
                    const peer::OpenCpuPushGetRequest&, urma::GetTpList,
                    urma::ImportJettyEx, std::uint32_t psn, peer::OpenCpuPushGetResponse*);
  Status ReadRequest(std::uint64_t* sequence);
  Status ReadDescriptor(std::uint64_t sequence);
  Status WriteBatch();
  Status Complete(std::uint64_t sequence, Result result);
  void PrintProfile() const;
  Status BenchmarkWrite(std::uint8_t* source_memory, std::uint64_t source_base,
                        std::uint64_t source_capacity, const WriteBenchmarkOptions&,
                        WriteBenchmarkResult*, const WriteBenchmarkDiagnosticOptions* = nullptr,
                        WriteBenchmarkDiagnostics* = nullptr);
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
    struct Source { std::uint32_t local, row, offset; };
    struct Marker { std::uint64_t* value; std::uint32_t progress; };
    std::vector<urma::Sge> local, remote;
    std::vector<urma_jfs_wr_t> wrs;
    std::vector<Source> sources;
    std::vector<Marker> markers;
    // Whole-batch, unsplit tokens have one SGE per contiguous descriptor row.
    std::uint32_t first_row{UINT32_MAX};
  };
  struct PreparedWrite {
    std::array<PreparedQueue, kJettyCount> queues;
    std::array<Tracker::Fence, kJettyCount> fences;
    std::size_t fence_count{};
    Tracker::Handle handle;
  };
  Status PrepareWrites(std::uint32_t count);
  void PrepareCompletions();
  void AppendPrepared(PreparedQueue&, std::uint32_t queue, std::uint64_t remote,
                      std::uint32_t bytes, std::uint32_t sges, bool ordered = false);
  void SealPrepared(PreparedWrite&);
  Status BeginPrepared(PreparedWrite&, std::uint64_t id);
  Status PostPrepared(PreparedWrite&, bool refresh_sources = false,
                      std::uint64_t* first_post_ns = nullptr);
  Status PollPrepared(std::uint64_t id, Profile* profile = nullptr);
  void RefreshPrepared(PreparedQueue&);
  Status ReadControl(std::uint64_t offset, std::uint64_t bytes);
  Status Post(Queue&);
  Status Wait(Queue&);
  void AddControlRead(Queue&, std::uint64_t remote, std::uint32_t bytes);
  void ResetRequests(Queue&);
  std::array<Queue, kJettyCount> queues_;
  std::vector<std::unique_ptr<PreparedWrite>> prepared_;
  PreparedWrite done_;
  PreparedWrite profiled_done_;
  Tracker tracker_{kJettyCount, 8192};
  std::uint32_t prepared_count_{UINT32_MAX}, next_poll_queue_{};
  std::uint32_t descriptor_prefetch_count_{};
  bool prepared_batch_done_only_{};
  urma_context_t* context_{};
  urma_target_seg_t* source_{};
  urma_target_seg_t* remote_{};
  urma_target_seg_t* control_{};
  urma_token_id_t* control_token_{};
  std::uint8_t* memory_{};
  std::uint64_t hbm_base_{}, session_{};
  std::uint32_t max_sge_{}, max_bytes_{}, workers_{}, token_bytes_{};
  std::uint32_t post_batch_wrs_{128};
  Layout layout_;
};
}  // namespace kvaf::cpu_push
