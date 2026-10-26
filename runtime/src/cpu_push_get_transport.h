#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "cpu_push_get_layout.h"
#include "cpu_urma_channel.h"
#include "peer_protocol.h"
#include "status.h"

namespace kvaf::cpu_push {
// One CPU worker exclusively owns these queues and their CQ polling.
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
  Status Shutdown();
  const Request& request() const { return *reinterpret_cast<const Request*>(memory_); }
  const Layout& layout() const { return layout_; }
 private:
  struct Queue {
    cpu_runtime_detail::UrmaChannel channel;
    std::vector<urma::Sge> local, remote;
    std::vector<urma_jfs_wr_t> wrs;
    std::uint64_t sequence{}, pending{};
    bool poisoned{};
  };
  Status Transfer(bool write, std::uint64_t offset, std::uint64_t bytes);
  Status Post(Queue&);
  Status Wait(Queue&);
  void Add(Queue&, urma_opcode_t, std::uint64_t remote, std::uint32_t bytes,
           std::uint32_t first_sge, std::uint32_t sges);
  void ResetRequests(Queue&);
  std::array<Queue, kJettyCount> queues_;
  urma_context_t* context_{};
  urma_target_seg_t* source_{};
  urma_target_seg_t* remote_{};
  urma_target_seg_t* control_{};
  urma_token_id_t* control_token_{};
  std::uint8_t* memory_{};
  std::uint64_t hbm_base_{};
  std::uint32_t max_sge_{}, max_bytes_{}, workers_{}, token_bytes_{};
  Layout layout_;
};
}  // namespace kvaf::cpu_push
