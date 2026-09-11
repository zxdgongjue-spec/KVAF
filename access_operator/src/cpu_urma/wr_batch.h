#pragma once

#include "cpu_urma/sgl_batch_plan.h"
#include <urma_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kvaf::cpu_urma_detail {

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
struct SglBatchQueue {
  // Each WR has exactly one contiguous local SGE.
  std::vector<SglSgeAbi> local;
  std::vector<SglSgeAbi> remote;
  std::vector<urma_jfs_wr_t> wrs;
  bool template_valid{false};
  bool template_write{false};
  std::size_t template_remote_sgl_size{0};

  void Resize(std::size_t count, bool write, std::size_t remote_sgl_size) {
    if (wrs.size() != count || template_write != write ||
        template_remote_sgl_size != remote_sgl_size) template_valid = false;
    local.resize(count);
    remote.resize(count * remote_sgl_size);
    wrs.resize(count);
    template_write = write;
    template_remote_sgl_size = remote_sgl_size;
  }
};

struct SglBatchJob {
  bool write{false};
  std::uint64_t remote_base{0};
  urma_target_seg_t* remote_segment{nullptr};
  urma_target_jetty_t* const* remote_jettys{nullptr};
  const CpuUrmaTransport::HostSge* buffers{nullptr};
  std::size_t count{0U};
  std::size_t remote_sgl_size{1U};
  std::size_t wr_count{0U};
  const cpu_urma_detail::SglGroup* groups{nullptr};
  urma_target_seg_t* single_host_segment{nullptr};
  std::vector<SglBatchQueue> queues;
  std::vector<std::uint64_t> remote_offsets;
  std::vector<urma_target_seg_t*> host_segments;
  std::chrono::steady_clock::time_point deadline{};
};

inline void PrepareSglBatchEntry(SglBatchJob& job, std::size_t jetty, std::size_t position) {
  const std::size_t wr_index = position * job.queues.size() + jetty;
  const std::size_t first = job.groups != nullptr
      ? job.groups[wr_index].first - job.groups[0].first
      : wr_index;
  const std::size_t row_count = job.groups != nullptr
      ? job.groups[wr_index].count
      : 1U;
  const bool scatter = job.remote_sgl_size > 1U;
  SglBatchQueue& queue = job.queues[jetty];
  SglSgeAbi* local = &queue.local[position];
  const std::uint64_t bytes = scatter ? job.groups[wr_index].bytes : job.buffers[first].bytes;
  if (!queue.template_valid) *local = {};
  local->addr = reinterpret_cast<std::uint64_t>(job.buffers[first].data);
  local->len = static_cast<std::uint32_t>(bytes);
  local->tseg = job.single_host_segment != nullptr
      ? job.single_host_segment : job.host_segments[first];
  SglSgeAbi* remote = queue.remote.data() + position * job.remote_sgl_size;
  const std::size_t remote_count = scatter ? row_count : 1U;
  for (std::size_t i = 0U; i < remote_count; ++i) {
    if (!queue.template_valid) remote[i] = {};
    remote[i].addr = job.remote_base + job.remote_offsets[first + i];
    remote[i].len = static_cast<std::uint32_t>(
        scatter ? job.buffers[first + i].bytes : bytes);
    remote[i].tseg = job.remote_segment;
  }
  urma_jfs_wr_t& wr = queue.wrs[position];
  if (!queue.template_valid) {
    wr = {};
    wr.opcode = job.write ? URMA_OPC_WRITE : URMA_OPC_READ;
    const bool completes_queue = position + 1U == queue.wrs.size();
    // WRITE ordering stays zero; READ keeps its existing ordering contract.
    if (!job.write) {
      wr.flag.bs.place_order = completes_queue ? URMA_STRONG_ORDER : URMA_RELAX_ORDER;
      wr.flag.bs.comp_order = 1U;
    }
    wr.flag.bs.complete_enable = completes_queue ? 1U : 0U;
    wr.tjetty = job.remote_jettys[jetty];
    wr.user_ctx = 0x5A1B0000ULL + (static_cast<std::uint64_t>(jetty) << 32U) + position;
    auto& local_sg = job.write ? wr.rw.src : wr.rw.dst;
    auto& remote_sg = job.write ? wr.rw.dst : wr.rw.src;
    local_sg.sge = reinterpret_cast<urma_sge_t*>(local);
    local_sg.num_sge = 1U;
    remote_sg.sge = reinterpret_cast<urma_sge_t*>(remote);
  }
  // A scatter group's row count can change even when queue sizes do not.
  auto& remote_sg = job.write ? wr.rw.dst : wr.rw.src;
  remote_sg.num_sge = static_cast<std::uint32_t>(remote_count);
  // Dispatch cuts chunk tails; restore the full chain on every preparation.
  wr.next = position + 1U < queue.wrs.size()
                ? &queue.wrs[position + 1U]
                : nullptr;
}


}  // namespace kvaf::cpu_urma_detail
