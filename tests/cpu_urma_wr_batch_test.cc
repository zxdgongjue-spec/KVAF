#include "wr_batch.h"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using namespace kvaf::cpu_urma_detail;
void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

struct Fixture {
  SglBatchJob job;
  std::array<urma_target_jetty_t, 4> targets{};
  std::array<urma_target_jetty_t*, 4> target_ptrs{};
  std::array<urma_target_seg_t, 2> segments{};
  std::size_t generation{0};

  Fixture() {
    job.queues.resize(4);
    for (std::size_t i = 0; i < 4; ++i) target_ptrs[i] = &targets[i];
    job.remote_jettys = target_ptrs.data();
  }

  void Transfer(std::size_t count, std::size_t sgl, bool write) {
    ++generation;
    // Vary lengths, addresses and segment handles while reusing the same WRs.
    const std::size_t bytes = generation % 2 ? 8 : 16;
    std::vector<unsigned char> host(count * bytes), remote(count * 32, 0xEE);
    std::vector<unsigned char> expected(remote.size(), 0xEE);
    for (std::size_t i = 0; i < host.size(); ++i) host[i] = (i + generation * 7) % 251;
    const auto original = host;
    std::vector<kvaf::CpuUrmaTransport::HostSge> buffers(count);
    job.remote_offsets.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      buffers[i] = {host.data() + i * bytes, bytes};
      job.remote_offsets[i] = (count - 1 - i) * 32;
      std::memcpy(expected.data() + job.remote_offsets[i], buffers[i].data, bytes);
    }
    if (!write) {
      remote = expected;
      std::fill(host.begin(), host.end(), 0);
    }
    std::vector<SglGroup> groups;
    if (sgl > 1) Check(PlanWriteScatterGroups(buffers.data(), job.remote_offsets.data(),
        count, sgl, remote.size(), &groups).ok(), "scatter plan failed");
    job.groups = sgl > 1 ? groups.data() : nullptr;
    job.buffers = buffers.data();
    job.write = write;
    job.remote_sgl_size = sgl;
    job.remote_base = reinterpret_cast<std::uintptr_t>(remote.data());
    job.remote_segment = &segments[generation % 2];
    job.single_host_segment = generation % 2 ? &segments[0] : nullptr;
    job.host_segments.assign(count, &segments[1]);
    const auto wr_count = sgl > 1 ? groups.size() : count;
    for (std::size_t j = 0; j < 4; ++j) {
      auto& queue = job.queues[j];
      queue.Resize(wr_count / 4 + (j < wr_count % 4), write, sgl);
      for (std::size_t p = 0; p < queue.wrs.size(); ++p) PrepareSglBatchEntry(job, j, p);
      queue.template_valid = true;
      std::size_t traversed = 0, completions = 0;
      for (auto* wr = queue.wrs.empty() ? nullptr : queue.wrs.data(); wr; wr = wr->next) {
        ++traversed;
        completions += wr->flag.bs.complete_enable;
        Check(wr->tjetty == target_ptrs[j], "wrong target Jetty");
        Check(wr->opcode == (write ? URMA_OPC_WRITE : URMA_OPC_READ), "stale opcode");
        Check(wr->flag.bs.complete_enable == (wr->next == nullptr), "stale completion marker");
        Check(wr->flag.bs.comp_order == (write ? 0U : 1U), "stale completion ordering");
        Check(wr->flag.bs.place_order == (write ? 0U :
            (wr->next ? URMA_RELAX_ORDER : URMA_STRONG_ORDER)), "stale placement ordering");
        const auto& local = write ? wr->rw.src : wr->rw.dst;
        const auto& distant = write ? wr->rw.dst : wr->rw.src;
        Check(local.num_sge == 1, "local SGE count");
        auto* source = reinterpret_cast<SglSgeAbi*>(local.sge);
        auto* destinations = reinterpret_cast<SglSgeAbi*>(distant.sge);
        Check(source->tseg == (job.single_host_segment ? job.single_host_segment : &segments[1]),
              "stale local segment");
        std::size_t offset = 0;
        for (std::size_t i = 0; i < distant.num_sge; ++i) {
          const auto& destination = destinations[i];
          Check(destination.tseg == job.remote_segment, "stale remote segment");
          Check(offset + destination.len <= source->len, "SGE lengths disagree");
          auto* local_address = reinterpret_cast<void*>(source->addr + offset);
          auto* remote_address = reinterpret_cast<void*>(destination.addr);
          if (write) std::memcpy(remote_address, local_address, destination.len);
          else std::memcpy(local_address, remote_address, destination.len);
          offset += destination.len;
        }
        Check(offset == source->len, "missing remote SGE");
      }
      Check(traversed == queue.wrs.size(), "truncated chain reused");
      Check(completions == (queue.wrs.empty() ? 0U : 1U), "completion count");
      // Simulate a previous post splitting the chain into smaller chunks.
      for (auto& wr : queue.wrs) wr.next = nullptr;
    }
    Check(write ? remote == expected : host == original, "WR replay corrupted payload or gaps");
  }
};
}  // namespace

int main() {
  try {
    Fixture fixture;
    fixture.Transfer(16, 1, true);
    fixture.Transfer(16, 1, true);  // Cached descriptors, changed data and cut tails.
    fixture.Transfer(16, 1, false);
    fixture.Transfer(16, 1, true);  // READ ordering must not survive into WRITE.
    fixture.Transfer(16, 3, true);
    fixture.Transfer(17, 3, true);  // Same queue sizes, different final SGE count.
    fixture.Transfer(15, 3, true);
    fixture.Transfer(32769, 1, true);  // Force storage growth and new pointers.
    fixture.Transfer(1, 1, true);      // Empty queues, moved completion marker.
    fixture.Transfer(5, 1, true);
    fixture.job = {};                 // Shutdown/reinitialization invalidates templates.
    fixture.job.queues.resize(4);
    fixture.job.remote_jettys = fixture.target_ptrs.data();
    fixture.Transfer(5, 1, true);
    std::cout << "CPU URMA WR reuse tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
