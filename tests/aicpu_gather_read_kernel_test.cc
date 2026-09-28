#include "aicpu_urma/gather_read.h"
#include "aicpu_urma/wqe.h"
#include "batch_get_abi.h"
#include "cpu_gather.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

extern "C" std::uint32_t KvafAicpuUrmaGatherRead(kvaf::aicpu_urma::GatherReadArgs* args);

int main() {
  std::array<std::uint8_t, 256> remote{};
  std::array<std::uint8_t, 256> output{};
  output.fill(0xa5U);
  std::array<std::uint32_t, 1024> sq{};
  std::array<std::uint32_t, 512> cq{};
  std::uint32_t sq_doorbell = 0U, cq_doorbell = 0U;
  const auto base = reinterpret_cast<std::uint64_t>(remote.data());
  const std::vector<kvaf::peer::GatherSpan> spans{
      {base + 80U, 16U, 0U}, {base, 16U, 0U}, {base + 48U, 16U, 0U}};
  const std::array<kvaf::GatherOutputSpan, 9> output_spans{{
      {128U, 5U, 0U}, {0U, 5U, 0U}, {112U, 5U, 0U},
      {16U, 5U, 0U}, {96U, 5U, 0U}, {32U, 5U, 0U},
      {80U, 5U, 0U}, {48U, 5U, 0U}, {64U, 8U, 0U}}};
  for (std::uint32_t index = 0U; index < 128U; ++index)
    remote[index] = static_cast<std::uint8_t>(index);
  KvafBatchGetDirectUrmaEntity entity{};
  entity.magic = 0x4b564155U;
  entity.version = 1U;
  entity.remote_host_ddr_address = base;
  entity.remote_host_ddr_size = remote.size();
  entity.remote_token_id = 3U;
  entity.remote_token_value = 4U;
  entity.local_hbm_address = reinterpret_cast<std::uint64_t>(output.data());
  entity.local_hbm_size = output.size();
  entity.local_token_id = 7U;
  entity.sq_va = reinterpret_cast<std::uint64_t>(sq.data());
  entity.sq_db_va = reinterpret_cast<std::uint64_t>(&sq_doorbell);
  entity.sq_depth = 64U;
  entity.wqe_size = 64U;
  entity.cq_va = reinterpret_cast<std::uint64_t>(cq.data());
  entity.cq_db_va = reinterpret_cast<std::uint64_t>(&cq_doorbell);
  entity.cq_depth = 64U;
  entity.cqe_size = 32U;
  entity.tp_id = 5U;
  KvafBatchGetQueueState queue{};
  queue.magic = 0x4b564151U;
  queue.initialized = 1U;
  queue.sq_producer_index = 60U;
  sq_doorbell = 60U;
  kvaf::aicpu_urma::GatherReadStatus status{};
  kvaf::aicpu_urma::GatherReadArgs args{};
  args.channel_entity = reinterpret_cast<std::uint64_t>(&entity);
  args.queue_state = reinterpret_cast<std::uint64_t>(&queue);
  args.core_status = reinterpret_cast<std::uint64_t>(&status);
  args.output = entity.local_hbm_address;
  args.remote_staging = base + 128U;
  args.remote_command = base + 208U;
  args.remote_notify = base + 192U;
  args.remote_ack = base + 200U;
  args.output_spans = reinterpret_cast<std::uint64_t>(output_spans.data());
  args.output_extent = 133U;
  args.output_span_count = output_spans.size();
  args.sgl_size = kvaf::kGatherMaxSglSize + 2U;
  args.sequence = 1U;
  args.bytes = 48U;

  std::atomic<bool> stopped{false};
  std::thread hardware([&] {
    std::uint32_t producer = 60U;
    std::uint32_t completion = 0U;
    while (producer < 68U && !stopped.load(std::memory_order_acquire)) {
      if (__atomic_load_n(&sq_doorbell, __ATOMIC_ACQUIRE) == producer) {
        std::this_thread::yield();
        continue;
      }
      std::array<std::uint32_t, kvaf::aicpu_urma::kMaxScatterWqebbs * 16U> wqe{};
      const std::uint32_t* first = sq.data() + (producer % entity.sq_depth) * 16U;
      const auto opcode = (first[1] >> 8U) & 0xffU;
      const std::uint32_t blocks = opcode == kvaf::aicpu_urma::kWriteWithNotifyOpcode
                                       ? 2U : kvaf::aicpu_urma::ScatterWqebbs(first[2] >> 24U);
      for (std::uint32_t block = 0U; block < blocks; ++block)
        std::memcpy(wqe.data() + block * 16U,
                    sq.data() + ((producer + block) % entity.sq_depth) * 16U, 64U);
      const auto* words = wqe.data();
      const auto address = [](std::uint32_t low, std::uint32_t high) {
        return static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32U);
      };
      if (opcode == kvaf::aicpu_urma::kWriteWithNotifyOpcode) {
        assert(words[20] == 8U && words[21] == entity.local_token_id);
        const auto local = address(words[22], words[23]);
        std::memcpy(reinterpret_cast<void*>(address(words[10], words[11])),
                    reinterpret_cast<void*>(local), 8U);
        const auto sequence = address(words[16], words[17]);
        std::memcpy(reinterpret_cast<void*>(address(words[14], words[15])),
                    &sequence, 8U);
        kvaf::peer::GatherIntoStaging(remote.data(), base, args.remote_staging, spans);
        const kvaf::peer::GatherCpuProfile cpu{11U, 22U, 33U, 44U, 55U};
        std::memcpy(remote.data() + 216U, &cpu, sizeof(cpu));
        std::memcpy(reinterpret_cast<void*>(args.remote_ack), &sequence, 8U);
      } else {
        assert(opcode == kvaf::aicpu_urma::kReadOpcode);
        std::uint64_t remote_address = address(words[10], words[11]);
        const auto sge_count = words[2] >> 24U;
        assert(sge_count <= kvaf::kGatherMaxSglSize);
        for (std::uint32_t index = 0U; index < sge_count; ++index) {
          const auto offset = 12U + index * 4U;
          assert(words[offset + 1U] == entity.local_token_id);
          std::memcpy(reinterpret_cast<void*>(address(words[offset + 2U], words[offset + 3U])),
                      reinterpret_cast<void*>(remote_address), words[offset]);
          remote_address += words[offset];
        }
      }
      producer += blocks;
      if ((words[0] & (1U << 19U)) != 0U) {
        __atomic_store_n(&cq[completion * 8U], 4U, __ATOMIC_RELEASE);
        ++completion;
      }
    }
  });
  const auto result = KvafAicpuUrmaGatherRead(&args);
  stopped.store(true, std::memory_order_release);
  hardware.join();
  assert(result == 0U);
  assert(status.magic == kvaf::aicpu_urma::kGatherStatusMagic);
  assert(status.result == 0U && status.phase == 3U && status.ack_seen == 1U);
  assert(status.sequence == 1U && status.cqe_word == 4U);
  assert(status.sq_producer_index == 68U && status.cq_consumer_index == 3U);
  assert(status.ack_polls == 1U && status.read_wqes == 2U);
  assert(status.cpu.dispatch_ns == 11U && status.cpu.notify_spin_ns == 22U);
  assert(status.cpu.source_cache_ns == 33U && status.cpu.copy_ns == 44U);
  assert(status.cpu.staging_flush_ns == 55U);
  assert(status.notify_ns > 0U && status.ack_poll_ns > 0U);
  assert(status.notify_post_ns > 0U && status.notify_cq_ns > 0U);
  assert(status.ack_post_ns > 0U && status.ack_cq_ns > 0U);
  assert(status.scatter_post_ns > 0U && status.scatter_cq_ns > 0U);
  assert(sq_doorbell == 68U && cq_doorbell == 3U);
  std::array<std::uint8_t, 256> expected{};
  expected.fill(0xa5U);
  std::size_t staging_offset = 0U;
  for (const auto& span : output_spans) {
    std::memcpy(expected.data() + span.offset,
                remote.data() + 128U + staging_offset, span.length);
    staging_offset += span.length;
  }
  assert(staging_offset == 48U);
  for (std::uint32_t index = 0U; index < args.output_extent; ++index)
    assert(output[index] == expected[index]);
  args.sgl_size = 0U;
  const auto invalid_sgl_result = KvafAicpuUrmaGatherRead(&args);
  assert(invalid_sgl_result != 0U);
  assert(status.result == 2U && sq_doorbell == 68U);
  args.sgl_size = kvaf::kGatherMaxSglSize;
  args.bytes = 100U;
  const auto invalid_result = KvafAicpuUrmaGatherRead(&args);
  assert(invalid_result != 0U);
  assert(status.result == 2U && sq_doorbell == 68U);
  std::cout << "AICPU gather/read kernel simulation: PASS\n";
}
