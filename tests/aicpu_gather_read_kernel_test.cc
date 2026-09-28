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
  std::array<std::uint8_t, 128> output{};
  std::array<std::uint32_t, 1024> sq{};
  std::array<std::uint32_t, 512> cq{};
  std::uint32_t sq_doorbell = 0U, cq_doorbell = 0U;
  const auto base = reinterpret_cast<std::uint64_t>(remote.data());
  const std::vector<kvaf::peer::GatherSpan> spans{
      {base + 80U, 16U, 0U}, {base, 16U, 0U}, {base + 48U, 16U, 0U}};
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
  args.sequence = 1U;
  args.bytes = 48U;

  std::atomic<bool> stopped{false};
  std::thread hardware([&] {
    std::uint32_t producer = 0U;
    std::uint32_t completion = 0U;
    while (producer < 4U && !stopped.load(std::memory_order_acquire)) {
      if (__atomic_load_n(&sq_doorbell, __ATOMIC_ACQUIRE) == producer) {
        std::this_thread::yield();
        continue;
      }
      const std::uint32_t* words = sq.data() + (producer % entity.sq_depth) * 16U;
      const auto opcode = (words[1] >> 8U) & 0xffU;
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
        std::memcpy(reinterpret_cast<void*>(args.remote_ack), &sequence, 8U);
        producer += 2U;
      } else {
        assert(opcode == kvaf::aicpu_urma::kReadOpcode);
        std::memcpy(reinterpret_cast<void*>(address(words[14], words[15])),
                    reinterpret_cast<void*>(address(words[10], words[11])), words[12]);
        ++producer;
      }
      __atomic_store_n(&cq[completion * 8U], 4U, __ATOMIC_RELEASE);
      ++completion;
    }
  });
  const auto result = KvafAicpuUrmaGatherRead(&args);
  stopped.store(true, std::memory_order_release);
  hardware.join();
  assert(result == 0U);
  assert(status.magic == kvaf::aicpu_urma::kGatherStatusMagic);
  assert(status.result == 0U && status.phase == 3U && status.ack_seen == 1U);
  assert(status.sequence == 1U && status.cqe_word == 4U);
  assert(status.sq_producer_index == 4U && status.cq_consumer_index == 3U);
  assert(sq_doorbell == 4U && cq_doorbell == 3U);
  for (std::uint32_t index = 0U; index < 16U; ++index) {
    assert(output[index] == 80U + index);
    assert(output[16U + index] == index);
    assert(output[32U + index] == 48U + index);
  }
  args.bytes = 100U;
  const auto invalid_result = KvafAicpuUrmaGatherRead(&args);
  assert(invalid_result != 0U);
  assert(status.result == 2U && sq_doorbell == 4U);
  std::cout << "AICPU gather/read kernel simulation: PASS\n";
}
