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
extern "C" std::uint32_t KvafAicpuUrmaGatherScatter(kvaf::aicpu_urma::GatherReadArgs* args);

void TestRpcGather() {
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
  assert(status.ack_cache_ns == 0U && status.scatter_cache_ns == 0U);
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
}

void TestUrmaDescriptorGather() {
  alignas(64) std::array<std::uint8_t, 512> remote{};
  std::array<std::uint8_t, 512> output{};
  output.fill(0xa5U);
  std::array<std::uint32_t, 1024> sq{};
  std::array<std::uint32_t, 512> cq{};
  std::uint32_t sq_doorbell = 0U, cq_doorbell = 0U;
  const auto base = reinterpret_cast<std::uint64_t>(remote.data());
  const std::array<kvaf::peer::GatherSpan, 2> sources{{
      {base + 32U, 16U, 0U}, {base, 16U, 0U}}};
  const std::array<kvaf::GatherOutputSpan, 2> targets{{
      {32U, 16U, 0U}, {0U, 16U, 0U}}};
  for (std::uint32_t index = 0U; index < 64U; ++index)
    remote[index] = static_cast<std::uint8_t>(index + 1U);
  const kvaf::peer::GatherDescriptorHeader header{
      kvaf::peer::kGatherDescriptorMagic, 1U, base + 128U, 2U, 32U};
  std::memcpy(output.data() + 128U, &header, sizeof(header));
  std::memcpy(output.data() + 128U + sizeof(header), sources.data(), sizeof(sources));

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
  args.output_spans = reinterpret_cast<std::uint64_t>(targets.data());
  args.output_extent = 48U;
  args.output_span_count = targets.size();
  args.sgl_size = 2U;
  args.bytes = 32U;
  args.sequence = 1U;
  args.remote_staging = base + 128U;
  args.remote_descriptor = base + 256U;
  args.source_descriptor = entity.local_hbm_address + 128U;
  args.source_span_count = sources.size();
  args.remote_notify = base + 448U;
  args.remote_ack = base + 456U;
  args.remote_command = base + 464U;

  std::atomic<bool> stopped{false};
  std::thread hardware([&] {
    std::uint32_t producer = 0U;
    std::uint32_t completion = 0U;
    while (producer < 7U && !stopped.load(std::memory_order_acquire)) {
      if (__atomic_load_n(&sq_doorbell, __ATOMIC_ACQUIRE) == producer) {
        std::this_thread::yield();
        continue;
      }
      const auto* first = sq.data() + producer * 16U;
      const auto opcode = (first[1] >> 8U) & 0xffU;
      const auto blocks = opcode == kvaf::aicpu_urma::kWriteWithNotifyOpcode
          ? 2U : opcode == kvaf::aicpu_urma::kReadOpcode
              ? kvaf::aicpu_urma::ScatterWqebbs(first[2] >> 24U) : 1U;
      std::array<std::uint32_t, 48> words{};
      std::memcpy(words.data(), first, blocks * 64U);
      const auto address = [](std::uint32_t low, std::uint32_t high) {
        return static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32U);
      };
      if (opcode == kvaf::aicpu_urma::kWriteOpcode) {
        assert(words[12] == kvaf::peer::GatherDescriptorBytes(sources.size()));
        assert(address(words[10], words[11]) == args.remote_descriptor);
        std::memcpy(reinterpret_cast<void*>(args.remote_descriptor),
                    reinterpret_cast<const void*>(address(words[14], words[15])), words[12]);
      } else if (opcode == kvaf::aicpu_urma::kWriteWithNotifyOpcode) {
        assert(address(words[10], words[11]) == args.remote_command);
        assert(address(words[14], words[15]) == args.remote_notify);
        std::memcpy(reinterpret_cast<void*>(args.remote_command),
                    reinterpret_cast<const void*>(address(words[22], words[23])), 8U);
        const auto sequence = address(words[16], words[17]);
        std::memcpy(reinterpret_cast<void*>(args.remote_notify), &sequence, 8U);
        kvaf::peer::GatherDescriptorHeader received{};
        std::memcpy(&received, reinterpret_cast<const void*>(args.remote_descriptor),
                    sizeof(received));
        assert(received.magic == kvaf::peer::kGatherDescriptorMagic);
        assert(received.sequence == sequence && received.staging_address == args.remote_staging);
        std::vector<kvaf::peer::GatherSpan> gathered(received.span_count);
        std::memcpy(gathered.data(),
                    reinterpret_cast<const std::uint8_t*>(args.remote_descriptor) +
                        sizeof(received), gathered.size() * sizeof(gathered.front()));
        assert(kvaf::peer::ValidateDirectGather(
            base, remote.size(), args.remote_descriptor, received, gathered));
        kvaf::peer::GatherIntoStaging(remote.data(), base, received.staging_address, gathered);
        const auto started = kvaf::peer::GatherStartedAck(sequence);
        std::memcpy(reinterpret_cast<void*>(args.remote_ack), &started, 8U);
      } else {
        assert(opcode == kvaf::aicpu_urma::kReadOpcode);
        auto source = address(words[10], words[11]);
        for (std::uint32_t index = 0U; index < words[2] >> 24U; ++index) {
          const auto offset = 12U + index * 4U;
          std::memcpy(reinterpret_cast<void*>(address(words[offset + 2U], words[offset + 3U])),
                      reinterpret_cast<const void*>(source), words[offset]);
          source += words[offset];
        }
        if (address(words[10], words[11]) == args.remote_ack)
          std::memcpy(reinterpret_cast<void*>(args.remote_ack), &args.sequence, 8U);
      }
      producer += blocks;
      if ((words[0] & (1U << 19U)) != 0U)
        __atomic_store_n(&cq[completion++ * 8U], 4U, __ATOMIC_RELEASE);
    }
  });
  const auto result = KvafAicpuUrmaGatherRead(&args);
  stopped.store(true, std::memory_order_release);
  hardware.join();
  assert(result == 0U && status.result == 0U && status.phase == 4U);
  assert(status.source_write_ns > 0U && status.source_write_cq_ns > 0U);
  assert(status.ack_started_ns > 0U && status.ack_polls == 2U);
  assert(status.ack_cache_ns == 0U && status.scatter_cache_ns == 0U);
  assert(status.sq_producer_index == 7U && status.cq_consumer_index == 5U);
  for (std::size_t index = 0U; index < 16U; ++index) {
    assert(output[32U + index] == remote[32U + index]);
    assert(output[index] == remote[index]);
  }
  args.remote_descriptor = base + 500U;
  assert(KvafAicpuUrmaGatherRead(&args) != 0U);
  assert(status.result == 2U && sq_doorbell == 7U);
}

void TestScatterStagingBoundary() {
  alignas(64) std::array<std::uint8_t, 512> remote{};
  std::array<std::uint8_t, 256> output{};
  output.fill(0xa5U);
  std::array<std::uint32_t, 128U * 16U> sq{};
  std::array<std::uint32_t, 512> cq{};
  std::array<kvaf::GatherOutputSpan, 25> targets{};
  for (std::uint32_t index = 0U; index < targets.size(); ++index) {
    remote[256U + index] = static_cast<std::uint8_t>(index + 1U);
    targets[index] = {index * 2U, 1U, 0U};
  }
  std::uint32_t sq_doorbell = 124U;
  std::uint32_t cq_doorbell = 0U;
  const auto base = reinterpret_cast<std::uint64_t>(remote.data());
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
  entity.sq_depth = 128U;
  entity.wqe_size = 64U;
  entity.cq_va = reinterpret_cast<std::uint64_t>(cq.data());
  entity.cq_db_va = reinterpret_cast<std::uint64_t>(&cq_doorbell);
  entity.cq_depth = 64U;
  entity.cqe_size = 32U;
  entity.tp_id = 5U;
  KvafBatchGetQueueState queue{};
  queue.magic = 0x4b564151U;
  queue.initialized = 1U;
  queue.sq_producer_index = 124U;
  kvaf::aicpu_urma::GatherReadStatus status{};
  kvaf::aicpu_urma::GatherReadArgs args{};
  args.channel_entity = reinterpret_cast<std::uint64_t>(&entity);
  args.queue_state = reinterpret_cast<std::uint64_t>(&queue);
  args.core_status = reinterpret_cast<std::uint64_t>(&status);
  args.output = entity.local_hbm_address;
  args.output_spans = reinterpret_cast<std::uint64_t>(targets.data());
  args.output_extent = 49U;
  args.output_span_count = targets.size();
  args.sgl_size = 1U;
  args.bytes = targets.size();
  args.sequence = 1U;
  args.remote_staging = base + 256U;
  args.remote_notify = base + 448U;
  args.remote_ack = base + 456U;
  args.remote_command = base + 464U;

  std::atomic<bool> stopped{false};
  std::thread hardware([&] {
    std::uint32_t producer = 124U;
    std::uint32_t completion = 0U;
    while (producer < 152U && !stopped.load(std::memory_order_acquire)) {
      if (__atomic_load_n(&sq_doorbell, __ATOMIC_ACQUIRE) == producer) {
        std::this_thread::yield();
        continue;
      }
      std::uint32_t words[32]{};
      const auto* first = sq.data() + (producer % entity.sq_depth) * 16U;
      const auto opcode = (first[1] >> 8U) & 0xffU;
      const auto blocks = opcode == kvaf::aicpu_urma::kWriteWithNotifyOpcode ? 2U : 1U;
      for (std::uint32_t block = 0U; block < blocks; ++block)
        std::memcpy(words + block * 16U,
                    sq.data() + ((producer + block) % entity.sq_depth) * 16U, 64U);
      const auto address = [](std::uint32_t low, std::uint32_t high) {
        return static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32U);
      };
      if (opcode == kvaf::aicpu_urma::kWriteWithNotifyOpcode) {
        const auto sequence = address(words[16], words[17]);
        std::memcpy(reinterpret_cast<void*>(args.remote_notify), &sequence, 8U);
        std::memcpy(reinterpret_cast<void*>(args.remote_ack), &sequence, 8U);
      } else {
        assert(opcode == kvaf::aicpu_urma::kReadOpcode);
        std::memcpy(reinterpret_cast<void*>(address(words[14], words[15])),
                    reinterpret_cast<const void*>(address(words[10], words[11])), words[12]);
      }
      producer += blocks;
      if ((words[0] & (1U << 19U)) != 0U)
        __atomic_store_n(&cq[completion++ * 8U], 4U, __ATOMIC_RELEASE);
    }
  });
  const auto result = KvafAicpuUrmaGatherRead(&args);
  stopped.store(true, std::memory_order_release);
  hardware.join();
  assert(result == 0U && status.result == 0U);
  assert(status.read_wqes == targets.size() && status.sq_producer_index == 152U);
  assert(status.cq_consumer_index == 3U && cq_doorbell == 3U);
  for (std::uint32_t index = 0U; index < targets.size(); ++index) {
    assert(output[index * 2U] == index + 1U);
    assert(output[index * 2U + 1U] == 0xa5U);
  }
}

void TestParallelScatter(std::uint32_t workers, std::uint32_t span_count) {
  using kvaf::aicpu_urma::GatherReadStatus;
  constexpr std::uint32_t depth = 128U;
  alignas(64) std::array<std::uint8_t, 4096> remote{};
  std::array<std::uint8_t, 2048> output{};
  std::vector<kvaf::GatherOutputSpan> targets(span_count);
  std::vector<kvaf::peer::GatherSpan> sources(span_count);
  std::array<KvafBatchGetDirectUrmaEntity, 8> entities{};
  std::array<KvafBatchGetQueueState, 8> queues{};
  alignas(64) std::array<std::uint8_t, 8U * KVAF_AIV_BATCH_GET_STATUS_BYTES> statuses{};
  std::array<std::array<std::uint32_t, depth * 16U>, 8> sq{};
  std::array<std::array<std::uint32_t, 64U * 8U>, 8> cq{};
  std::array<std::uint32_t, 8> sq_doorbells{};
  std::array<std::uint32_t, 8> cq_doorbells{};
  const auto base = reinterpret_cast<std::uint64_t>(remote.data());
  std::uint32_t total_bytes = 0U;
  for (std::uint32_t index = 0U; index < span_count; ++index) {
    targets[index] = {(span_count - 1U - index) * 16U, index % 7U + 1U, 0U};
    sources[index] = {base + targets[index].offset, targets[index].length, 0U};
    total_bytes += targets[index].length;
  }
  for (std::uint32_t worker = 0U; worker < workers; ++worker) {
    auto& entity = entities[worker];
    entity.magic = 0x4b564155U;
    entity.version = 1U;
    entity.remote_host_ddr_address = base;
    entity.remote_host_ddr_size = remote.size();
    entity.remote_token_id = 3U;
    entity.remote_token_value = 4U;
    entity.local_hbm_address = reinterpret_cast<std::uint64_t>(output.data());
    entity.local_hbm_size = output.size();
    entity.local_token_id = worker + 7U;
    entity.sq_va = reinterpret_cast<std::uint64_t>(sq[worker].data());
    entity.sq_db_va = reinterpret_cast<std::uint64_t>(&sq_doorbells[worker]);
    entity.sq_depth = depth;
    entity.wqe_size = 64U;
    entity.cq_va = reinterpret_cast<std::uint64_t>(cq[worker].data());
    entity.cq_db_va = reinterpret_cast<std::uint64_t>(&cq_doorbells[worker]);
    entity.cq_depth = 64U;
    entity.cqe_size = 32U;
    entity.tp_id = worker + 5U;
    queues[worker].magic = 0x4b564151U;
    queues[worker].initialized = 1U;
    queues[worker].sq_producer_index = depth - 4U;
    sq_doorbells[worker] = depth - 4U;
  }
  kvaf::aicpu_urma::GatherReadArgs args{};
  args.channel_entity = reinterpret_cast<std::uint64_t>(entities.data());
  args.queue_state = reinterpret_cast<std::uint64_t>(queues.data());
  args.core_status = reinterpret_cast<std::uint64_t>(statuses.data());
  args.output = entities[0].local_hbm_address;
  args.output_spans = reinterpret_cast<std::uint64_t>(targets.data());
  args.output_extent = span_count * 16U;
  args.output_span_count = span_count;
  args.sgl_size = 6U;
  args.bytes = total_bytes;
  args.worker_count = workers;
  args.remote_staging = base + 1024U;
  args.remote_notify = base + 3008U;
  args.remote_ack = base + 3016U;
  args.remote_command = base + 3024U;
  args.remote_descriptor = base + 2048U;
  args.source_descriptor = args.output + ((args.output_extent + 63U) & ~std::uint64_t{63U}) +
      kvaf::aicpu_urma::kGatherScratchBytes;
  args.source_span_count = span_count;
  std::atomic<bool> stopped{false};
  std::atomic<std::uint32_t> notifications{0U};
  std::atomic<int> error_worker{-1};
  std::vector<std::thread> hardware;
  for (std::uint32_t worker = 0U; worker < workers; ++worker) {
    hardware.emplace_back([&, worker] {
      std::uint32_t producer = depth - 4U;
      std::uint32_t completion = 0U;
      const auto address = [](std::uint32_t low, std::uint32_t high) {
        return static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32U);
      };
      while (!stopped.load(std::memory_order_acquire)) {
        if (__atomic_load_n(&sq_doorbells[worker], __ATOMIC_ACQUIRE) == producer) {
          std::this_thread::yield();
          continue;
        }
        const auto* first = sq[worker].data() + (producer % depth) * 16U;
        const auto opcode = (first[1] >> 8U) & 0xffU;
        const auto blocks = opcode == kvaf::aicpu_urma::kWriteWithNotifyOpcode
            ? 2U : opcode == kvaf::aicpu_urma::kWriteOpcode
                ? 1U : kvaf::aicpu_urma::ScatterWqebbs(first[2] >> 24U);
        std::array<std::uint32_t, kvaf::aicpu_urma::kMaxScatterWqebbs * 16U> words{};
        for (std::uint32_t block = 0U; block < blocks; ++block)
          std::memcpy(words.data() + block * 16U,
                      sq[worker].data() + ((producer + block) % depth) * 16U, 64U);
        if (opcode == kvaf::aicpu_urma::kWriteOpcode) {
          assert(worker == 0U && address(words[10], words[11]) == args.remote_descriptor);
          std::memcpy(reinterpret_cast<void*>(args.remote_descriptor),
                      reinterpret_cast<const void*>(address(words[14], words[15])), words[12]);
        } else if (opcode == kvaf::aicpu_urma::kWriteWithNotifyOpcode) {
          assert(worker == 0U);
          const auto sequence = address(words[16], words[17]);
          kvaf::peer::GatherDescriptorHeader header{};
          std::memcpy(&header, reinterpret_cast<const void*>(args.remote_descriptor), sizeof(header));
          assert(header.sequence == sequence && header.span_count == span_count);
          std::vector<kvaf::peer::GatherSpan> received(span_count);
          std::memcpy(received.data(), reinterpret_cast<const std::uint8_t*>(args.remote_descriptor) +
                          sizeof(header), received.size() * sizeof(received.front()));
          assert(kvaf::peer::ValidateDirectGather(base, remote.size(), args.remote_descriptor,
                                                 header, received));
          kvaf::peer::GatherIntoStaging(remote.data(), base, args.remote_staging, received);
          std::memcpy(reinterpret_cast<void*>(args.remote_notify), &sequence, 8U);
          std::memcpy(reinterpret_cast<void*>(args.remote_ack), &sequence, 8U);
          ++notifications;
        } else {
          assert(opcode == kvaf::aicpu_urma::kReadOpcode);
          auto source = address(words[10], words[11]);
          for (std::uint32_t index = 0U; index < words[2] >> 24U; ++index) {
            const auto offset = 12U + index * 4U;
            assert(words[offset + 1U] == entities[worker].local_token_id);
            std::memcpy(reinterpret_cast<void*>(address(words[offset + 2U], words[offset + 3U])),
                        reinterpret_cast<const void*>(source), words[offset]);
            source += words[offset];
          }
        }
        producer += blocks;
        if ((words[0] & (1U << 19U)) != 0U) {
          const auto error = error_worker.load() == static_cast<int>(worker) ? 1U << 16U : 0U;
          const auto owner = ((completion / 64U) & 1U) == 0U ? 4U : 0U;
          __atomic_store_n(&cq[worker][(completion++ % 64U) * 8U], owner | error, __ATOMIC_RELEASE);
        }
      }
    });
  }
  const auto run_scatter = [&] {
    std::vector<std::thread> cores;
    std::array<std::uint32_t, 8> results{};
    for (std::uint32_t core = 0U; core < workers; ++core)
      cores.emplace_back([&, core] { results[core] = KvafAicpuUrmaGatherScatter(&args); });
    for (auto& core : cores) core.join();
    return results;
  };
  const auto prepare_sources = [&] {
    const kvaf::peer::GatherDescriptorHeader header{
        kvaf::peer::kGatherDescriptorMagic, args.sequence, args.remote_staging,
        span_count, total_bytes};
    std::memcpy(reinterpret_cast<void*>(args.source_descriptor), &header, sizeof(header));
    std::memcpy(reinterpret_cast<std::uint8_t*>(args.source_descriptor) + sizeof(header),
                sources.data(), sources.size() * sizeof(sources.front()));
  };
  for (std::uint32_t round = 1U; round <= 3U; ++round) {
    args.sequence = round;
    total_bytes = 0U;
    for (std::uint32_t index = 0U; index < span_count; ++index) {
      targets[index].length = (index + round) % 7U + 1U;
      sources[index].length = targets[index].length;
      total_bytes += targets[index].length;
    }
    args.bytes = total_bytes;
    output.fill(0xa5U);
    for (std::uint32_t index = 0U; index < args.output_extent; ++index)
      remote[index] = static_cast<std::uint8_t>(index + round);
    prepare_sources();
    assert(KvafAicpuUrmaGatherRead(&args) == 0U);
    for (std::uint32_t worker = 0U; worker < workers; ++worker) {
      std::uint64_t expected_offset = 0U;
      const auto first = span_count * worker / workers;
      for (std::uint32_t index = 0U; index < first; ++index)
        expected_offset += targets[index].length;
      assert(queues[worker].reserved[kvaf::aicpu_urma::kGatherStagingOffsetSlot] == expected_offset);
    }
    for (std::uint32_t index = 0U; index < args.output_extent; ++index)
      assert(output[index] == 0xa5U);
    const auto results = run_scatter();
    assert(queues[0].reserved[0] == workers);
    std::uint64_t wqes = 0U;
    for (std::uint32_t worker = 0U; worker < workers; ++worker) {
      assert(results[worker] == 0U);
      GatherReadStatus status{};
      std::memcpy(&status, statuses.data() + worker * KVAF_AIV_BATCH_GET_STATUS_BYTES, sizeof(status));
      assert(status.magic == kvaf::aicpu_urma::kGatherStatusMagic && status.sequence == round);
      assert(status.result == 0U && status.scatter_begin_ns > 0U);
      assert(status.scatter_end_ns >= status.scatter_begin_ns);
      assert(status.scatter_post_ns == status.scatter_encode_ns + status.scatter_sq_write_ns +
                                          status.scatter_doorbell_ns);
      assert(status.scatter_init_ns + status.scatter_plan_ns + status.scatter_post_ns +
                 status.scatter_cq_ns <= status.scatter_end_ns - status.scatter_begin_ns);
      if (status.read_wqes != 0U) assert(status.scatter_post_ns > 0U);
      assert(status.sq_producer_index == queues[worker].sq_producer_index);
      assert(status.cq_consumer_index == queues[worker].cq_consumer_index);
      assert(status.scatter_cache_ns == 0U);
      wqes += status.read_wqes;
    }
    std::uint32_t expected_wqes = 0U;
    for (std::uint32_t worker = 0U; worker < workers; ++worker) {
      const auto first = span_count * worker / workers;
      const auto last = span_count * (worker + 1U) / workers;
      expected_wqes += (last - first + 5U) / 6U;
    }
    assert(wqes == expected_wqes && notifications.load() == round);
    std::array<std::uint8_t, 2048> expected{};
    expected.fill(0xa5U);
    std::size_t prefix = 0U;
    for (const auto& span : targets) {
      std::memcpy(expected.data() + span.offset, remote.data() + 1024U + prefix, span.length);
      prefix += span.length;
    }
    assert(std::memcmp(output.data(), expected.data(), args.output_extent) == 0);
    assert(KvafAicpuUrmaGatherScatter(&args) != 0U);
  }
  args.sequence = 4U;
  prepare_sources();
  assert(KvafAicpuUrmaGatherRead(&args) == 0U);
  error_worker.store(static_cast<int>(workers - 1U));
  run_scatter();
  GatherReadStatus failed{};
  std::memcpy(&failed, statuses.data() + (workers - 1U) * KVAF_AIV_BATCH_GET_STATUS_BYTES, sizeof(failed));
  assert(failed.result == 8U && (failed.cqe_word & (1U << 16U)) != 0U);
  error_worker.store(-1);
  args.sequence = 5U;
  prepare_sources();
  assert(KvafAicpuUrmaGatherRead(&args) == 0U);
  entities[workers - 1U].local_token_id = 0U;
  run_scatter();
  std::memcpy(&failed, statuses.data() + (workers - 1U) * KVAF_AIV_BATCH_GET_STATUS_BYTES, sizeof(failed));
  assert(failed.result == 2U && failed.sequence == args.sequence);
  entities[workers - 1U].local_token_id = workers + 6U;
  args.sequence = 6U;
  ++args.bytes;
  assert(KvafAicpuUrmaGatherRead(&args) != 0U);
  const auto before = queues;
  run_scatter();
  for (std::uint32_t worker = 0U; worker < workers; ++worker)
    assert(queues[worker].sq_producer_index == before[worker].sq_producer_index);
  assert(notifications.load() == 5U);
  stopped.store(true, std::memory_order_release);
  for (auto& device : hardware) device.join();
}

int main() {
  TestRpcGather();
  TestUrmaDescriptorGather();
  TestScatterStagingBoundary();
  TestParallelScatter(2U, 41U);
  TestParallelScatter(4U, 41U);
  TestParallelScatter(8U, 41U);
  TestParallelScatter(8U, 9U);
  TestParallelScatter(8U, 3U);
  std::cout << "AICPU gather/read kernel simulation: PASS\n";
}
