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

void TestRpcGather(std::uint32_t task_spans = 0U) {
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
  alignas(64) std::array<std::uint8_t, 512> descriptor_storage{};
  std::memcpy(descriptor_storage.data(), output_spans.data(), sizeof(output_spans));
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
  args.output_spans = reinterpret_cast<std::uint64_t>(descriptor_storage.data());
  args.read_task_spans = task_spans;
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
  const auto completions = task_spans != 0U ? 4U : 3U;
  assert(status.sq_producer_index == 68U && status.cq_consumer_index == completions);
  assert(status.read_tasks == (task_spans != 0U ? 2U : 1U));
  assert(status.ack_polls == 1U && status.read_wqes == 2U);
  assert(status.cpu.dispatch_ns == 11U && status.cpu.notify_spin_ns == 22U);
  assert(status.cpu.source_cache_ns == 33U && status.cpu.copy_ns == 44U);
  assert(status.cpu.staging_flush_ns == 55U);
  assert(status.notify_ns > 0U && status.ack_poll_ns > 0U);
  assert(status.ack_cache_ns == 0U && status.scatter_cache_ns == 0U);
  assert(status.notify_post_ns > 0U && status.notify_cq_ns > 0U);
  assert(status.ack_post_ns > 0U && status.ack_cq_ns > 0U);
  assert(status.scatter_post_ns > 0U && status.scatter_cq_ns > 0U);
  assert(sq_doorbell == 68U && cq_doorbell == completions);
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

void TestUrmaDescriptorGather(std::uint32_t pipeline_chunk_bytes = 0U) {
  alignas(64) std::array<std::uint8_t, 512> remote{};
  std::array<std::uint8_t, 1024> output{};
  output.fill(0xa5U);
  std::array<std::uint32_t, 1024> sq{};
  std::array<std::uint32_t, 512> cq{};
  std::uint32_t sq_doorbell = 0U, cq_doorbell = 0U;
  const auto base = reinterpret_cast<std::uint64_t>(remote.data());
  const std::array<kvaf::peer::GatherSpan, 2> sources{{
      {base + 32U, 16U, 0U}, {base, 16U, 0U}}};
  const std::array<kvaf::GatherOutputSpan, 2> targets{{
      {32U, 16U, 0U}, {0U, 16U, 0U}}};
  alignas(64) std::array<std::uint8_t, 256> target_storage{};
  std::memcpy(target_storage.data(), targets.data(), sizeof(targets));
  const auto source_offset = 64U + kvaf::aicpu_urma::GatherScratchBytes(pipeline_chunk_bytes);
  for (std::uint32_t index = 0U; index < 64U; ++index)
    remote[index] = static_cast<std::uint8_t>(index + 1U);
  const kvaf::peer::GatherDescriptorHeader header{
      kvaf::peer::kGatherDescriptorMagic, 1U, base + 128U, 2U, 32U, pipeline_chunk_bytes};
  std::memcpy(output.data() + source_offset, &header, sizeof(header));
  std::memcpy(output.data() + source_offset + sizeof(header), sources.data(), sizeof(sources));

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
  args.output_spans = reinterpret_cast<std::uint64_t>(target_storage.data());
  args.output_extent = 48U;
  args.output_span_count = targets.size();
  args.sgl_size = 2U;
  args.bytes = 32U;
  args.sequence = 1U;
  args.remote_staging = base + 128U;
  args.remote_descriptor = base + 256U;
  args.source_descriptor = entity.local_hbm_address + source_offset;
  args.source_span_count = sources.size();
  args.remote_notify = base + 448U;
  args.remote_ack = base + 456U;
  args.remote_command = base + 464U;
  args.pipeline_chunk_bytes = pipeline_chunk_bytes;
  args.read_task_spans = pipeline_chunk_bytes != 0U ? 2U : 0U;
  args.worker_count = 1U;

  std::atomic<bool> stopped{false};
  std::thread hardware([&] {
    std::uint32_t producer = 0U;
    std::uint32_t completion = 0U;
    while (!stopped.load(std::memory_order_acquire)) {
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
        if (pipeline_chunk_bytes != 0U)
          __atomic_store_n(reinterpret_cast<std::uint64_t*>(args.remote_descriptor +
              offsetof(kvaf::peer::GatherDescriptorHeader, ready_bytes)),
              static_cast<std::uint64_t>(args.bytes), __ATOMIC_RELEASE);
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
  const auto expected_producer = pipeline_chunk_bytes == 0U ? 7U : 8U;
  assert(status.sq_producer_index == expected_producer &&
         status.cq_consumer_index == (pipeline_chunk_bytes == 0U ? 5U : 6U));
  for (std::size_t index = 0U; index < 16U; ++index) {
    assert(output[32U + index] == remote[32U + index]);
    assert(output[index] == remote[index]);
  }
  args.remote_descriptor = base + 500U;
  assert(KvafAicpuUrmaGatherRead(&args) != 0U);
  assert(status.result == 2U && sq_doorbell == expected_producer);
}

void TestScatterStagingBoundary(std::uint32_t sq_batch_wqes,
                                std::uint32_t span_count = 25U,
                                std::uint32_t sgl_size = 1U, bool detailed = false,
                                std::uint32_t sq_depth = 256U,
                                kvaf::GatherSqCopyMode copy_mode = kvaf::GatherSqCopyMode::kMemcpy) {
  alignas(64) std::array<std::uint8_t, 1024> remote{};
  std::array<std::uint8_t, 1024> output{};
  output.fill(0xa5U);
  std::array<std::uint32_t, 256U * 16U> sq{};
  std::array<std::uint32_t, 512> cq{};
  std::vector<kvaf::GatherOutputSpan> targets(span_count);
  for (std::uint32_t index = 0U; index < targets.size(); ++index) {
    remote[128U + index] = static_cast<std::uint8_t>(index + 1U);
    targets[index] = {index * 2U, 1U, 0U};
  }
  std::uint32_t sq_doorbell = sq_depth - 4U;
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
  entity.sq_depth = sq_depth;
  entity.wqe_size = 64U;
  entity.cq_va = reinterpret_cast<std::uint64_t>(cq.data());
  entity.cq_db_va = reinterpret_cast<std::uint64_t>(&cq_doorbell);
  entity.cq_depth = 64U;
  entity.cqe_size = 32U;
  entity.tp_id = 5U;
  KvafBatchGetQueueState queue{};
  queue.magic = 0x4b564151U;
  queue.initialized = 1U;
  queue.sq_producer_index = sq_depth - 4U;
  kvaf::aicpu_urma::GatherReadStatus status{};
  kvaf::aicpu_urma::GatherReadArgs args{};
  args.channel_entity = reinterpret_cast<std::uint64_t>(&entity);
  args.queue_state = reinterpret_cast<std::uint64_t>(&queue);
  args.core_status = reinterpret_cast<std::uint64_t>(&status);
  args.output = entity.local_hbm_address;
  args.output_spans = reinterpret_cast<std::uint64_t>(targets.data());
  args.output_extent = 2U * span_count - 1U;
  args.output_span_count = targets.size();
  args.sgl_size = sgl_size;
  args.profile_detail = detailed ? 1U : 0U;
  args.sq_batch_wqes = sq_batch_wqes;
  args.sq_copy_mode = copy_mode;
  args.bytes = targets.size();
  args.sequence = 1U;
  args.remote_staging = base + 128U;
  args.remote_notify = base + 768U;
  args.remote_ack = base + 776U;
  args.remote_command = base + 784U;
  const auto expected_wqes = (span_count + sgl_size - 1U) / sgl_size;
  std::uint32_t expected_blocks = 0U;
  for (std::uint32_t index = 0U; index < span_count; index += sgl_size)
    expected_blocks += kvaf::aicpu_urma::ScatterWqebbs(std::min(sgl_size, span_count - index));
  const auto expected_producer = sq_depth - 1U + expected_blocks;

  std::atomic<bool> stopped{false};
  std::thread hardware([&] {
    std::uint32_t producer = sq_depth - 4U;
    std::uint32_t completion = 0U;
    while (producer < expected_producer && !stopped.load(std::memory_order_acquire)) {
      if (__atomic_load_n(&sq_doorbell, __ATOMIC_ACQUIRE) == producer) {
        std::this_thread::yield();
        continue;
      }
      std::uint32_t words[48]{};
      const auto* first = sq.data() + (producer % entity.sq_depth) * 16U;
      const auto opcode = (first[1] >> 8U) & 0xffU;
      const auto blocks = opcode == kvaf::aicpu_urma::kWriteWithNotifyOpcode
          ? 2U : kvaf::aicpu_urma::ScatterWqebbs(first[2] >> 24U);
      for (std::uint32_t block = 0U; block < blocks; ++block)
        std::memcpy(words + block * 16U,
                    sq.data() + ((producer + block) % entity.sq_depth) * 16U, 64U);
      const auto address = [](std::uint32_t low, std::uint32_t high) {
        return static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32U);
      };
      assert((words[0] >> 31U) == ((producer / entity.sq_depth) & 1U));
      if (opcode == kvaf::aicpu_urma::kWriteWithNotifyOpcode) {
        const auto sequence = address(words[16], words[17]);
        std::memcpy(reinterpret_cast<void*>(args.remote_notify), &sequence, 8U);
        std::memcpy(reinterpret_cast<void*>(args.remote_ack), &sequence, 8U);
      } else {
        assert(opcode == kvaf::aicpu_urma::kReadOpcode);
        auto source = address(words[10], words[11]);
        for (std::uint32_t index = 0U; index < words[2] >> 24U; ++index) {
          const auto offset = 12U + index * 4U;
          std::memcpy(reinterpret_cast<void*>(address(words[offset + 2U], words[offset + 3U])),
                      reinterpret_cast<const void*>(source), words[offset]);
          source += words[offset];
        }
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
  assert(status.read_wqes == expected_wqes && status.sq_producer_index == expected_producer);
  assert(status.cq_consumer_index == 3U && cq_doorbell == 3U);
  assert(status.sq_wqebbs == expected_blocks);
  assert(status.sq_write_calls == (expected_wqes + sq_batch_wqes - 1U) / sq_batch_wqes + 1U);
  if (detailed)
    assert(status.scatter_post_ns == status.scatter_encode_ns + status.scatter_sq_write_ns +
                                        status.scatter_doorbell_ns);
  for (std::uint32_t index = 0U; index < targets.size(); ++index) {
    assert(output[index * 2U] == static_cast<std::uint8_t>(index + 1U));
    assert(output[index * 2U + 1U] == 0xa5U);
  }
}

void TestParallelScatter(std::uint32_t workers, std::uint32_t span_count,
                         bool detailed = true, std::uint32_t sq_batch_wqes = 8U,
                         std::uint32_t task_spans = 0U, bool delay_followers = false,
                         bool packed_metadata = false,
                         kvaf::GatherSqCopyMode copy_mode = kvaf::GatherSqCopyMode::kMemcpy,
                         std::uint32_t pipeline_chunk_bytes = 0U) {
  using kvaf::aicpu_urma::GatherReadStatus;
  constexpr std::uint32_t depth = 128U;
  alignas(64) std::array<std::uint8_t, 4096> remote{};
  alignas(64) std::array<std::uint8_t, 4096> output{};
  std::vector<kvaf::GatherOutputSpan> targets(span_count);
  std::vector<std::uint64_t> target_storage(
      (kvaf::aicpu_urma::GatherOutputStorageBytes(span_count, task_spans) + 63U) / 8U + 1U);
  const auto target_address = (reinterpret_cast<std::uint64_t>(target_storage.data()) + 63U) &
                              ~std::uint64_t{63U};
  std::array<std::atomic<std::uint32_t>, 2048> output_writes{};
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
  args.output_spans = target_address;
  args.output_extent = span_count * 16U;
  args.output_span_count = span_count;
  args.sgl_size = 6U;
  args.bytes = total_bytes;
  args.worker_count = workers;
  args.profile_detail = detailed ? 1U : 0U;
  args.sq_batch_wqes = sq_batch_wqes;
  args.sq_copy_mode = copy_mode;
  args.read_task_spans = task_spans;
  args.pipeline_chunk_bytes = pipeline_chunk_bytes;
  args.remote_staging = base + 1024U;
  args.remote_notify = base + 3008U;
  args.remote_ack = base + 3016U;
  args.remote_command = base + 3024U;
  args.remote_descriptor = base + 2048U;
  args.source_descriptor = args.output + ((args.output_extent + 63U) & ~std::uint64_t{63U}) +
      kvaf::aicpu_urma::GatherScratchBytes(pipeline_chunk_bytes);
  args.source_span_count = span_count;
  if (packed_metadata) {
    assert(kvaf::aicpu_urma::GatherReadPackedOutputBytes(args.output_extent,
        kvaf::peer::GatherDescriptorBytes(span_count), span_count, task_spans,
        pipeline_chunk_bytes) <= output.size());
    args.output_spans = args.source_descriptor + kvaf::aicpu_urma::GatherPackedTargetOffset(
        kvaf::peer::GatherDescriptorBytes(span_count));
  }
  std::atomic<bool> stopped{false};
  std::atomic<std::uint32_t> notifications{0U};
  std::atomic<int> error_worker{-1};
  std::atomic<bool> first_read{false};
  std::atomic<bool> read_before_gather_done{false};
  std::atomic<bool> progress_error{false};
  std::thread cpu;
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
        assert((words[0] >> 31U) == ((producer / depth) & 1U));
        assert(words[9] == 0U);
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
          std::memcpy(reinterpret_cast<void*>(args.remote_notify), &sequence, 8U);
          if (pipeline_chunk_bytes == 0U) {
            kvaf::peer::GatherIntoStaging(remote.data(), base, args.remote_staging, received);
            std::memcpy(reinterpret_cast<void*>(args.remote_ack), &sequence, 8U);
          } else {
            assert(!cpu.joinable() && header.pipeline_chunk_bytes == pipeline_chunk_bytes);
            cpu = std::thread([&, received, sequence] {
              std::vector<std::uint8_t> gathered(args.bytes);
              std::size_t offset = 0U;
              for (const auto& span : received) {
                std::memcpy(gathered.data() + offset, reinterpret_cast<const void*>(span.address), span.length);
                offset += span.length;
              }
              auto* progress = reinterpret_cast<std::uint64_t*>(args.remote_descriptor +
                  offsetof(kvaf::peer::GatherDescriptorHeader, ready_bytes));
              if (progress_error.load(std::memory_order_acquire)) {
                __atomic_store_n(progress, kvaf::peer::kGatherErrorBit, __ATOMIC_RELEASE);
                __atomic_store_n(reinterpret_cast<std::uint64_t*>(args.remote_ack),
                                 sequence | kvaf::peer::kGatherErrorBit, __ATOMIC_RELEASE);
                return;
              }
              for (std::uint32_t begin = 0U; begin < args.bytes; begin += pipeline_chunk_bytes) {
                const auto end = std::min(begin + pipeline_chunk_bytes, args.bytes);
                std::memcpy(reinterpret_cast<void*>(args.remote_staging + begin),
                            gathered.data() + begin, end - begin);
                __atomic_store_n(progress, static_cast<std::uint64_t>(end), __ATOMIC_RELEASE);
                if (begin == 0U && end < args.bytes)
                  while (!first_read.load(std::memory_order_acquire)) std::this_thread::yield();
              }
              __atomic_store_n(reinterpret_cast<std::uint64_t*>(args.remote_ack), sequence, __ATOMIC_RELEASE);
            });
          }
          ++notifications;
        } else {
          assert(opcode == kvaf::aicpu_urma::kReadOpcode);
          auto source = address(words[10], words[11]);
          if (pipeline_chunk_bytes != 0U && source >= args.remote_staging &&
              source - args.remote_staging < args.bytes) {
            const auto ready = __atomic_load_n(reinterpret_cast<const std::uint64_t*>(
                args.remote_descriptor + offsetof(kvaf::peer::GatherDescriptorHeader, ready_bytes)),
                __ATOMIC_ACQUIRE);
            std::uint64_t bytes = 0U;
            for (std::uint32_t index = 0U; index < words[2] >> 24U; ++index)
              bytes += words[12U + index * 4U];
            assert(source - args.remote_staging + bytes <= ready);
            if (ready < args.bytes) read_before_gather_done.store(true, std::memory_order_release);
            first_read.store(true, std::memory_order_release);
          }
          for (std::uint32_t index = 0U; index < words[2] >> 24U; ++index) {
            const auto offset = 12U + index * 4U;
            assert(words[offset + 1U] == entities[worker].local_token_id);
            std::memcpy(reinterpret_cast<void*>(address(words[offset + 2U], words[offset + 3U])),
                        reinterpret_cast<const void*>(source), words[offset]);
            const auto destination = address(words[offset + 2U], words[offset + 3U]);
            if (destination >= args.output && destination - args.output < args.output_extent) {
              assert(words[offset] <= args.output_extent - (destination - args.output));
              for (std::uint32_t byte = 0U; byte < words[offset]; ++byte)
                ++output_writes[destination - args.output + byte];
            }
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
  const auto run_scatter = [&](bool first_synchronously = false) {
    std::vector<std::thread> cores;
    std::array<std::uint32_t, 8> results{};
    if (first_synchronously) results[0U] = KvafAicpuUrmaGatherScatter(&args);
    for (std::uint32_t core = first_synchronously ? 1U : 0U; core < workers; ++core)
      cores.emplace_back([&, core] { results[core] = KvafAicpuUrmaGatherScatter(&args); });
    for (auto& core : cores) core.join();
    return results;
  };
  const auto prepare_sources = [&] {
    const kvaf::peer::GatherDescriptorHeader header{
        kvaf::peer::kGatherDescriptorMagic, args.sequence, args.remote_staging,
        span_count, total_bytes, pipeline_chunk_bytes};
    const auto target_offset = kvaf::aicpu_urma::GatherPackedTargetOffset(
        kvaf::peer::GatherDescriptorBytes(span_count));
    std::vector<std::uint8_t> metadata(target_offset + targets.size() * sizeof(targets.front()));
    auto* source_destination = packed_metadata ? metadata.data()
        : reinterpret_cast<std::uint8_t*>(args.source_descriptor);
    auto* target_destination = packed_metadata ? metadata.data() + target_offset
        : reinterpret_cast<std::uint8_t*>(args.output_spans);
    std::memcpy(target_destination, targets.data(), targets.size() * sizeof(targets.front()));
    std::memcpy(source_destination, &header, sizeof(header));
    std::memcpy(source_destination + sizeof(header),
                sources.data(), sources.size() * sizeof(sources.front()));
    if (packed_metadata)
      std::memcpy(reinterpret_cast<void*>(args.source_descriptor), metadata.data(), metadata.size());
  };
  for (std::uint32_t round = 1U; round <= 3U; ++round) {
    first_read.store(false, std::memory_order_relaxed);
    read_before_gather_done.store(false, std::memory_order_relaxed);
    args.sequence = round;
    total_bytes = 0U;
    for (std::uint32_t index = 0U; index < span_count; ++index) {
      if (packed_metadata) {
        targets[index].offset = ((index + round) % span_count) * 16U;
        sources[index].address = base + targets[index].offset;
      }
      targets[index].length = (index + round) % 7U + 1U;
      sources[index].length = targets[index].length;
      total_bytes += targets[index].length;
    }
    args.bytes = total_bytes;
    output.fill(0xa5U);
    for (auto& writes : output_writes) writes.store(0U);
    for (std::uint32_t index = 0U; index < args.output_extent; ++index)
      remote[index] = static_cast<std::uint8_t>(index + round);
    prepare_sources();
    assert(KvafAicpuUrmaGatherRead(&args) == 0U);
    for (std::uint32_t worker = 0U; worker < workers; ++worker) {
      std::uint64_t expected_offset = 0U;
      const auto first = span_count * worker / workers;
      for (std::uint32_t index = 0U; index < first; ++index)
        expected_offset += targets[index].length;
      if (task_spans == 0U)
        assert(queues[worker].reserved[kvaf::aicpu_urma::kGatherStagingOffsetSlot] == expected_offset);
    }
    const auto task_count = kvaf::aicpu_urma::GatherTaskCount(span_count, task_spans);
    if (task_spans != 0U) {
      const auto* tasks = reinterpret_cast<const kvaf::aicpu_urma::GatherTaskState*>(
          args.output_spans + kvaf::aicpu_urma::GatherTaskStateOffset(span_count));
      assert(tasks->next_task == 0U);
      const auto* offsets = reinterpret_cast<const std::uint64_t*>(tasks + 1U);
      std::uint64_t expected_offset = 0U;
      for (std::uint32_t index = 0U; index < span_count; ++index) {
        if (index % task_spans == 0U) assert(offsets[index / task_spans] == expected_offset);
        expected_offset += targets[index].length;
      }
    }
    for (std::uint32_t index = 0U; index < args.output_extent; ++index)
      assert(output[index] == 0xa5U);
    const auto results = run_scatter(delay_followers);
    if (cpu.joinable()) cpu.join();
    if (pipeline_chunk_bytes != 0U && args.bytes > pipeline_chunk_bytes)
      assert(read_before_gather_done.load(std::memory_order_acquire));
    assert(queues[0].reserved[0] == workers);
    std::uint64_t wqes = 0U;
    std::uint64_t completed_tasks = 0U;
    for (std::uint32_t worker = 0U; worker < workers; ++worker) {
      assert(results[worker] == 0U);
      GatherReadStatus status{};
      std::memcpy(&status, statuses.data() + worker * KVAF_AIV_BATCH_GET_STATUS_BYTES, sizeof(status));
      assert(status.magic == kvaf::aicpu_urma::kGatherStatusMagic && status.sequence == round);
      assert(status.result == 0U && status.scatter_begin_ns > 0U);
      assert(status.scatter_end_ns >= status.scatter_begin_ns);
      assert(status.profile_detail == args.profile_detail);
      if (detailed)
        assert(status.scatter_post_ns == status.scatter_encode_ns + status.scatter_sq_write_ns +
                                            status.scatter_doorbell_ns);
      else
        assert(status.scatter_encode_ns == 0U && status.scatter_sq_write_ns == 0U &&
               status.scatter_doorbell_ns == 0U);
      assert(status.scatter_init_ns + status.scatter_schedule_ns + status.ready_wait_ns +
                 status.scatter_plan_ns + status.scatter_post_ns +
                 status.scatter_cq_ns <= status.scatter_end_ns - status.scatter_begin_ns);
      if (status.read_wqes != 0U) {
        assert(status.scatter_post_ns > 0U);
        assert(status.sq_write_calls > 0U && status.sq_wqebbs >= status.read_wqes);
      } else {
        assert(status.sq_write_calls == 0U && status.sq_wqebbs == 0U);
      }
      assert(status.sq_producer_index == queues[worker].sq_producer_index);
      assert(status.cq_consumer_index == queues[worker].cq_consumer_index);
      assert(status.scatter_cache_ns == 0U);
      wqes += status.read_wqes;
      completed_tasks += status.read_tasks;
      if (task_spans != 0U && delay_followers)
        assert(status.read_tasks == (worker == 0U ? task_count : 0U));
    }
    std::uint32_t expected_wqes = 0U;
    for (std::uint32_t part = 0U; part < (task_spans != 0U ? task_count : workers); ++part) {
      const auto first = task_spans != 0U ? part * task_spans : span_count * part / workers;
      const auto last = task_spans != 0U ? std::min(first + task_spans, span_count)
                                        : span_count * (part + 1U) / workers;
      expected_wqes += (last - first + 5U) / 6U;
    }
    assert(completed_tasks == (task_spans != 0U ? task_count : std::min(workers, span_count)));
    assert(wqes == expected_wqes && notifications.load() == round);
    std::array<std::uint8_t, 2048> expected{};
    std::array<std::uint32_t, 2048> expected_writes{};
    expected.fill(0xa5U);
    std::size_t prefix = 0U;
    for (const auto& span : targets) {
      std::memcpy(expected.data() + span.offset, remote.data() + 1024U + prefix, span.length);
      prefix += span.length;
      for (std::uint32_t byte = 0U; byte < span.length; ++byte)
        ++expected_writes[span.offset + byte];
    }
    assert(std::memcmp(output.data(), expected.data(), args.output_extent) == 0);
    for (std::uint32_t byte = 0U; byte < args.output_extent; ++byte)
      assert(output_writes[byte].load() == expected_writes[byte]);
    assert(KvafAicpuUrmaGatherScatter(&args) != 0U);
  }
  args.sequence = 4U;
  prepare_sources();
  assert(KvafAicpuUrmaGatherRead(&args) == 0U);
  const auto failing_worker = task_spans != 0U ? 0U : workers - 1U;
  error_worker.store(static_cast<int>(failing_worker));
  first_read.store(true, std::memory_order_relaxed);
  run_scatter(task_spans != 0U);
  if (cpu.joinable()) cpu.join();
  GatherReadStatus failed{};
  std::memcpy(&failed, statuses.data() + failing_worker * KVAF_AIV_BATCH_GET_STATUS_BYTES, sizeof(failed));
  assert(failed.result == 8U && (failed.cqe_word & (1U << 16U)) != 0U);
  error_worker.store(-1);
  args.sequence = 5U;
  prepare_sources();
  assert(KvafAicpuUrmaGatherRead(&args) == 0U);
  entities[workers - 1U].local_token_id = 0U;
  run_scatter();
  if (cpu.joinable()) cpu.join();
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
  --args.bytes;
  args.profile_detail = 2U;
  assert(KvafAicpuUrmaGatherRead(&args) != 0U);
  args.profile_detail = 0U;
  args.sq_batch_wqes = 7U;
  assert(KvafAicpuUrmaGatherRead(&args) != 0U);
  assert(notifications.load() == 5U);
  args.sq_batch_wqes = 8U;
  args.read_task_spans = 65537U;
  assert(KvafAicpuUrmaGatherRead(&args) != 0U);
  assert(KvafAicpuUrmaGatherScatter(&args) != 0U);
  args.read_task_spans = task_spans;
  args.sq_copy_mode = static_cast<kvaf::GatherSqCopyMode>(2U);
  assert(KvafAicpuUrmaGatherRead(&args) != 0U);
  assert(KvafAicpuUrmaGatherScatter(&args) != 0U);
  if (pipeline_chunk_bytes != 0U) {
    args.sq_copy_mode = copy_mode;
    prepare_sources();
    progress_error.store(true, std::memory_order_release);
    assert(KvafAicpuUrmaGatherRead(&args) == 0U);
    const auto results = run_scatter();
    if (cpu.joinable()) cpu.join();
    bool failed = false;
    for (std::uint32_t worker = 0U; worker < workers; ++worker) {
      GatherReadStatus status{};
      std::memcpy(&status, statuses.data() + worker * KVAF_AIV_BATCH_GET_STATUS_BYTES, sizeof(status));
      assert(status.read_wqes == 0U);
      if (status.result != 0U) {
        assert(status.result == 10U);
        failed = true;
      }
    }
    assert(failed && std::any_of(results.begin(), results.begin() + workers,
                                 [](std::uint32_t result) { return result != 0U; }));
  }
  stopped.store(true, std::memory_order_release);
  for (auto& device : hardware) device.join();
}

int main() {
  TestRpcGather();
  TestRpcGather(6U);
  TestUrmaDescriptorGather();
  TestUrmaDescriptorGather(64U);
  TestScatterStagingBoundary(8U);
  TestScatterStagingBoundary(16U);
  TestScatterStagingBoundary(32U);
  TestScatterStagingBoundary(64U);
  TestScatterStagingBoundary(8U, 384U, 6U);
  TestScatterStagingBoundary(16U, 384U, 6U);
  TestScatterStagingBoundary(32U, 384U, 6U);
  TestScatterStagingBoundary(64U, 384U, 6U, true);
  TestScatterStagingBoundary(8U, 384U, 6U, true, 193U);
  for (std::uint32_t workers = 2U; workers <= 8U; ++workers)
    TestParallelScatter(workers, 41U);
  TestParallelScatter(5U, 3U);
  TestParallelScatter(6U, 3U);
  TestParallelScatter(8U, 9U);
  TestParallelScatter(8U, 3U);
  TestParallelScatter(8U, 41U, false);
  TestParallelScatter(8U, 41U, true, 16U);
  TestParallelScatter(8U, 41U, false, 32U);
  TestParallelScatter(8U, 41U, true, 64U);
  TestParallelScatter(5U, 41U, true, 8U, 6U, true);
  TestParallelScatter(5U, 41U, true, 8U, 10U);
  TestParallelScatter(6U, 41U, false, 16U, 12U);
  TestParallelScatter(8U, 41U, true, 32U, 1U);
  TestParallelScatter(5U, 3U, true, 64U, 192U);
  TestParallelScatter(5U, 41U, true, 8U, 0U, false, true);
  TestParallelScatter(5U, 41U, true, 8U, 10U, false, true);
  TestParallelScatter(8U, 41U, true, 8U, 1U, false, true);
  TestScatterStagingBoundary(8U, 384U, 6U, true, 193U, kvaf::GatherSqCopyMode::kPaired);
  TestScatterStagingBoundary(64U, 384U, 6U, true, 256U, kvaf::GatherSqCopyMode::kPaired);
  TestParallelScatter(5U, 41U, true, 8U, 0U, false, true, kvaf::GatherSqCopyMode::kPaired);
  TestParallelScatter(5U, 41U, true, 8U, 10U, false, true, kvaf::GatherSqCopyMode::kPaired);
  TestParallelScatter(5U, 41U, true, 8U, 6U, false, true,
                      kvaf::GatherSqCopyMode::kMemcpy, 64U);
  TestParallelScatter(8U, 41U, false, 16U, 1U, false, false,
                      kvaf::GatherSqCopyMode::kPaired, 64U);
  TestParallelScatter(5U, 3U, true, 8U, 1U, false, true,
                      kvaf::GatherSqCopyMode::kMemcpy, 128U);
  std::cout << "AICPU gather/read kernel simulation: PASS\n";
}
