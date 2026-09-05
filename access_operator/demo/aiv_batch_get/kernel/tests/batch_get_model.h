#ifndef KVAF_BATCH_GET_HOST_MODEL_H_
#define KVAF_BATCH_GET_HOST_MODEL_H_
#include "kvaf_batch_get_device.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

namespace kvaf_host_model {
constexpr uint32_t kOwners = 2U;
constexpr uint32_t kJettys = 2U;
constexpr uint32_t kCapacity = 4608U;
constexpr uint32_t kTokenBytes = 13U;
constexpr uint32_t kTokensPerBlock = 8U;
constexpr uint32_t kTableCapacity = kCapacity / kTokensPerBlock;

struct QueueModel {
  explicit QueueModel(uint32_t depth) : sq(depth * 16U), cq(8U * 8U) {}
  std::vector<uint32_t> sq;
  std::vector<uint32_t> cq;
  uint32_t sq_db = 0U;
  uint32_t cq_db = 0U;
  uint64_t processed_pi = 0U;
  uint64_t produced_cq = 0U;
  uint32_t doorbells = 0U;
  bool timeout = false;
  bool error = false;
};

struct Fixture {
  std::vector<uint8_t> remote = std::vector<uint8_t>(kCapacity * kTokenBytes);
  std::vector<uint8_t> output = std::vector<uint8_t>(kCapacity * kTokenBytes);
  std::vector<uint64_t> address_table = std::vector<uint64_t>(kTableCapacity);
  std::vector<uint8_t> valid = std::vector<uint8_t>(kTableCapacity, 1U);
  std::vector<int32_t> indices = std::vector<int32_t>(kCapacity);
  std::vector<int32_t> offsets = std::vector<int32_t>(kCapacity);
  std::vector<int32_t> destinations = std::vector<int32_t>(kCapacity);
  alignas(64) std::array<KvafBatchGetCoreStatus, kOwners> status{};
  alignas(64) std::array<KvafBatchGetQueueState, kOwners * kJettys> queues{};
  alignas(64) std::array<KvafBatchGetDirectUrmaEntity, kOwners * kJettys> entities{};
  std::vector<std::unique_ptr<QueueModel>> transport;
  KvafExecuteBatchGetKernelArgs args{};
  static inline Fixture *current = nullptr;

  explicit Fixture(uint32_t depth = 64U) {
    current = this;
    AscendC::events = {};
    AscendC::doorbell_hook = RingDoorbell;
    for (uint32_t i = 0U; i < remote.size(); ++i) {
      remote[i] = static_cast<uint8_t>(i * 17U + i / 7U);
    }
    std::fill(output.begin(), output.end(), 0xA5U);
    for (uint32_t i = 0U; i < kTableCapacity; ++i) {
      address_table[i] = reinterpret_cast<uint64_t>(
          remote.data() + i * kTokensPerBlock * kTokenBytes);
    }
    for (uint32_t i = 0U; i < kCapacity; ++i) {
      indices[i] = static_cast<int32_t>(i / kTokensPerBlock);
      offsets[i] = static_cast<int32_t>((i % kTokensPerBlock) * kTokenBytes);
      destinations[i] = static_cast<int32_t>((i * 17U) % kCapacity);
    }
    for (uint32_t i = 0U; i < entities.size(); ++i) {
      transport.emplace_back(std::make_unique<QueueModel>(depth));
      auto &model = *transport.back();
      auto &entity = entities[i];
      entity.magic = kvaf_device::kEntityMagic;
      entity.version = kvaf_device::kEntityVersion;
      entity.remote_host_ddr_address = reinterpret_cast<uint64_t>(remote.data());
      entity.remote_host_ddr_size = remote.size();
      entity.remote_token_id = 1U;
      entity.remote_token_value = 2U;
      entity.local_hbm_address = reinterpret_cast<uint64_t>(output.data());
      entity.local_hbm_size = output.size();
      entity.local_token_id = 3U;
      entity.sq_va = reinterpret_cast<uint64_t>(model.sq.data());
      entity.sq_db_va = reinterpret_cast<uint64_t>(&model.sq_db);
      entity.sq_depth = depth;
      entity.wqe_size = 64U;
      entity.cq_va = reinterpret_cast<uint64_t>(model.cq.data());
      entity.cq_db_va = reinterpret_cast<uint64_t>(&model.cq_db);
      entity.cq_depth = 8U;
      entity.cqe_size = 32U;
    }
    args.address_table = reinterpret_cast<uint64_t>(address_table.data());
    args.valid_table = reinterpret_cast<uint64_t>(valid.data());
    args.indices = reinterpret_cast<uint64_t>(indices.data());
    args.offsets = reinterpret_cast<uint64_t>(offsets.data());
    args.output_indices = reinterpret_cast<uint64_t>(destinations.data());
    args.output = reinterpret_cast<uint64_t>(output.data());
    args.core_status = reinterpret_cast<uint64_t>(status.data());
    args.channel_entity = reinterpret_cast<uint64_t>(entities.data());
    args.queue_state = reinterpret_cast<uint64_t>(queues.data());
    args.table_capacity = kTableCapacity;
    args.request_capacity = kCapacity;
    args.output_rows = kCapacity;
    args.token_bytes = kTokenBytes;
    args.tokens_per_block = kTokensPerBlock;
    args.block_count = kOwners;
    args.jettys_per_core = kJettys;
  }

  static void RingDoorbell(uint32_t head, uint32_t *address) {
    for (uint32_t i = 0U; i < current->transport.size(); ++i) {
      auto &model = *current->transport[i];
      const auto &entity = current->entities[i];
      if (address != &model.sq_db) {
        continue;
      }
      ++model.doorbells;
      for (; model.processed_pi < head; ++model.processed_pi) {
        const uint32_t *words = model.sq.data() +
            (model.processed_pi % entity.sq_depth) * 16U;
        const uint32_t expected_owner =
            1U - static_cast<uint32_t>((model.processed_pi / entity.sq_depth) & 1U);
        assert((words[0] >> 31U) == expected_owner);
        assert((words[1] >> 8U) == 6U);
        const uint64_t remote = words[10] | (static_cast<uint64_t>(words[11]) << 32U);
        const uint64_t local = words[14] | (static_cast<uint64_t>(words[15]) << 32U);
        std::memcpy(reinterpret_cast<void *>(local),
                    reinterpret_cast<const void *>(remote), words[12]);
        if ((words[0] & (1U << 21U)) != 0U && !model.timeout) {
          const uint32_t cq_owner =
              1U - static_cast<uint32_t>((model.produced_cq / entity.cq_depth) & 1U);
          model.cq[(model.produced_cq % entity.cq_depth) * 8U] =
              (cq_owner << 2U) | (model.error ? 1U << 24U : 0U);
          ++model.produced_cq;
        }
      }
      return;
    }
  }

  uint32_t Doorbells() const {
    uint32_t count = 0U;
    for (const auto &queue : transport) {
      count += queue->doorbells;
    }
    return count;
  }

  void VerifyPayload(uint32_t begin, uint32_t count) const {
    for (uint32_t request = begin; request < begin + count; ++request) {
      const uint8_t *actual = output.data() + destinations[request] * kTokenBytes;
      const uint8_t *expected = remote.data() + request * kTokenBytes;
      assert(std::memcmp(actual, expected, kTokenBytes) == 0);
    }
  }

  ~Fixture() {
    for (const auto &from : AscendC::events) {
      for (const auto &to : from) {
        for (int event : to) {
          assert(event == 0);
        }
      }
    }
    AscendC::doorbell_hook = nullptr;
    current = nullptr;
  }
};

}  // namespace kvaf_host_model

#endif
