#pragma once
#include "kernel_operator.h"
#include "ubmem_batch_abi.h"

namespace kvaf::ubmem {
// Two-bank scheduling follows UnionCopy in vllm-ascend PR #18490 (1b180deb):
// https://github.com/vllm-project/vllm-ascend/pull/18490
// Shared by Get, Put and raw-address tests, with arbitrary row sizes within the existing 32 KiB UB.
// Event IDs 0/1 belong exclusively to this UBmem Get/Put kernel.
class MteCopyPipeline {
 public:
  __aicore__ inline explicit MteCopyPipeline(uint64_t row_bytes) {
    const uint32_t limit = kMteBufferBytes / 2;
    chunk_bytes_ = row_bytes < limit ? row_bytes : limit;
    const uint32_t bank_bytes = (chunk_bytes_ + 31U) / 32U * 32U;
    for (uint32_t bank = 0; bank < 2; ++bank) {
      buffers_[bank].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
      buffers_[bank].address_.bufferAddr = bank * bank_bytes;
      buffers_[bank].address_.dataLen = bank_bytes;
    }
  }

  __aicore__ inline void Copy(uint64_t dst, uint64_t src, uint64_t bytes) {
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    for (uint64_t offset = 0; offset < bytes;) {
      const uint32_t bank = issued_ % 2;
      const auto event = static_cast<event_t>(bank);
      // Only wait for the bank being reused; the other bank may still be busy.
      if (writing_[bank]) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        writing_[bank] = false;
      }
      const uint32_t length = bytes - offset < chunk_bytes_ ? bytes - offset : chunk_bytes_;
      AscendC::GlobalTensor<uint8_t> source;
      source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(src + offset));
      AscendC::DataCopyPad(buffers_[bank], source,
          AscendC::DataCopyExtParams{1, length, 0, 0, 0},
          AscendC::DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
      destinations_[bank] = dst + offset;
      lengths_[bank] = length;
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
      ++issued_;
      if (issued_ - written_ == 2) WriteOldest();
      offset += length;
    }
  }

  __aicore__ inline void Finish() {
    while (written_ < issued_) WriteOldest();
    for (uint32_t bank = 0; bank < 2; ++bank) {
      if (writing_[bank])
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(bank));
    }
    // Cross-MTE events protect UB reuse. This scalar fence also ensures the
    // final GM write is finished before GetSystemCycle / completion reporting.
    if (issued_ != 0) {
      AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
    }
  }

 private:
  __aicore__ inline void WriteOldest() {
    const uint32_t bank = written_ % 2;
    const auto event = static_cast<event_t>(bank);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
    AscendC::GlobalTensor<uint8_t> destination;
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destinations_[bank]));
    AscendC::DataCopyPad(destination, buffers_[bank],
        AscendC::DataCopyExtParams{1, lengths_[bank], 0, 0, 0});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    writing_[bank] = true;
    ++written_;
  }

  AscendC::LocalTensor<uint8_t> buffers_[2];
  uint64_t destinations_[2];
  uint32_t lengths_[2], chunk_bytes_;
  bool writing_[2] = {false, false};
  uint64_t issued_ = 0, written_ = 0;
};

}  // namespace kvaf::ubmem
