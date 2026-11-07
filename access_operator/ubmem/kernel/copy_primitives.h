#pragma once

#include "kernel_operator.h"
#define KVAF_UBMEM_INLINE __aicore__ inline
#include "ubmem/transfer_abi.h"
#undef KVAF_UBMEM_INLINE

namespace kvaf_ubmem_device {
using namespace AscendC;
using namespace kvaf_ubmem;

// Eight exact-length DMA stages overlap Host-DDR reads with HBM writes.
// Destination owners serialize duplicate rows and shared partial sectors.
// Drain every credit before returning, including on a validation failure.
template <uint32_t StageBytes>
class CopyPipeline {
 public:
  __aicore__ inline CopyPipeline() {
    pipe_.InitBuffer(payload_, kDepth * kStageBytes);
    pipe_.InitBuffer(control_, 32);
  }

  // Descriptors/tables live in HBM and can change between graph replays.
  // Invalidate once per launch, then use scalar cache lines rather than one
  // MTE2 transfer plus a scalar wait for each 1/4/8-byte field.
  __aicore__ inline void InvalidateMetadata(uint64_t address) {
    GlobalTensor<uint8_t> metadata;
    metadata.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(address));
    DataCacheCleanAndInvalid<uint8_t, CacheLine::ENTIRE_DATA_CACHE>(metadata);
  }

  template <typename T>
  __aicore__ inline T Load(uint64_t address) {
    GlobalTensor<T> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
    return source.GetValue(0);
  }

  __aicore__ inline void Copy(uint64_t source_address,
                              uint64_t destination_address, uint64_t bytes) {
    if (!issued_) {
      SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
      for (uint32_t i = 0; i < kDepth; ++i)
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i));
    }
    for (uint64_t offset = 0; offset < bytes;) {
      const uint32_t size = static_cast<uint32_t>(
          bytes - offset > kStageBytes ? kStageBytes : bytes - offset);
      const uint32_t stage = issued_ % kDepth;
      WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(stage));
      GlobalTensor<uint8_t> source;
      source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(source_address + offset));
      DataCopyExtParams copy{1, size, 0, 0, 0};
      DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
      DataCopyPad<uint8_t, PaddingMode::Normal>(
          payload_.Get<uint8_t>()[stage * kStageBytes], source, copy, pad);
      SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(stage));
      destinations_[stage] = destination_address + offset;
      sizes_[stage] = size;
      ++issued_;
      if (issued_ - stored_ == kDepth) StoreOne();
      offset += size;
    }
  }

  __aicore__ inline void Finish() {
    if (!issued_) return;
    while (stored_ < issued_) StoreOne();
    for (uint32_t i = 0; i < kDepth; ++i)
      WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(i));
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    issued_ = stored_ = 0;
  }

  __aicore__ inline void WriteStatus(uint64_t address, uint64_t result,
                                      uint64_t failed, uint64_t copied) {
    if (address == 0) return;
    auto local = control_.Get<uint64_t>();
    local.SetValue(0, kStatusMagic);
    local.SetValue(1, result);
    local.SetValue(2, failed);
    local.SetValue(3, copied);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    GlobalTensor<uint64_t> destination;
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(address));
    DataCopyExtParams copy{1, sizeof(CoreStatus), 0, 0, 0};
    DataCopyPad<uint64_t, PaddingMode::Normal>(destination, local, copy);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }

 private:
  static constexpr uint32_t kDepth = 8;
  static constexpr uint32_t kStageBytes = StageBytes;
  __aicore__ inline void StoreOne() {
    const uint32_t stage = stored_ % kDepth;
    WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(stage));
    GlobalTensor<uint8_t> destination;
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destinations_[stage]));
    DataCopyExtParams copy{1, sizes_[stage], 0, 0, 0};
    DataCopyPad<uint8_t, PaddingMode::Normal>(
        destination, payload_.Get<uint8_t>()[stage * kStageBytes], copy);
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(stage));
    ++stored_;
  }
  uint64_t destinations_[kDepth]{};
  uint32_t sizes_[kDepth]{};
  uint64_t issued_{0}, stored_{0};
  TPipe pipe_;
  TBuf<TPosition::VECCALC> payload_, control_;
};
using GetCopier = CopyPipeline<2624>;
using ByteCopier = CopyPipeline<kCopyChunkBytes>;
}  // namespace kvaf_ubmem_device
