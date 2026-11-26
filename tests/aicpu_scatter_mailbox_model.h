#pragma once

#include <cstring>
#include <vector>

#include "get/scatter/urma_aicpu/kernel/cpu_push_scatter.h"

namespace kvaf::cpu_push::test {

class Mailbox {
 public:
  void Bind(KernelArgs& args, const Layout& layout) {
    source_.resize(layout.staging / sizeof(Word));
    memory_.resize(layout.staging / sizeof(Word));
    args.mailbox_context = {1, 1};
    args.mailbox_source = reinterpret_cast<std::uint64_t>(source_.data());
    args.mailbox_remote = reinterpret_cast<std::uint64_t>(memory_.data());
  }

  std::uint8_t* data() { return reinterpret_cast<std::uint8_t*>(memory_.data()); }

  static std::uint32_t Write(const aicpu_urma::UmdkContext*, bool write,
                             const aicpu_urma::UmdkTransfer* transfers, std::uint32_t count,
                             std::uint32_t worker) {
    if (!write || worker || (count != 2 && count != 3)) return 1;
    const auto& marker = transfers[count - 1];
    if (marker.bytes != sizeof(std::uint64_t)) return 1;
    // Model ordered DMA and pair publication with the CPU producer's acquire.
    for (std::uint32_t i = 0; i < count - 1; ++i)
      std::memcpy(reinterpret_cast<void*>(transfers[i].remote),
                  reinterpret_cast<const void*>(transfers[i].local), transfers[i].bytes);
    const auto* source = reinterpret_cast<const std::uint64_t*>(marker.local);
    auto* notify = reinterpret_cast<std::uint64_t*>(marker.remote);
    __atomic_store_n(notify, *source, __ATOMIC_RELEASE);
    return 0;
  }

 private:
  std::vector<Word> source_, memory_;
};

}  // namespace kvaf::cpu_push::test
