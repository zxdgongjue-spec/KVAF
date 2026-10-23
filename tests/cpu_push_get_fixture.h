#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cpu_push_get_mock.h"
#include "cpu_push_get_transport.h"

inline void PushRequire(bool valid, const char* message) { if (!valid) throw std::runtime_error(message); }
struct PushFixture {
  std::uint32_t bytes, count, workers;
  kvaf::cpu_push::Layout layout;
  void* memory{};
  std::vector<std::uint8_t> source;
  urma_device_t device{};
  urma_context_t context{};
  urma_target_seg_t segment{};
  std::unique_ptr<kvaf::cpu_push::Transport> transport;
  explicit PushFixture(std::uint32_t row_bytes = 656, std::uint32_t rows = 2053, std::uint32_t threads = 4)
      : bytes(row_bytes), count(rows), workers(threads), source(std::size_t(rows) * row_bytes * 3) {
    using namespace kvaf::cpu_push;
    PushRequire(MakeLayout(bytes, workers, &layout), "bad fixture shape");
    PushRequire(posix_memalign(&memory, 4096, layout.bytes) == 0, "fixture allocation");
    std::memset(memory, 0, layout.staging + std::uint64_t{rows} * bytes);
    context.dev = &device;
    segment.seg.ubva.va = reinterpret_cast<std::uint64_t>(source.data());
    segment.seg.len = source.size();
    kvaf::peer::OpenCpuPushGetRequest open{};
    open.source_bytes = source.size();
    open.hbm_base = reinterpret_cast<std::uint64_t>(memory);
    open.hbm_bytes = layout.bytes;
    open.npu_tp = 1; open.npu_psn = 1; open.token_value = 1;
    open.segment_bytes = sizeof(urma_seg_t); open.token_bytes = bytes; open.workers = workers;
    urma_seg_t remote{}; remote.ubva.va = open.hbm_base; remote.len = open.hbm_bytes;
    std::memcpy(open.segment, &remote, sizeof(remote));
    for (auto& channel : open.channels) channel.bytes = 32;
    transport = std::make_unique<Transport>();
    kvaf::peer::OpenCpuPushGetResponse response{};
    const auto result = transport->Initialize(&context, &segment, segment.seg.ubva.va, open,
        &push_mock::QueryTp, &push_mock::ImportJetty, 1, &response);
    PushRequire(result.ok(), result.message().c_str());
  }
  ~PushFixture() {
    if (transport) { auto status = transport->Shutdown(); if (!status.ok()) std::terminate(); }
    std::free(memory);
  }
  template <class T> T* At(std::uint64_t offset) {
    return reinterpret_cast<T*>(static_cast<std::uint8_t*>(memory) + offset);
  }
  void Describe(std::uint64_t sequence) {
    using namespace kvaf::cpu_push;
    *At<Request>(layout.request) = {kMagic, sequence, count, bytes, workers, 0, {}};
    auto* spans = At<kvaf::peer::GatherSpan>(layout.descriptor);
    for (std::uint32_t row = 0; row < count; ++row)
      spans[row] = {reinterpret_cast<std::uint64_t>(source.data()) + std::uint64_t{count - row - 1} * bytes * 3, bytes, 0};
    At<Word>(layout.notify)->value = sequence;
  }
};
