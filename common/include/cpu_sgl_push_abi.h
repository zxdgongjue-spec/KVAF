#pragma once
#include "cpu_push_kernel_abi.h"

namespace kvaf::cpu_push {
constexpr std::uint32_t kPushChunkedOutput = 5;
constexpr std::uint32_t kPushChunkCount = (kMaxRequests + kChunkRows - 1) / kChunkRows;
constexpr std::uint32_t kPushControlWindow = 4;
// Registered HBM sources remain immutable until their control CQEs retire.
// Only the publisher touches sent/state; builders own the workspace Ready lines.
struct alignas(64) ChunkPushState {
  Word commits[kPushChunkCount + kMaxWorkers];
  Word sent[kMaxWorkers];
  Word state;  // value=sent chunks, reserved[0]=pending, [1]=notified, [2]=error sent.
};
// Independent device-push launch ABI; legacy Scatter/SGL entry layouts stay fixed.
struct MailboxLaunch {
  std::uint64_t remote_base{}, bytes{}, aiv_context{}, session{}, owner_pid{};
};
struct CpuSglPushArgs {
  CpuSglWriteArgs write;
  MailboxLaunch mailbox;
};
static_assert(sizeof(MailboxLaunch) == 40 && offsetof(CpuSglPushArgs, mailbox) == 320);
static_assert(sizeof(CpuSglPushArgs) == 384);
}  // namespace kvaf::cpu_push
