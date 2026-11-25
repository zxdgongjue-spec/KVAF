#pragma once
#include "cpu_push_kernel_abi.h"

namespace kvaf::cpu_push {
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
