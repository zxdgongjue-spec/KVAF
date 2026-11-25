#pragma once
#include "cpu_sgl_push_abi.h"
#include "aicpu_urma/transport.h"

namespace kvaf::cpu_push {
inline std::uint32_t PostDescriptorMailbox(const MailboxLaunch& mailbox, const KernelArgs& a,
    const Layout& layout, std::uint32_t count, bool suffix, bool failed = false) {
  const auto head = std::min(count, kPipelineHeadRows);
  aicpu_urma::UmdkTransfer transfers[4]{};
  std::uint32_t size = 0;
  auto add = [&](std::uint64_t offset, std::uint64_t bytes) {
    if (bytes) transfers[size++] = {mailbox.remote_base + offset, a.workspace + offset, bytes};
  };
  if (suffix) {
    if (!failed) add(layout.descriptor + std::uint64_t{head} * 8, std::uint64_t{count - head} * 8);
    add(kSuffixNotifyOffset, 8);
  } else {
    add(layout.request, sizeof(Request));
    add(layout.output_target, sizeof(OutputTarget));
    add(layout.descriptor, std::uint64_t{head} * 8);
    add(layout.notify, 8);
  }
  const aicpu_urma::UmdkContext context{mailbox.session, mailbox.owner_pid};
  return aicpu_urma::UmdkTransferPost(&context, transfers, size, 0);
}
inline std::uint32_t RetireDescriptorMailbox(const MailboxLaunch& mailbox) {
  const aicpu_urma::UmdkContext context{mailbox.session, mailbox.owner_pid};
  return aicpu_urma::UmdkTransferWait(&context, 0);
}
}  // namespace kvaf::cpu_push
