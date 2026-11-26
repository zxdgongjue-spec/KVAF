#pragma once
#include "cpu_sgl_push_layout.h"
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
inline std::uint32_t PostChunkDescriptorMailbox(const MailboxLaunch& mailbox, const KernelArgs& a,
    const Layout& layout, std::uint32_t count, std::uint32_t chunk, std::uint32_t owner, bool first, bool failed, std::uint32_t head = kChunkRows) {
  aicpu_urma::UmdkTransfer transfers[5]{};
  std::uint32_t size = 0;
  auto add = [&](std::uint64_t local, std::uint64_t remote, std::uint64_t bytes) {
    if (bytes) transfers[size++] = {mailbox.remote_base + remote, a.workspace + local, bytes};
  };
  if (first) { add(layout.request, layout.request, sizeof(Request)); add(layout.output_target, layout.output_target, sizeof(OutputTarget)); }
  if (!failed) {
    const auto begin = PushChunkBegin(chunk, head), end = PushChunkEnd(count, chunk, head);
    add(layout.descriptor + std::uint64_t(begin) * 8, layout.descriptor + std::uint64_t(begin) * 8, std::uint64_t(end - begin) * 8);
  }
  const auto slot = failed ? kPushChunkCount + owner : chunk;
  add(layout.staging + (failed ? std::uint64_t(slot) * sizeof(Word) : PushCommitOffset(chunk)), layout.ready + std::uint64_t(owner) * sizeof(Word), 8);
  if (first) add(layout.notify, layout.notify, 8);
  const aicpu_urma::UmdkContext context{mailbox.session, mailbox.owner_pid};
  return aicpu_urma::UmdkTransferPost(&context, transfers, size, 0);
}
}  // namespace kvaf::cpu_push
