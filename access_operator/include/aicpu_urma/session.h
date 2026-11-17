#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "execution_resources.h"
#include "peer_link.h"
#include "aicpu_urma/control_abi.h"

namespace kvaf { struct PutProviderTiming; }

namespace kvaf::aicpu_urma {

struct UmdkSessionConfig {
  std::string kernel_json;
  std::uint32_t workers{1U};
  std::shared_ptr<peer::PeerLink> peer_link;
  std::array<std::uint8_t, 16> local_eid{};
  UmdkQueueProfile queue_profile{UmdkQueueProfile::Gather};
};

class UmdkSession;
// Narrow execution capability: no session setup, registration or shutdown API.
class UmdkWriteView {
 public:
  UmdkWriteView() = default;
  using Launch = int32_t (*)(const UmdkControlArgs*, aclrtStream);
  Status Submit(std::uint64_t, std::uint32_t, aclrtStream, Launch,
                PutProviderTiming* timing = nullptr) const;
 private:
  friend class UmdkSession;
  explicit UmdkWriteView(UmdkSession* session) : session_(session) {}
  UmdkSession* session_{};
};

class UmdkSession {
 public:
  explicit UmdkSession(UmdkSessionConfig config) : config_(std::move(config)) {}
  UmdkWriteView WriteView() { return UmdkWriteView(this); }
  Status Load();
  Status Ensure(std::uint64_t region_base, void* output, std::size_t output_bytes,
                bool prepared_only = false);
  // Establish transport once without re-registering a released setup buffer.
  Status EnsureSession(std::uint64_t region_base, void* setup_buffer, std::size_t setup_bytes);
  Status Shutdown();
  // A failed launch/completion blocks reuse until Shutdown succeeds.
  Status Drain();
  Status ReleaseBuffers();
  Status ReleaseBuffer(void* buffer, std::size_t bytes);
  struct WriteState {
    bool attempted{false};
    PutCompletion completion;
  };
  void BindWriteState(WriteState* state) { write_ = state ? state : &default_write_; }
  Status ResolveWrite(PutCompletion*);

  using Launch = int32_t (*)(const UmdkControlArgs*, aclrtStream);
  Status WriteDescriptors(std::uint64_t descriptors, std::uint32_t count, aclrtStream stream,
                          Launch, PutProviderTiming* timing = nullptr);
  const UmdkSessionConfig& config() const { return config_; }
  std::uint64_t session() const { return args_.session; }
  std::uint64_t owner_pid() const { return args_.owner_pid; }

 private:
  Status Run(UmdkControlOp op, UmdkControlStatus& status, aclrtStream stream = nullptr,
             Launch launch = nullptr, PutProviderTiming* timing = nullptr);
  Status CompletePending(UmdkControlStatus& status);
  UmdkSessionConfig config_;
  UmdkControlArgs args_{};
  aclrtStream stream_{nullptr};
  aclrtStream pending_{nullptr};
  bool connected_{false}, failed_{false};
  bool kernel_loaded_{false};
  std::vector<std::pair<std::uint64_t, std::size_t>> registered_buffers_;
  WriteState default_write_;
  WriteState* write_{&default_write_};
};

inline Status UmdkWriteView::Submit(std::uint64_t descriptors, std::uint32_t count,
                                  aclrtStream stream, Launch launch, PutProviderTiming* timing) const {
  return session_ ? session_->WriteDescriptors(descriptors, count, stream, launch, timing)
      : Status::InvalidState("empty UMDK write capability");
}

}  // namespace kvaf::aicpu_urma
