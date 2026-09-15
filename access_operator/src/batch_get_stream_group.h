#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "kvaf_aiv_batch_get.h"
#include "status.h"

namespace kvaf {

// The caller supplies resource bindings for each stream, using one shared or
// two independent transports. The caller stream runs shard 0; an owned stream
// runs shard 1. The fork/join also gives graph capture
// a path from the caller stream into the side stream and back.
// The owner must serialize submissions/replays and complete caller streams
// before teardown. This object does not own caller streams or request buffers.
class BatchGetStreamGroup {
 public:
  BatchGetStreamGroup() = default;
  BatchGetStreamGroup(const BatchGetStreamGroup&) = delete;
  BatchGetStreamGroup& operator=(const BatchGetStreamGroup&) = delete;

  Status Initialize(std::uint32_t stream_count) {
    if (stream_count != 1U && stream_count != 2U) {
      return Status::InvalidArgument("BatchGet stream count must be 1 or 2");
    }
    stream_count_ = stream_count;
    failed_ = false;
    if (stream_count == 1U) return Status::OK();
    // Ex events snapshot each record, so repeated eager calls can enqueue
    // without host synchronization or resetting a still-in-use event.
    if (aclrtCreateStream(&side_stream_) != ACL_SUCCESS ||
        aclrtCreateEventExWithFlag(&ready_, ACL_EVENT_SYNC) != ACL_SUCCESS ||
        aclrtCreateEventExWithFlag(&done_, ACL_EVENT_SYNC) != ACL_SUCCESS) {
      Destroy();
      return Status::Internal("failed to create BatchGet stream/events");
    }
    return Status::OK();
  }

  // Call only after caller work and captured graphs have finished. Also drain
  // the side stream in case a prior submission failed before the join.
  Status Drain() {
    if (side_stream_ != nullptr && aclrtSynchronizeStream(side_stream_) != ACL_SUCCESS) {
      return Status::Internal("failed to drain BatchGet side stream");
    }
    return Status::OK();
  }

  void Destroy() {
    if (side_stream_ != nullptr) (void)aclrtDestroyStream(side_stream_);
    if (done_ != nullptr) (void)aclrtDestroyEvent(done_);
    if (ready_ != nullptr) (void)aclrtDestroyEvent(ready_);
    side_stream_ = nullptr;
    ready_ = nullptr;
    done_ = nullptr;
    stream_count_ = 1U;
    failed_ = false;
  }

  // Each lane already points to its own entity/queue/status slice. Request and
  // output pointers stay global; the kernel partitions the live miss_count.
  Status Launch(const std::array<KvafExecuteBatchGetKernelArgs, 2>& lanes,
                aclrtStream caller) {
    if (failed_) {
      return Status::InvalidState("BatchGet stream group failed; complete outstanding work and reinitialize");
    }
    if (caller == nullptr || lanes[0].block_count == 0U ||
        (stream_count_ == 2U && (side_stream_ == nullptr || ready_ == nullptr || done_ == nullptr ||
                                lanes[1].block_count != lanes[0].block_count))) {
      return Status::InvalidArgument("invalid BatchGet stream group launch");
    }
    auto first = lanes[0];
    first.shard_id = 0U;
    first.shard_count = stream_count_;
    if (stream_count_ == 1U) {
      return KvafExecuteBatchGetLaunch(&first, caller) == KVAF_BATCH_GET_SUCCESS
                 ? Status::OK() : Status::Internal("failed to launch BatchGet");
    }

    auto second = lanes[1];
    second.shard_id = 1U;
    second.shard_count = stream_count_;

    if (aclrtRecordEvent(ready_, caller) != ACL_SUCCESS ||
        aclrtStreamWaitEvent(side_stream_, ready_) != ACL_SUCCESS) {
      return Fail("fork");
    }
    const int32_t side_result = KvafExecuteBatchGetLaunch(&second, side_stream_);
    const int32_t first_result = side_result == KVAF_BATCH_GET_SUCCESS
                                    ? KvafExecuteBatchGetLaunch(&first, caller)
                                    : side_result;
    // Join even after a launch error: a submitted sibling may still be using
    // request/output memory. Never let a later successful call reuse its queues.
    if (aclrtRecordEvent(done_, side_stream_) != ACL_SUCCESS ||
        aclrtStreamWaitEvent(caller, done_) != ACL_SUCCESS) {
      return Fail("join; synchronize the device before releasing request buffers");
    }
    if (side_result != KVAF_BATCH_GET_SUCCESS || first_result != KVAF_BATCH_GET_SUCCESS) {
      return Fail("kernel launch");
    }
    return Status::OK();
  }

 private:
  Status Fail(const char* stage) {
    failed_ = true;
    return Status::Internal(std::string("BatchGet dual-stream submission failed at ") + stage);
  }

  std::uint32_t stream_count_{1U};
  aclrtStream side_stream_{nullptr};
  aclrtEvent ready_{nullptr};
  aclrtEvent done_{nullptr};
  bool failed_{false};
};

}  // namespace kvaf
