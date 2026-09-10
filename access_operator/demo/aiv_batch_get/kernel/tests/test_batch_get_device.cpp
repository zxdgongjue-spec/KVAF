#include "batch_get_model.h"

namespace {
using namespace kvaf_host_model;

void TestWindowedPrefixAndPersistentQueues() {
  Fixture fixture;
  for (uint32_t repeat = 0U; repeat < 3U; ++repeat) {
    const uint64_t previous_pi = fixture.queues[2].sq_producer_index;
    assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 2048U) ==
           kvaf_device::kSuccess);
    fixture.VerifyPayload(7U, 2048U);
    assert(fixture.status[1].processed_request_count == 2048U);
    assert(fixture.queues[2].sq_producer_index > previous_pi);
    assert(fixture.queues[2].last_error == 0U);
    assert(fixture.queues[3].last_error == 0U);
    assert(fixture.queues[0].initialized == 0U);
    assert(fixture.queues[1].initialized == 0U);
    assert(fixture.status[0].magic == 0U);
  }
}

void TestFullPrefixValidationBeforeFirstPost() {
  Fixture fixture;
  fixture.indices[2054] = -1;
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 2048U) ==
         kvaf_device::kInvalidLookup);
  assert(fixture.Doorbells() == 0U);
  assert(fixture.status[1].failed_request_index == 2054U);
  assert(std::all_of(fixture.output.begin(), fixture.output.end(),
                     [](uint8_t value) { return value == 0xA5U; }));
  fixture.indices[2054] = 2054 / kTokensPerBlock;
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 2048U) ==
         kvaf_device::kSuccess);
}

void TestTimeoutOrErrorDrainsPeerAndPoisonsQueue(bool timeout) {
  Fixture fixture;
  fixture.transport[2]->timeout = timeout;
  fixture.transport[2]->error = !timeout;
  const uint64_t expected =
      timeout ? kvaf_device::kCqTimeout : kvaf_device::kCqError;
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 2048U) == expected);
  assert(fixture.status[1].processed_request_count == 63U);
  assert(fixture.queues[3].cq_consumer_index == 1U);
  assert(fixture.queues[2].last_error == expected);
  assert(fixture.queues[3].last_error == expected);
  const uint32_t posted = fixture.Doorbells();
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 2048U) ==
         kvaf_device::kQueuePoisoned);
  assert(fixture.Doorbells() == posted);
  assert(fixture.queues[2].last_error == expected);
}

void TestEmptyAndInvalidRanges() {
  Fixture fixture;
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, kCapacity, 0U) ==
         kvaf_device::kSuccess);
  assert(fixture.Doorbells() == 0U);
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, kCapacity, 1U) ==
         kvaf_device::kInvalidArgs);
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, UINT32_MAX, 2U) ==
         kvaf_device::kInvalidArgs);
  assert(fixture.Doorbells() == 0U);
  assert(!kvaf_device::BatchGetRequestRangeValid(UINT32_MAX, UINT32_MAX, 1U));
  assert(kvaf_device::BatchGetWindowRequestCount(UINT32_MAX, UINT32_MAX) ==
         UINT32_MAX - 1U);
}

void TestSmallQueue() {
  Fixture fixture(2U);
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 65U) ==
         kvaf_device::kSuccess);
  fixture.VerifyPayload(7U, 65U);
  assert(fixture.queues[2].completed_batch_count == 65U);
}
}  // namespace

void TestAsyncFinalWindow() {
  Fixture fixture;
  kvaf_device::AsyncBatchGet pending;
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 2048U, &pending) == 0U);
  assert(pending.used_jettys != 0U);
  const uint32_t doorbells = fixture.Doorbells();
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 1U, &pending) ==
         kvaf_device::kInvalidQueue);
  assert(fixture.Doorbells() == doorbells);
  assert(kvaf_device::KvafBatchGetWait(pending) == 0U);
  assert(pending.used_jettys == 0U);
  fixture.VerifyPayload(7U, 2048U);
  assert(kvaf_device::KvafBatchGetWait(pending) == 0U);
}

void TestAsyncTimeoutPoisonsOwner() {
  Fixture fixture(4096U);
  fixture.transport[2]->timeout = true;
  kvaf_device::AsyncBatchGet pending;
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 64U, &pending) == 0U);
  assert(kvaf_device::KvafBatchGetWait(pending) == kvaf_device::kCqTimeout);
  assert(pending.used_jettys == 2U);
  assert(fixture.queues[3].completed_batch_count == 1U);
  assert(fixture.queues[2].last_error != 0U && fixture.queues[3].last_error != 0U);
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 1U, &pending) ==
         kvaf_device::kCqTimeout);
}

void TestInvocationEntityCacheKeepsDynamicValidation() {
  Fixture fixture;
  kvaf_device::AsyncBatchGet pending;
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 65U, &pending, true) == 0U);
  assert(pending.validated_entities != 0U && pending.validated_sq_depth >= 2U);
  assert(kvaf_device::KvafBatchGetWait(pending) == 0U);
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 65U, &pending, true) == 0U);
  assert(kvaf_device::KvafBatchGetWait(pending) == 0U);
  fixture.VerifyPayload(7U, 65U);
  const auto posted = fixture.Doorbells();
  fixture.indices[7] = -1;
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 65U, &pending, true) ==
         kvaf_device::kInvalidLookup);
  assert(fixture.Doorbells() == posted);
  fixture.indices[7] = 0;
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, kCapacity, 1U, &pending, true) ==
         kvaf_device::kInvalidArgs);
  fixture.queues[2].last_error = kvaf_device::kCqTimeout;
  assert(kvaf_device::KvafBatchGetDevice(fixture.args, 1U, 7U, 1U, &pending, true) ==
         kvaf_device::kQueuePoisoned);
  assert(fixture.Doorbells() == posted);
}

int main() {
  TestInvocationEntityCacheKeepsDynamicValidation();
  TestAsyncFinalWindow();
  TestAsyncTimeoutPoisonsOwner();
  TestWindowedPrefixAndPersistentQueues();
  TestFullPrefixValidationBeforeFirstPost();
  TestTimeoutOrErrorDrainsPeerAndPoisonsQueue(true);
  TestTimeoutOrErrorDrainsPeerAndPoisonsQueue(false);
  TestEmptyAndInvalidRanges();
  TestSmallQueue();
  std::puts("PASS: shared BatchGet CPU control-flow model (9 scenarios; no NPU/transport execution)");
}
