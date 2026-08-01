#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "fake_acl_control.h"
#include "kvaf/access_operator/a5/direct_urma_resources.h"

namespace {

using kvaf::access::a5::DirectUrmaExecutionDomain;
using kvaf::access::a5::DirectUrmaCloseDisposition;
using kvaf::access::a5::DirectUrmaClosePhase;
using kvaf::access::a5::DirectUrmaOpenSpec;
using kvaf::access::a5::DirectUrmaResources;
using kvaf::access::a5::DirectUrmaRole;

int Fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

DirectUrmaOpenSpec MakeSpec(std::vector<std::uint8_t>* backing,
                            const std::string& scenario,
                            const std::string& provider_library) {
  DirectUrmaOpenSpec spec;
  const bool host = scenario.rfind("host_", 0) == 0;
  spec.execution_domain = host
      ? DirectUrmaExecutionDomain::kHostProcess
      : DirectUrmaExecutionDomain::kAicpu;
  spec.role = DirectUrmaRole::kClient;
  spec.provider_library_path = provider_library;
  spec.provider_soname = "libkvaf_urma_provider_device.so";
  spec.urma_device = "udmac0d1e4";
  spec.local_eid = "0000:0000:0000:0200:0010:0000:df00:4100";
  spec.peer_eid = "0000:0000:007f:0400:0010:0000:df0a:0b00";
  spec.local_address =
      reinterpret_cast<std::uint64_t>(backing->data());
  spec.local_allocation_bytes = backing->size();
  spec.segment_payload_bytes = 4096;
  spec.control_timeout_seconds = 10;
  return spec;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    return Fail("expected a fault-injection scenario and provider library");
  }
  const std::string scenario = argv[1];
  kvaf_fake_acl_set_scenario(scenario.c_str());
  std::vector<std::uint8_t> backing(8192);
  DirectUrmaResources resources;
  const auto open = resources.Open(MakeSpec(&backing, scenario, argv[2]));

  if (scenario == "sync_twice") {
    if (open.ok() || resources.transport_teardown_complete() ||
        kvaf_fake_acl_sync_calls() != 2 ||
        kvaf_fake_acl_free_calls() != 0 ||
        kvaf_fake_acl_destroy_stream_calls() != 0) {
      return Fail("double sync failure released or misclassified ownership");
    }
    return 0;
  }
  if (scenario == "open_launch_failure") {
    if (open.ok() || resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        resources.close_phase() !=
            DirectUrmaClosePhase::kUnsafeUnknown ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kUnsafeUnknown ||
        kvaf_fake_acl_close_launches() != 0 ||
        kvaf_fake_acl_free_calls() != 0 ||
        kvaf_fake_acl_destroy_stream_calls() != 0) {
      return Fail("ambiguous Open launch released provider backing");
    }
    return 0;
  }
  if (scenario == "foreign_already_open") {
    if (open.ok() || !resources.transport_teardown_complete() ||
        kvaf_fake_acl_close_launches() != 0 ||
        kvaf_fake_acl_free_calls() != 2 ||
        kvaf_fake_acl_destroy_stream_calls() != 1) {
      return Fail("foreign session was closed or local state was retained");
    }
    return 0;
  }
  if (scenario == "open_d2h_failure") {
    if (open.ok() || !resources.transport_teardown_complete() ||
        kvaf_fake_acl_close_launches() != 1 ||
        kvaf_fake_acl_open_session() !=
            kvaf_fake_acl_last_close_session()) {
      return Fail("open D2H failure did not close its exact session");
    }
    return 0;
  }
  if (scenario == "open_cleanup_retryable") {
    if (open.ok() || resources.transport_teardown_complete() ||
        !resources.transport_teardown_retryable() ||
        resources.close_phase() !=
            DirectUrmaClosePhase::kProviderRetryable ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kProviderRetryable ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("retryable failed Open was promoted to unsafe or lost");
    }
    if (!resources.Close().ok() ||
        !resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        kvaf_fake_acl_close_launches() != 2 ||
        kvaf_fake_acl_open_session() !=
            kvaf_fake_acl_last_close_session()) {
      return Fail("same-session failed Open cleanup retry did not converge");
    }
    return 0;
  }
  if (scenario == "host_open_cleanup_retryable") {
    if (open.ok() || resources.transport_teardown_complete() ||
        !resources.transport_teardown_retryable() ||
        resources.close_phase() !=
            DirectUrmaClosePhase::kProviderRetryable ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kProviderRetryable ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("Host retryable failed Open was promoted or lost");
    }
    if (!resources.Close().ok() ||
        !resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        kvaf_fake_acl_close_launches() != 2 ||
        kvaf_fake_acl_open_session() !=
            kvaf_fake_acl_last_close_session()) {
      return Fail("Host same-session failed Open retry did not converge");
    }
    return 0;
  }
  if (scenario == "runtime_capability_partial" ||
      scenario == "runtime_public_abi_missing") {
    if (open.ok() || !resources.transport_teardown_complete() ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("unproved runtime contract was accepted or not closed");
    }
    return 0;
  }
  if (!open.ok()) {
    std::cerr << open.message() << '\n';
    return Fail("scenario unexpectedly failed Open");
  }

  if (scenario == "kernel_type") {
    if (kvaf_fake_acl_last_kernel_type() != 5U) {
      return Fail("AICPU launch did not use the KFC kernel-type ABI value");
    }
    if (!resources.Close().ok()) {
      return Fail("kernel-type scenario failed Close");
    }
    return 0;
  }

  if (scenario == "close_d2h_retry") {
    if (resources.Close().ok() || resources.transport_teardown_complete() ||
        !resources.transport_teardown_retryable() ||
        resources.close_phase() !=
            DirectUrmaClosePhase::kAicpuResultPending ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kAicpuResultReadPending ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("lost close result did not retain its read-only cursor");
    }
    if (!resources.Close().ok() ||
        !resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        resources.close_phase() != DirectUrmaClosePhase::kComplete ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kTransportComplete ||
        kvaf_fake_acl_close_launches() != 1 ||
        kvaf_fake_acl_open_session() !=
            kvaf_fake_acl_last_close_session()) {
      return Fail("same-buffer close result reread did not converge");
    }
    return 0;
  }

  if (scenario == "close_d2h_provider_retryable") {
    if (resources.Close().ok() ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kAicpuResultReadPending ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("lost RETRYABLE result did not retain read cursor");
    }
    if (resources.Close().ok() ||
        resources.close_phase() !=
            DirectUrmaClosePhase::kProviderRetryable ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kProviderRetryable ||
        !resources.transport_teardown_retryable() ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("result reread did not expose provider RETRYABLE");
    }
    if (!resources.Close().ok() ||
        !resources.transport_teardown_complete() ||
        kvaf_fake_acl_close_launches() != 2) {
      return Fail("provider-authorized relaunch did not converge");
    }
    return 0;
  }

  if (scenario == "close_prelaunch_d2h_retry") {
    if (resources.Close().ok() || resources.transport_teardown_complete() ||
        !resources.transport_teardown_retryable() ||
        resources.close_phase() !=
            DirectUrmaClosePhase::kAicpuPreLaunch ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kPreLaunchRetryable ||
        kvaf_fake_acl_close_launches() != 0) {
      return Fail("safe pre-launch copy failure was misclassified");
    }
    if (!resources.Close().ok() ||
        !resources.transport_teardown_complete() ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("safe pre-launch Close retry did not converge");
    }
    return 0;
  }

  if (scenario == "close_queue_not_drained" ||
      scenario == "close_launch_failure") {
    if (resources.Close().ok() ||
        resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        resources.close_phase() !=
            DirectUrmaClosePhase::kUnsafeUnknown ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kUnsafeUnknown) {
      return Fail("ambiguous pre-provider Close was not marked unsafe");
    }
    const int expected_launches =
        scenario == "close_launch_failure" ? 1 : 0;
    if (kvaf_fake_acl_close_launches() != expected_launches ||
        resources.Close().ok() ||
        kvaf_fake_acl_close_launches() != expected_launches) {
      return Fail("unsafe Close was touched again");
    }
    return 0;
  }

  if (scenario == "close_unsafe") {
    if (resources.Close().ok() ||
        resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        kvaf_fake_acl_close_launches() != 1 ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kUnsafeUnknown) {
      return Fail("unsafe provider Close was not retained as unknown");
    }
    if (resources.Close().ok() ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("unsafe provider Close was incorrectly retried");
    }
    return 0;
  }

  if (scenario == "host_close_unsafe") {
    if (resources.Close().ok() ||
        resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        kvaf_fake_acl_close_launches() != 1 ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kUnsafeUnknown) {
      return Fail("unsafe Host provider Close was not retained as unknown");
    }
    if (resources.Close().ok() ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("unsafe Host provider Close was incorrectly retried");
    }
    return 0;
  }

  if (scenario == "free_retry") {
    if (resources.Close().ok() ||
        !resources.transport_teardown_complete()) {
      return Fail("local free failure changed transport safety");
    }
    if (!resources.Close().ok() ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("local free retry repeated provider teardown");
    }
    return 0;
  }

  if (scenario == "destroy_stream_retry") {
    if (resources.Close().ok() ||
        !resources.transport_teardown_complete()) {
      return Fail("stream handle failure changed transport safety");
    }
    if (!resources.Close().ok() ||
        kvaf_fake_acl_close_launches() != 1 ||
        kvaf_fake_acl_destroy_stream_calls() != 2) {
      return Fail("stream destroy retry repeated provider teardown");
    }
    return 0;
  }

  if (scenario == "sync_retry") {
    if (!resources.Close().ok() || kvaf_fake_acl_sync_calls() != 3 ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("bounded stream sync retry did not converge");
    }
    return 0;
  }

  return Fail("unknown scenario");
}
