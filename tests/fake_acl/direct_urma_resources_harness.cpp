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
  spec.control_server_address = "127.0.0.1";
  spec.control_bind_address = "127.0.0.1";
  spec.local_address =
      reinterpret_cast<std::uint64_t>(backing->data());
  spec.local_allocation_bytes = backing->size();
  spec.segment_payload_bytes = 4096;
  spec.control_timeout_seconds = 10;
  if (spec.role == DirectUrmaRole::kServer) {
    spec.control_bind_address.clear();
  }
  if (scenario == "invalid_control_ip") {
    spec.control_server_address = "not-an-ip";
  } else if (scenario == "mixed_control_ip_family") {
    spec.control_bind_address = "::1";
  } else if (scenario == "server_control_bind") {
    spec.role = DirectUrmaRole::kServer;
  }
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

  if (scenario == "invalid_control_ip" ||
      scenario == "mixed_control_ip_family" ||
      scenario == "server_control_bind") {
    if (open.ok() || kvaf_fake_acl_open_session() != 0 ||
        kvaf_fake_acl_close_launches() != 0 ||
        kvaf_fake_acl_free_calls() != 0) {
      return Fail("invalid TCP control address acquired provider resources");
    }
    return 0;
  }

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
  if (scenario == "open_failed_safe_bad_result") {
    if (open.ok() || resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        resources.close_phase() !=
            DirectUrmaClosePhase::kUnsafeUnknown ||
        kvaf_fake_acl_free_calls() != 0 ||
        kvaf_fake_acl_destroy_stream_calls() != 0) {
      return Fail("contradictory OPEN_FAILED_SAFE evidence was trusted");
    }
    return 0;
  }
  if (scenario == "foreign_already_open") {
    if (open.ok() || !resources.transport_teardown_complete() ||
        kvaf_fake_acl_close_launches() != 0 ||
        kvaf_fake_acl_free_calls() != 5 ||
        kvaf_fake_acl_destroy_stream_calls() != 2) {
      return Fail("foreign session was closed or local state was retained");
    }
    return 0;
  }
  if (scenario == "clock_failed_before_open" ||
      scenario == "lease_expired_after_open") {
    const int expected_close_launches =
        scenario == "lease_expired_after_open" ? 1 : 0;
    if (open.ok() || !resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        kvaf_fake_acl_close_launches() != expected_close_launches ||
        kvaf_fake_acl_free_calls() != 5 ||
        kvaf_fake_acl_destroy_stream_calls() != 2) {
      return Fail("safe pre-READY terminal proof was not accepted");
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
    if (open.ok() || !resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        kvaf_fake_acl_close_launches() != 1 ||
        kvaf_fake_acl_open_session() !=
            kvaf_fake_acl_last_close_session() ||
        open.message().find("provider_result_status=7") ==
            std::string::npos ||
        open.message().find("provider_status=-3") ==
            std::string::npos ||
        open.message().find("owner_result_code=2") ==
            std::string::npos) {
      return Fail("failed Open did not teardown in its owner invocation");
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

  if (scenario == "ready_timeout_once") {
    if (!resources.Close().ok() ||
        !resources.transport_teardown_complete() ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("bounded READY poll retry relaunched or lost the owner");
    }
    return 0;
  }

  if (scenario == "data_stream_fifo") {
    kvaf::access::AivDirectUrmaChannel channel{};
    const auto local = resources.local_memory();
    const auto remote = resources.remote_memory();
    if (!resources.SnapshotDirectChannel(
             local.address, 4096, remote.address, 4096,
             &channel).ok()) {
      return Fail("could not snapshot candidate data channel");
    }
    kvaf::access::DeviceBatchGetParams params{};
    kvaf::access::DeviceCompletion completion{};
    if (!resources.EnqueueBatch(&params, &completion, &channel).ok() ||
        kvaf_fake_acl_data_launches() != 1 ||
        kvaf_fake_acl_data_stream_matches_control() != 1) {
      return Fail("batch was not enqueued on the owned control stream");
    }
    // Deliberately do not call SynchronizeDataPlane. Close must freeze new
    // submissions and append shutdown to this same FIFO stream.
    if (!resources.Close().ok() ||
        !resources.transport_teardown_complete() ||
        kvaf_fake_acl_close_launches() != 1 ||
        kvaf_fake_acl_data_stream_matches_control() != 1 ||
        resources.EnqueueBatch(&params, &completion, &channel).ok()) {
      return Fail("Close did not FIFO-order shutdown after accepted data");
    }
    return 0;
  }

  if (scenario == "close_d2h_retry") {
    if (resources.Close().ok() || resources.transport_teardown_complete() ||
        !resources.transport_teardown_retryable() ||
        resources.close_phase() !=
            DirectUrmaClosePhase::kAivShutdownResultPending ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kAivShutdownResultReadPending ||
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

  if (scenario == "close_status_d2h_retry") {
    if (resources.Close().ok() ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kAicpuEpochOwnerStatusReadPending ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("lost terminal status did not retain read cursor");
    }
    if (!resources.Close().ok() ||
        !resources.transport_teardown_complete() ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("terminal status reread rejoined or relaunched owner");
    }
    return 0;
  }

  if (scenario == "close_prelaunch_h2d_retry") {
    if (resources.Close().ok() || resources.transport_teardown_complete() ||
        !resources.transport_teardown_retryable() ||
        resources.close_phase() !=
            DirectUrmaClosePhase::kAivShutdownPreLaunch ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kAivShutdownPreLaunchRetryable ||
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
      scenario == "owner_join_sync_twice") {
    if (resources.Close().ok() ||
        resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        resources.close_phase() !=
            DirectUrmaClosePhase::kUnsafeUnknown ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kUnsafeUnknown) {
      return Fail("ambiguous pre-provider Close was not marked unsafe");
    }
    const int expected_launches = 1;
    if (kvaf_fake_acl_close_launches() != expected_launches ||
        resources.Close().ok() ||
        kvaf_fake_acl_close_launches() != expected_launches) {
      return Fail("unsafe Close was touched again");
    }
    return 0;
  }

  if (scenario == "close_unsafe" ||
      scenario == "close_terminal_incomplete" ||
      scenario == "close_command_reserved_nonzero") {
    if (resources.Close().ok() ||
        resources.transport_teardown_complete() ||
        resources.transport_teardown_retryable() ||
        kvaf_fake_acl_close_launches() != 1 ||
        resources.close_disposition() !=
            DirectUrmaCloseDisposition::kUnsafeUnknown) {
      return Fail("incomplete epoch teardown proof was not retained");
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
        kvaf_fake_acl_destroy_stream_calls() != 3) {
      return Fail("stream destroy retry repeated provider teardown");
    }
    return 0;
  }

  if (scenario == "sync_retry") {
    if (!resources.Close().ok() || kvaf_fake_acl_sync_calls() != 4 ||
        kvaf_fake_acl_close_launches() != 1) {
      return Fail("bounded stream sync retry did not converge");
    }
    return 0;
  }

  return Fail("unknown scenario");
}
