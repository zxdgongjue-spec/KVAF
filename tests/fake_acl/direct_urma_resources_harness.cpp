#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "fake_acl_control.h"
#include "kvaf/access_operator/a5/direct_urma_resources.h"

namespace {

using kvaf::access::a5::DirectUrmaExecutionDomain;
using kvaf::access::a5::DirectUrmaOpenSpec;
using kvaf::access::a5::DirectUrmaResources;
using kvaf::access::a5::DirectUrmaRole;

int Fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

DirectUrmaOpenSpec MakeSpec(std::vector<std::uint8_t>* backing) {
  DirectUrmaOpenSpec spec;
  spec.execution_domain = DirectUrmaExecutionDomain::kAicpu;
  spec.role = DirectUrmaRole::kClient;
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
  if (argc != 2) {
    return Fail("expected one fault-injection scenario");
  }
  const std::string scenario = argv[1];
  kvaf_fake_acl_set_scenario(scenario.c_str());
  std::vector<std::uint8_t> backing(8192);
  DirectUrmaResources resources;
  const auto open = resources.Open(MakeSpec(&backing));

  if (scenario == "sync_twice") {
    if (open.ok() || resources.transport_teardown_complete() ||
        kvaf_fake_acl_sync_calls() != 2 ||
        kvaf_fake_acl_free_calls() != 0 ||
        kvaf_fake_acl_destroy_stream_calls() != 0) {
      return Fail("double sync failure released or misclassified ownership");
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
    if (resources.Close().ok() || resources.transport_teardown_complete()) {
      return Fail("lost close result was incorrectly accepted as safe");
    }
    if (!resources.Close().ok() ||
        !resources.transport_teardown_complete() ||
        kvaf_fake_acl_close_launches() != 2 ||
        kvaf_fake_acl_open_session() !=
            kvaf_fake_acl_last_close_session()) {
      return Fail("same-session close retry did not converge");
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
