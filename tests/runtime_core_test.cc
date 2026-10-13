#include "runtime.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Lifecycle {
  bool operator_live{false};
  bool premature_release{false};
  int allocations{0};
  int releases{0};
  int operator_initializations{0};
  int operator_shutdowns{0};
  int registrations{0};
  int shutdown_failures{0};
  bool fail_operator_initialize{false};
  bool fail_registration{false};
  std::vector<std::string> events;
};

class FakeAllocator final : public kvaf::DdrMemoryAllocator {
 public:
  explicit FakeAllocator(Lifecycle& lifecycle) : lifecycle_(lifecycle) {}

  kvaf::Status Allocate(const kvaf::RuntimeInitRequest& request,
                        kvaf::RemoteRegionDesc* region) override {
    ++lifecycle_.allocations;
    lifecycle_.events.push_back("allocate");
    bytes_.resize(request.ddr_bytes);
    *region = {reinterpret_cast<kvaf::Address>(bytes_.data()),
               request.ddr_bytes, request.slot_size, request.ddr_device};
    return kvaf::Status::OK();
  }

  kvaf::Status Release(const kvaf::RemoteRegionDesc& region) override {
    ++lifecycle_.releases;
    lifecycle_.events.push_back("release");
    if (lifecycle_.operator_live) {
      lifecycle_.premature_release = true;
      return kvaf::Status::Internal("DDR released before operator retirement");
    }
    if (bytes_.empty() ||
        region.base != reinterpret_cast<kvaf::Address>(bytes_.data()) ||
        region.length != bytes_.size()) {
      return kvaf::Status::InvalidState("DDR allocation identity changed");
    }
    bytes_.clear();
    return kvaf::Status::OK();
  }

 private:
  Lifecycle& lifecycle_;
  std::vector<std::uint8_t> bytes_;
};

class FakeAccessOperator final : public kvaf::AccessOperator {
 public:
  explicit FakeAccessOperator(Lifecycle& lifecycle) : lifecycle_(lifecycle) {}

  kvaf::Status Initialize(const kvaf::RemoteRegionDesc&) override {
    ++lifecycle_.operator_initializations;
    lifecycle_.events.push_back("operator_initialize");
    // An error may occur after memory registration or queue creation.
    lifecycle_.operator_live = true;
    if (lifecycle_.fail_operator_initialize) {
      return kvaf::Status::NotImplemented("operator initialize failed");
    }
    return kvaf::Status::OK();
  }

  kvaf::Status Shutdown() override {
    ++lifecycle_.operator_shutdowns;
    lifecycle_.events.push_back("operator_shutdown");
    if (lifecycle_.shutdown_failures > 0) {
      --lifecycle_.shutdown_failures;
      return kvaf::Status::Busy("operator still owns registered DDR");
    }
    lifecycle_.operator_live = false;
    return kvaf::Status::OK();
  }

  kvaf::Status ExecuteBatchGet(const kvaf::BatchGetExecRequest&) override {
    return kvaf::Status::Internal("unexpected BatchGet");
  }
  kvaf::Status ExecuteBatchPut(const kvaf::BatchPutExecRequest&) override {
    return kvaf::Status::Internal("unexpected BatchPut");
  }

 private:
  Lifecycle& lifecycle_;
};

class FakeMetaManager final : public kvaf::MetaManager {
 public:
  explicit FakeMetaManager(Lifecycle& lifecycle) : lifecycle_(lifecycle) {}

  kvaf::Status RegisterRemoteRegion(const kvaf::RemoteRegionDesc&) override {
    ++lifecycle_.registrations;
    lifecycle_.events.push_back("register");
    if (lifecycle_.fail_registration) {
      return kvaf::Status::NotFound("metadata registration failed");
    }
    return kvaf::Status::OK();
  }

  kvaf::Status PreparePut(const std::vector<kvaf::KeyId>&,
                          const std::vector<std::uint64_t>&,
                          kvaf::PutContext*, kvaf::PutAccessDesc*) override {
    return Unexpected();
  }
  kvaf::Status CommitPut(const kvaf::PutContext&) override { return Unexpected(); }
  kvaf::Status AbortPut(const kvaf::PutContext&) override { return Unexpected(); }
  kvaf::Status LockKeys(const std::vector<kvaf::KeyId>&,
                        kvaf::MetaManagerLockMode,
                        kvaf::LockContext*) override {
    return Unexpected();
  }
  kvaf::Status BatchLookup(const std::vector<kvaf::KeyId>&,
                           const kvaf::LockContext&,
                           kvaf::FlatAccessDesc*) override {
    return Unexpected();
  }
  kvaf::Status UnlockKeys(const kvaf::LockContext&) override {
    return Unexpected();
  }
  kvaf::Status BatchFree(const std::vector<kvaf::KeyId>&) override {
    return Unexpected();
  }

 private:
  static kvaf::Status Unexpected() {
    return kvaf::Status::Internal("unexpected metadata operation");
  }
  Lifecycle& lifecycle_;
};

void TestInitializeCleanup(bool fail_operator, bool fail_cleanup) {
  Lifecycle lifecycle;
  lifecycle.fail_operator_initialize = fail_operator;
  lifecycle.fail_registration = !fail_operator;
  lifecycle.shutdown_failures = fail_cleanup ? 1 : 0;
  FakeAllocator allocator(lifecycle);
  FakeAccessOperator access_operator(lifecycle);
  FakeMetaManager meta_manager(lifecycle);
  const kvaf::RuntimeInitRequest request{4096, 128, kvaf::DeviceKind::kCpu};
  const std::string original_error = fail_operator
      ? "operator initialize failed" : "metadata registration failed";
  const kvaf::StatusCode original_code = fail_operator
      ? kvaf::StatusCode::kNotImplemented : kvaf::StatusCode::kNotFound;
  {
    kvaf::RuntimeCore runtime(meta_manager, access_operator, allocator);
    const kvaf::Status status = runtime.Initialize(request);
    Require(!status.ok(), "injected initialization error was lost");
    Require(lifecycle.allocations == 1, "DDR must be allocated exactly once");
    Require(lifecycle.operator_initializations == 1,
            "operator must be initialized exactly once");
    Require(lifecycle.registrations == (fail_operator ? 0 : 1),
            "metadata registration must follow successful operator init");
    Require(lifecycle.operator_shutdowns == 1,
            "failed initialization must attempt operator retirement");
    Require(!lifecycle.premature_release,
            "initialization cleanup released DDR while operator was live");

    if (fail_cleanup) {
      Require(status.message().find(original_error) != std::string::npos,
              "retained error must identify the initialization failure");
      Require(status.message().find("operator still owns registered DDR") !=
                  std::string::npos,
              "retained error must identify the failed retirement");
      Require(runtime.initialized(), "retained resources must remain owned");
      Require(lifecycle.releases == 0, "live DDR must not be released");
      const auto retained_region = runtime.remote_region();
      Require(retained_region.base != 0 && retained_region.length == 4096 &&
                  retained_region.slot_size == 128 &&
                  retained_region.device == kvaf::DeviceKind::kCpu,
              "retry must retain the original DDR region");
      Require(!runtime.Initialize(request).ok(),
              "reinitialization must reject pending cleanup");
      Require(lifecycle.allocations == 1,
              "pending cleanup must not allocate replacement DDR");
      Require(runtime.Shutdown().ok(), "shutdown retry must release resources");
      Require(lifecycle.operator_shutdowns == 2,
              "shutdown retry must retire the operator again");
    } else {
      Require(status.code() == original_code &&
                  status.message() == original_error,
              "successful cleanup must preserve the original error");
    }

    Require(!runtime.initialized(), "successful cleanup must clear ownership");
    Require(runtime.remote_region().base == 0,
            "successful cleanup must clear the remote region");
    Require(lifecycle.releases == 1, "DDR must be released exactly once");
    Require(!lifecycle.operator_live && !lifecycle.premature_release,
            "DDR release must follow successful operator retirement");
    Require(!runtime.Shutdown().ok(), "already cleaned runtime must be closed");
  }
  Require(lifecycle.releases == 1, "destructor must not double release DDR");
  std::vector<std::string> expected{"allocate", "operator_initialize"};
  if (!fail_operator) expected.push_back("register");
  expected.push_back("operator_shutdown");
  if (fail_cleanup) expected.push_back("operator_shutdown");
  expected.push_back("release");
  Require(lifecycle.events == expected, "resource cleanup order changed");
}

}  // namespace

int main() {
  try {
    TestInitializeCleanup(true, true);
    TestInitializeCleanup(false, true);
    TestInitializeCleanup(true, false);
    TestInitializeCleanup(false, false);
    std::cout << "RuntimeCore initialization cleanup tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
