#include "runtime.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(condition) do { if (!(condition)) { \
  std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); std::exit(1); \
} } while (false)

namespace {

class CaptureStderr {
 public:
  CaptureStderr() : file_(std::tmpfile()), saved_(dup(STDERR_FILENO)) {
    CHECK(file_ != nullptr && saved_ >= 0);
    std::fflush(stderr);
    CHECK(dup2(fileno(file_), STDERR_FILENO) >= 0);
  }
  ~CaptureStderr() {
    std::fflush(stderr);
    (void)dup2(saved_, STDERR_FILENO);
    close(saved_);
    std::fclose(file_);
  }
  std::string Read() {
    std::fflush(stderr);
    std::rewind(file_);
    std::string result;
    char buffer[1024];
    while (const auto count = std::fread(buffer, 1, sizeof(buffer), file_)) {
      result.append(buffer, count);
    }
    return result;
  }
 private:
  std::FILE* file_;
  int saved_;
};

class Allocator final : public kvaf::DdrMemoryAllocator {
 public:
  kvaf::Status Allocate(const kvaf::RuntimeInitRequest& request,
                        kvaf::RemoteRegionDesc* region) override {
    ++allocations;
    *region = {4096U, request.ddr_bytes, request.slot_size, kvaf::DeviceKind::kCpu};
    return kvaf::Status::OK();
  }
  kvaf::Status Release(const kvaf::RemoteRegionDesc&) override {
    ++releases;
    return kvaf::Status::OK();
  }
  int allocations{0};
  int releases{0};
};

class Operator final : public kvaf::AccessOperator {
 public:
  explicit Operator(bool fail = false) : fail_(fail) {}
  kvaf::Status Initialize(const kvaf::RemoteRegionDesc&) override {
    return fail_ ? kvaf::Status::Internal("injected transport failure") : kvaf::Status::OK();
  }
  kvaf::Status Shutdown() override { return kvaf::Status::OK(); }
  kvaf::Status ExecuteBatchGet(const kvaf::BatchGetExecRequest&) override {
    return kvaf::Status::NotImplemented("unused");
  }
  kvaf::Status ExecuteBatchPut(const kvaf::BatchPutExecRequest&) override {
    return kvaf::Status::NotImplemented("unused");
  }
 private:
  bool fail_;
};

}  // namespace

int main() {
  const kvaf::RuntimeInitRequest request{8192U, 4096U, kvaf::DeviceKind::kCpu};
  for (bool profile : {false, true}) {
    auto metadata = kvaf::CreateInMemoryMetaManager();
    Allocator allocator;
    Operator op;
    kvaf::RuntimeCore runtime(*metadata, op, allocator);
    runtime.SetInitializeProfiling(profile, 42U, 7U);
    CaptureStderr capture;
    CHECK(runtime.Initialize(request).ok());
    CHECK(runtime.initialized());
    CHECK(runtime.Shutdown().ok());
    CHECK(allocator.allocations == 1 && allocator.releases == 1);
    const std::string log = capture.Read();
    if (profile) {
      CHECK(log.find("\"phase\":\"transport_initialize\"") != std::string::npos);
      CHECK(log.find("\"phase\":\"metadata_initialize\"") != std::string::npos);
      CHECK(log.find("\"attempt\":42") != std::string::npos);
      CHECK(log.find("\"candidate\":7") != std::string::npos);
      CHECK(log.find("\"outcome\":\"failed\"") == std::string::npos);
    } else {
      CHECK(log.empty());
    }
  }
  {
    auto metadata = kvaf::CreateInMemoryMetaManager();
    Allocator allocator;
    Operator op(true);
    kvaf::RuntimeCore runtime(*metadata, op, allocator);
    runtime.SetInitializeProfiling(true, 43U, 8U);
    CaptureStderr capture;
    CHECK(!runtime.Initialize(request).ok());
    CHECK(!runtime.initialized());
    CHECK(allocator.allocations == 1 && allocator.releases == 1);
    const std::string log = capture.Read();
    CHECK(log.find("\"outcome\":\"failed\"") != std::string::npos);
    CHECK(log.find("\"phase\":\"metadata_initialize\"") == std::string::npos);
  }
  std::puts("runtime initialization profiling tests passed");
  return 0;
}
