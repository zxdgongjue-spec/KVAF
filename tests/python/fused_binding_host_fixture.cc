// Compile the real Python binding against real public NpuRuntime declarations,
// but fake device/runtime definitions. No NPU work or DMA is performed here.
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

#include "acl/acl.h"
#include "npu_python_runtime.h"
#include "npu_runtime.h"
#include "meta_manager.h"

namespace py = pybind11;
namespace {
struct Driver {
  int acquires{0};
  int release_calls{0};
  int fences{0};
  int deregistrations{0};
  int shutdowns{0};
  int destructions{0};
  int selected_device{-1};
  std::uint64_t last_stream{0};
  bool fail_acquire{false};
  bool fail_release{false};
  bool fail_set_device{false};
  int host_puts{0};
  std::vector<std::uint8_t> host_seed_bytes;
} driver;
std::uint64_t next_lease = 1;

class CpuSeedAllocator : public kvaf::DdrMemoryAllocator {
 public:
  kvaf::Status Allocate(const kvaf::RuntimeInitRequest& request, kvaf::RemoteRegionDesc* region) override {
    bytes.resize(request.ddr_bytes);
    *region = {reinterpret_cast<std::uint64_t>(bytes.data()), request.ddr_bytes, request.slot_size, kvaf::DeviceKind::kCpu};
    return kvaf::Status::OK();
  }
  kvaf::Status Release(const kvaf::RemoteRegionDesc&) override { bytes.clear(); return kvaf::Status::OK(); }
  std::vector<std::uint8_t> bytes;
};

class CpuSeedOperator : public kvaf::AccessOperator {
 public:
  kvaf::Status Initialize(const kvaf::RemoteRegionDesc&) override { return kvaf::Status::OK(); }
  kvaf::Status Shutdown() override { return kvaf::Status::OK(); }
  kvaf::Status ExecuteBatchGet(const kvaf::BatchGetExecRequest&) override { return kvaf::Status::NotImplemented("fixture"); }
  kvaf::Status ExecuteBatchPut(const kvaf::BatchPutExecRequest& exec) override {
    if (exec.request->input.device != kvaf::DeviceKind::kCpu || exec.execution.stream != nullptr) {
      return kvaf::Status::InvalidArgument("fixture expected streamless CPU seed");
    }
    ++driver.host_puts;
    driver.host_seed_bytes.clear();
    const auto* source = static_cast<const std::uint8_t*>(exec.request->input.data);
    for (std::size_t row = 0; row < exec.request->keys.size(); ++row) {
      const auto* begin = source + exec.request->input_offsets[row];
      driver.host_seed_bytes.insert(driver.host_seed_bytes.end(), begin, begin + exec.request->lengths[row]);
      std::memcpy(reinterpret_cast<void*>(exec.put_desc->entries[row].address), begin, exec.request->lengths[row]);
    }
    return kvaf::Status::OK();
  }
};
}

namespace kvaf {
struct NpuRuntime::Impl {
  explicit Impl(NpuRuntimeOptions value) : options(std::move(value)), metadata(CreateInMemoryMetaManager()),
      runtime_core(*metadata, cpu_operator, allocator) {}
  NpuRuntimeOptions options;
  bool initialized{false};
  bool quarantined{false};
  FusedBatchGetContext context;
  std::unique_ptr<MetaManager> metadata;
  CpuSeedOperator cpu_operator;
  CpuSeedAllocator allocator;
  RuntimeCore runtime_core;
};

NpuRuntime::NpuRuntime(NpuRuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
NpuRuntime::~NpuRuntime() { ++driver.destructions; }
Status NpuRuntime::LinkTo(const LinkTarget&) { return Status::OK(); }
Status NpuRuntime::AcceptLink(const LinkTarget&) { return Status::NotImplemented("fixture"); }
Status NpuRuntime::ReleaseLink(LinkId) { return Status::NotImplemented("fixture"); }
Status NpuRuntime::ServeConnection(int) { return Status::NotImplemented("fixture"); }
Status NpuRuntime::Initialize(std::uint64_t bytes, std::uint64_t slot) {
  const auto result = impl_->runtime_core.Initialize({bytes, slot, DeviceKind::kCpu});
  if (!result.ok()) return result;
  impl_->initialized = true;
  return Status::OK();
}
Status NpuRuntime::Shutdown() {
  ++driver.shutdowns;
  if (impl_->context.lease_id) return Status::Busy("fixture active lease");
  if (impl_->initialized) {
    const auto result = impl_->runtime_core.Shutdown();
    if (!result.ok()) return result;
  }
  impl_->initialized = false;
  return Status::OK();
}
bool NpuRuntime::initialized() const { return impl_->initialized; }
RuntimeCapabilities NpuRuntime::capabilities() const { return {}; }
std::uint32_t NpuRuntime::device_id() const { return impl_->options.device_id; }
std::uint32_t NpuRuntime::npu_core_count() const { return impl_->options.npu_core_count; }
std::uint32_t NpuRuntime::batch_put_npu_core_count() const {
  return impl_->options.batch_put_npu_core_count == 0U
      ? npu_core_count() : impl_->options.batch_put_npu_core_count;
}
std::uint32_t NpuRuntime::jettys_per_core() const { return impl_->options.jettys_per_core; }
bool NpuRuntime::local_host_ddr() const { return impl_->options.local_host_ddr; }
std::string NpuRuntime::local_ddr_page_mode() const { return impl_->options.local_ddr_page_mode; }
LocalHostDdrStats NpuRuntime::local_ddr_page_stats() const { return {}; }
std::uint32_t NpuRuntime::batch_get_token_bytes() const {
  return impl_->options.batch_get_token_bytes;
}
std::uint32_t NpuRuntime::batch_get_tokens_per_block() const {
  return impl_->options.batch_get_tokens_per_block;
}
Status NpuRuntime::SetBatchGetKernelProfiling(bool) {
  return impl_->context.lease_id ? Status::Busy("fixture active lease") : Status::OK();
}
Status NpuRuntime::CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>*) const {
  return Status::OK();
}
Status NpuRuntime::CopyBatchGetCqDiagnostics(std::vector<BatchGetCqDiagnostic>* diagnostics) const {
  if (!impl_->initialized) return Status::NotInitialized("fixture uninitialized runtime");
  BatchGetCqDiagnostic item;
  item.core = 29;
  item.jetty = 1;
  item.trace_valid = true;
  item.raw_cq_word = 4;
  item.expected_owner = 1;
  item.slow_elapsed_ticks = 1234;
  item.polls = 20003;
  item.flags = 35;
  item.fast_budget_exceeded = item.late_success = item.bypass_probe = true;
  item.reason = "late_success";
  diagnostics->assign(1, item);
  return Status::OK();
}
Status NpuRuntime::AcquireFusedBatchGetContext(void* output, std::uint64_t bytes,
                                             FusedBatchGetContext* context) {
  ++driver.acquires;
  if (driver.fail_acquire) return Status::Internal("fixture registration failure");
  if (impl_->context.lease_id) return Status::Busy("fixture active lease");
  impl_->context = {next_lease++, UINT64_C(0xf000000000000001), 0x2000, 0x3000,
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(output)), bytes,
      npu_core_count(), jettys_per_core(), batch_get_token_bytes(),
      batch_get_tokens_per_block()};
  *context = impl_->context;
  return Status::OK();
}
Status NpuRuntime::ReleaseFusedBatchGetContext(const FusedBatchGetContext& context,
                                             void* stream) {
  ++driver.release_calls;
  if (!context.lease_id || context.lease_id != impl_->context.lease_id) {
    return Status::InvalidArgument("fixture wrong lease");
  }
  if (impl_->quarantined) return Status::Internal("fixture quarantined completion");
  ++driver.fences;
  driver.last_stream = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(stream));
  py::object hook = py::module_::import("_kvaf_fused_binding_host").attr("_before_release");
  if (!hook.is_none()) hook();
  if (driver.fail_release) {
    impl_->quarantined = true;
    return Status::Internal("fixture stream fence or queue completion failure");
  }
  ++driver.deregistrations;
  impl_->context = {};
  return Status::OK();
}

// Unrelated entry points must link for the actual binding TU, but are never
// exercised by the lease tests. Fail loudly if the fixture scope expands.
RuntimeCore& NpuRuntime::core() { return impl_->runtime_core; }
const RuntimeCore& NpuRuntime::core() const { return impl_->runtime_core; }
}

extern "C" {
aclError aclrtSetDevice(int32_t device) {
  driver.selected_device = device;
  return driver.fail_set_device ? 1 : ACL_SUCCESS;
}
aclError aclrtMemset(void*, std::size_t, int32_t, std::size_t) { return 1; }
aclError aclrtMemcpy(void*, std::size_t, const void*, std::size_t, aclrtMemcpyKind) { return 1; }
}

PYBIND11_MODULE(_kvaf_fused_binding_host, module) {
  kvaf::BindNpuPythonRuntime(module);
  module.attr("_before_release") = py::none();
  module.def("reset", [] { driver = {}; });
  module.def("fail_acquire", [](bool value) { driver.fail_acquire = value; });
  module.def("fail_release", [](bool value) { driver.fail_release = value; });
  module.def("fail_set_device", [](bool value) { driver.fail_set_device = value; });
  module.def("stats", [] {
    py::dict result;
    result["acquires"] = driver.acquires;
    result["release_calls"] = driver.release_calls;
    result["fences"] = driver.fences;
    result["deregistrations"] = driver.deregistrations;
    result["shutdowns"] = driver.shutdowns;
    result["destructions"] = driver.destructions;
    result["selected_device"] = driver.selected_device;
    result["last_stream"] = driver.last_stream;
    result["host_puts"] = driver.host_puts;
    return result;
  });
  module.def("host_seed_bytes", [] {
    return py::bytes(reinterpret_cast<const char*>(driver.host_seed_bytes.data()), driver.host_seed_bytes.size());
  });
}
