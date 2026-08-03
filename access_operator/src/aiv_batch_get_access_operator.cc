#include "kvaf/access_operator/aiv_batch_get_access_operator.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "acl/acl.h"
#include "batch_get_abi.h"
#include "kvaf_aiv_batch_get.h"

namespace kvaf {

namespace {

Status InvalidArgument(std::string message) {
  return Status::Error(StatusCode::kInvalidArgument, std::move(message));
}

Status InvalidState(std::string message) {
  return Status::Error(StatusCode::kInvalidState, std::move(message));
}

Status Internal(std::string message) {
  return Status::Error(StatusCode::kInternal, std::move(message));
}

std::uint64_t PointerValue(const void* pointer) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

bool FitsU32(std::uint64_t value) {
  return value <= std::numeric_limits<std::uint32_t>::max();
}

}  // namespace

struct AivBatchGetAccessOperator::Impl {
  explicit Impl(AivBatchGetOperatorConfig init_config)
      : config(std::move(init_config)) {}

  AivBatchGetOperatorConfig config;
  RemoteRegionDesc remote_region;
  AivBatchGetOutputBinding binding;
  KvafBatchGetDdrRegistration* registration{nullptr};
  std::uint64_t channel_entity{0};
  bool initialized{false};
};

AivBatchGetAccessOperator::AivBatchGetAccessOperator(
    AivBatchGetOperatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

AivBatchGetAccessOperator::~AivBatchGetAccessOperator() { (void)Shutdown(); }

Status AivBatchGetAccessOperator::Initialize(const RemoteRegionDesc& remote_region) {
  if (impl_->initialized) {
    return InvalidState("AIV BatchGet operator has already been initialized");
  }
  if (remote_region.base == 0 || remote_region.length == 0 ||
      impl_->config.kernel_object_path.empty() || impl_->config.stream == nullptr ||
      impl_->config.aiv_core_count == 0 ||
      impl_->config.aiv_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES) {
    return InvalidArgument("invalid AIV BatchGet operator initialization config");
  }
  if (KvafExecuteBatchGetLoadKernel(impl_->config.kernel_object_path.c_str()) !=
      KVAF_BATCH_GET_SUCCESS) {
    return Internal("failed to load AIV ExecuteBatchGet kernel");
  }
  impl_->remote_region = remote_region;
  impl_->initialized = true;
  return Status::OK();
}

Status AivBatchGetAccessOperator::BindBatchGetOutput(
    const AivBatchGetOutputBinding& binding) {
  if (!impl_->initialized) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AIV BatchGet operator has not been initialized");
  }
  if (impl_->registration != nullptr) {
    return InvalidState("AIV BatchGet output is already bound");
  }
  if (binding.output.data == nullptr || binding.output.bytes == 0 ||
      binding.core_status.data == nullptr || binding.queue_state.data == nullptr ||
      binding.output.device != DeviceKind::kNpu ||
      binding.core_status.device != DeviceKind::kNpu ||
      binding.queue_state.device != DeviceKind::kNpu ||
      binding.core_status.bytes <
          static_cast<std::size_t>(impl_->config.aiv_core_count) *
              sizeof(KvafBatchGetCoreStatus) ||
      binding.queue_state.bytes <
          static_cast<std::size_t>(impl_->config.aiv_core_count) *
              sizeof(KvafBatchGetQueueState)) {
    return InvalidArgument("invalid caller-owned AIV BatchGet output binding");
  }

  KvafBatchGetDdrRegistration* registration = nullptr;
  std::uint64_t channel_entity = 0;
  const int32_t result = KvafBatchGetRegisterLocalDdr(
      reinterpret_cast<void*>(impl_->remote_region.base),
      impl_->remote_region.length, binding.output.data, binding.output.bytes,
      impl_->config.aiv_core_count, 1U, &registration, &channel_entity);
  if (result != KVAF_BATCH_GET_SUCCESS) {
    return Internal("failed to register Host DDR and caller output with Direct-HCCP");
  }

  impl_->binding = binding;
  impl_->registration = registration;
  impl_->channel_entity = channel_entity;
  return Status::OK();
}

Status AivBatchGetAccessOperator::UnbindBatchGetOutput() {
  if (impl_->registration != nullptr) {
    KvafBatchGetUnregisterLocalDdr(impl_->registration);
  }
  impl_->registration = nullptr;
  impl_->channel_entity = 0;
  impl_->binding = AivBatchGetOutputBinding{};
  return Status::OK();
}

Status AivBatchGetAccessOperator::Shutdown() {
  if (!impl_->initialized) {
    return Status::OK();
  }
  (void)UnbindBatchGetOutput();
  impl_->remote_region = RemoteRegionDesc{};
  impl_->initialized = false;
  return Status::OK();
}

Status AivBatchGetAccessOperator::ExecuteBatchGet(
    const BatchGetExecRequest& exec) {
  if (!impl_->initialized) {
    return Status::Error(StatusCode::kNotInitialized,
                         "AIV BatchGet operator has not been initialized");
  }
  if (exec.request == nullptr || impl_->registration == nullptr) {
    return InvalidState("AIV BatchGet requires a bound caller output");
  }
  const BatchGetRequest& request = *exec.request;
  if (request.indices.device != DeviceKind::kNpu ||
      request.output_indices.device != DeviceKind::kNpu ||
      request.mask.data == nullptr || request.mask.device != DeviceKind::kNpu ||
      request.output.device != DeviceKind::kNpu ||
      request.device_access_table.addresses.device != DeviceKind::kNpu ||
      request.device_access_table.valid.device != DeviceKind::kNpu ||
      request.output.data != impl_->binding.output.data ||
      request.output.bytes != impl_->binding.output.bytes ||
      request.output_block_bytes == 0 ||
      exec.host_access_table.size == 0 ||
      request.indices.bytes % sizeof(std::int32_t) != 0 ||
      request.output_indices.bytes != request.indices.bytes ||
      request.mask.bytes != request.indices.bytes / sizeof(std::int32_t)) {
    return InvalidArgument("AIV BatchGet request does not match the fixed-buffer contract");
  }

  const std::size_t request_count = request.indices.bytes / sizeof(std::int32_t);
  const std::size_t output_rows = request.output.bytes / request.output_block_bytes;
  if (!FitsU32(exec.host_access_table.size) || !FitsU32(request_count) ||
      !FitsU32(output_rows) || !FitsU32(request.output_block_bytes)) {
    return InvalidArgument("AIV BatchGet dimensions exceed the device ABI");
  }
  for (std::size_t index = 0; index < exec.host_access_table.size; ++index) {
    if (exec.host_access_table.valid[index] != 0 &&
        exec.host_access_table.lengths[index] != request.output_block_bytes) {
      return InvalidArgument("AIV BatchGet supports fixed-size valid blocks only");
    }
  }

  KvafExecuteBatchGetKernelArgs arguments{};
  arguments.address_table = PointerValue(request.device_access_table.addresses.data);
  arguments.valid_table = PointerValue(request.device_access_table.valid.data);
  arguments.indices = PointerValue(request.indices.data);
  arguments.mask = PointerValue(request.mask.data);
  arguments.output_indices = PointerValue(request.output_indices.data);
  arguments.output = PointerValue(request.output.data);
  arguments.core_status = PointerValue(impl_->binding.core_status.data);
  arguments.channel_entity = impl_->channel_entity;
  arguments.queue_state = PointerValue(impl_->binding.queue_state.data);
  arguments.table_capacity = static_cast<std::uint32_t>(exec.host_access_table.size);
  arguments.request_capacity = static_cast<std::uint32_t>(request_count);
  arguments.output_rows = static_cast<std::uint32_t>(output_rows);
  arguments.block_bytes = static_cast<std::uint32_t>(request.output_block_bytes);
  arguments.block_count = impl_->config.aiv_core_count;

  if (KvafExecuteBatchGetLaunch(
          &arguments, static_cast<aclrtStream>(impl_->config.stream)) !=
      KVAF_BATCH_GET_SUCCESS) {
    return Internal("failed to launch AIV ExecuteBatchGet");
  }
  if (aclrtSynchronizeStream(static_cast<aclrtStream>(impl_->config.stream)) !=
      ACL_SUCCESS) {
    return Internal("AIV ExecuteBatchGet stream synchronization failed");
  }
  return Status::OK();
}

Status AivBatchGetAccessOperator::ExecuteBatchPut(
    const BatchPutExecRequest&) {
  return Status::Error(StatusCode::kNotImplemented,
                       "AIV BatchGet operator does not implement BatchPut");
}

}  // namespace kvaf
