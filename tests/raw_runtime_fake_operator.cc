// Raw-only initialization must never construct the DDR AIV operator. Keep
// these symbols available for NpuRuntime's compiled but unused DDR branches.
#include "access_operator.h"
#include <stdexcept>

namespace kvaf {
struct DefaultAccessOperator::Impl {};
DefaultAccessOperator::DefaultAccessOperator(AccessOperatorConfig) {
  throw std::runtime_error("raw-only Runtime constructed the DDR AIV operator");
}
DefaultAccessOperator::~DefaultAccessOperator() = default;
Status DefaultAccessOperator::Initialize(const RemoteRegionDesc&) { return Status::NotImplemented("test DDR AIV"); }
Status DefaultAccessOperator::Shutdown() { return Status::OK(); }
Status DefaultAccessOperator::ExecuteBatchGet(const BatchGetExecRequest&) { return Status::NotImplemented("test DDR AIV"); }
Status DefaultAccessOperator::ExecuteBatchPut(const BatchPutExecRequest&) { return Status::NotImplemented("test DDR AIV"); }
Status DefaultAccessOperator::SetBatchGetKernelProfiling(bool) { return Status::NotImplemented("test DDR AIV"); }
Status DefaultAccessOperator::CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>*) const {
  return Status::NotImplemented("test DDR AIV");
}
}
