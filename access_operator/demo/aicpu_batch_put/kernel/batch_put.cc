#include "aicpu_batch_put/include/batch_put_abi.h"
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuBatchPut(
    kvaf::aicpu_urma::BatchPutKernelArgs* args) {
  if (!args || args->op != kvaf::aicpu_urma::UmdkControlOp::kWrite) return 1U;
  return KvafAicpuUmdkControl(args);
}
