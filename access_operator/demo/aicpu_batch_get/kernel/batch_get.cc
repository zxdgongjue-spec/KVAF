#include "gather_read.h"
// The common BatchGet entry currently selects gather_read exclusively.
extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuGatherRead(
    kvaf::aicpu_urma::UmdkBatchGetArgs* args) {
  return kvaf::aicpu_urma::ExecuteGatherRead(args);
}
