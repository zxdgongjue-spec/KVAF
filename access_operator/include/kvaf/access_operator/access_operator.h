#pragma once

#include "kvaf/requests.h"
#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf {

// KV Access Operator 的初版占位接口。
// AIV+SIMT、AICPU 以及后续执行路径属于 Operator 内部实现细节。
class AccessOperator {
 public:
  virtual ~AccessOperator() = default;

  virtual Status ExecuteBatchGet(const BatchGetRequest& request,
                                 Event* event) = 0;

  virtual Status ExecuteBatchPut(const BatchPutRequest& request,
                                 Event* event) = 0;
};

}  // namespace kvaf
