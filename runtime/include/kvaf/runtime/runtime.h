#pragma once

#include "kvaf/access_operator/access_operator.h"
#include "kvaf/meta_manager/meta_manager.h"
#include "kvaf/requests.h"
#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf {

// KVAF Runtime 的初版占位接口。
// Runtime 负责对外入口以及 Meta Manager / Access Operator 的组合关系。
class Runtime {
 public:
  Runtime(MetaManager& meta_manager, AccessOperator& access_operator)
      : meta_manager_(meta_manager), access_operator_(access_operator) {}

  virtual ~Runtime() = default;

  // TODO: 正式接口与 BatchGet/BatchPut/Delete 生命周期对齐。
  virtual Status BatchGet(const BatchGetRequest& request, Event* event) = 0;
  virtual Status BatchPut(const BatchPutRequest& request, Event* event) = 0;
  virtual Status Delete(const DeleteRequest& request, Event* event) = 0;

 protected:
  MetaManager& meta_manager_;
  AccessOperator& access_operator_;
};

}  // namespace kvaf
