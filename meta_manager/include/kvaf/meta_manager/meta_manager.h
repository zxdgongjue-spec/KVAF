#pragma once

#include "kvaf/requests.h"
#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf {

// Meta Manager 的初版占位接口。
// 具体的 Key 表示、地址划分、锁定语义和 AccessTable 内容后续细化。
class MetaManager {
 public:
  virtual ~MetaManager() = default;

  virtual Status Initialize(const MemoryRegionDesc& region) = 0;

  // TODO: 补充 Put/Lookup/Lock/Delete 等正式接口签名。
  virtual Status Lookup(const LookupRequest& request,
                        AccessTableHandle* access_table) = 0;
};

}  // namespace kvaf
