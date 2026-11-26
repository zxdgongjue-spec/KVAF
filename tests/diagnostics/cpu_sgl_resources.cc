#include "npu_runtime_config.h"
#include <cassert>
#include <cstdio>
int main() {
  using namespace kvaf;
  for (const auto executor : {Executor::Aiv, Executor::Aicpu}) {
    assert(!detail::ResolveExecutionResources(Transport::Urma, executor, true, false).sgl_push_chunked);
    ExecutionResources configured;
    configured.sgl_push_chunked = true;
    configured.sgl_push_cooperative = true;
    configured.sgl_push_head_rows = 256;
    configured.cores = 4;
    configured.sge_per_wr = 4;
    auto resolved = detail::ResolveExecutionResources(Transport::Urma, executor, true, false, &configured);
    assert(resolved.sgl_push_chunked && resolved.cores == 4 && resolved.sge_per_wr == 4);
    assert(resolved.sgl_push_head_rows == 256 && resolved.sgl_push_cooperative);
    configured.sgl_push_chunked = false;
    assert(!detail::ResolveExecutionResources(Transport::Urma, executor, true, false, &configured).sgl_push_chunked);
  }
  std::puts("CPU SGL resource resolution checks passed");
}
