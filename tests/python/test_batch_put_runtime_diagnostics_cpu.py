"""Run real Put host error paths with an ACL fault and no subsequent device calls."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "access_operator/demo/aiv_batch_put"


@pytest.fixture(scope="module")
def runtime_diagnostic_binary(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = tmp_path_factory.mktemp("put-runtime-diagnostic")
    host = (DEMO / "host/batch_put_host.cpp").read_text()
    capture = host[
        host.index("static void LogBatchPutAclFailure(") : host.index(
            "struct ProcessTokenQueryInfo"
        )
    ]
    synchronize = host[host.index('extern "C" int32_t KvafExecuteBatchPutSync(') :]
    operator = (ROOT / "access_operator/src/access_operator.cc").read_text()
    begin = operator.index("void LogBatchPutDeviceStatus(")
    formatter = operator[begin : operator.index("\n#endif", begin)]
    (root / "acl").mkdir()
    (root / "acl/acl.h").write_text("using aclError=int; using aclrtStream=void*;\n")
    source = r"""
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include "kvaf_aiv_batch_put.h"
#include "batch_put_internal.h"
constexpr aclError ACL_SUCCESS=0;
enum { ACL_MEMCPY_DEVICE_TO_HOST };
struct Jetty { void *handle=nullptr; };
struct Worker { Jetty device_user_ctl_jetty, jetty; };
struct Host { std::vector<Worker> workers; };
struct Api { void *library=nullptr; };
struct KvafBatchPutTransport {
  uint32_t aiv_core_count=24, jettys_per_core=2;
  std::mutex launch_mutex;
  Api api;
  std::vector<Worker> workers;
  Host *local_host=nullptr;
};
struct Memory { uint64_t registered_address=0, registered_bytes=0; uint32_t token_id=0; };
struct KvafBatchPutLocalBuffer {
  uint32_t reference_count=1;
  KvafBatchPutTransport *transport=nullptr;
  uint64_t input_hbm_address=0, input_hbm_bytes=0, device_channel_entity=1;
  Memory memory;
};
// Device queue table is an address passed to a launch; never read after failure.
// Extend the tiny transport with that scalar without involving device memory.
"""
    source = source.replace(
        "Api api;", "Api api; void *device_queue_state=reinterpret_cast<void*>(1);"
    )
    source += r"""
namespace hccp {
void VerboseLog(const char*,...) {}
void LogBatchPutJettyState(void*,void*,const char*,uint32_t,uint32_t,bool=false) {
  assert(false && "no queue query is permitted after synchronization fails");
}
}
int sync_calls=0, copy_calls=0, recent_calls=0, launch_calls=0;
int mode=0;
bool failed_acl=false;
const char *last_acl="none";
std::string recent_text;
aclError aclrtMemset(void *destination,size_t,int32_t value,size_t bytes) {
  assert(!failed_acl); last_acl="clear_status";
  if(mode==4) { failed_acl=true; return 507033; }
  std::memset(destination,value,bytes); return 0;
}
aclError aclrtSynchronizeStream(aclrtStream) {
  assert(!failed_acl); last_acl="synchronize"; ++sync_calls;
  if(mode<=3) { failed_acl=true; return mode==1 ? 507035 : 507034; }
  return 0;
}
aclError aclrtMemcpy(void *destination,size_t,const void *source,size_t bytes,int) {
  assert(!failed_acl); last_acl="copy_status"; ++copy_calls;
  if(mode==5) { failed_acl=true; return 507011; }
  std::memcpy(destination,source,bytes); return 0;
}
const char *aclGetRecentErrMsg() {
  ++recent_calls;
  // The message is read immediately after the failing ACL operation.
  assert(failed_acl);
  if(mode<=3) assert(std::strcmp(last_acl,"synchronize")==0);
  if(mode==4) assert(std::strcmp(last_acl,"clear_status")==0);
  if(mode==5) assert(std::strcmp(last_acl,"copy_status")==0);
  return mode==3 ? nullptr : recent_text.c_str();
}
extern "C" int32_t KvafExecuteBatchPutLaunchInternal(const KvafExecuteBatchPutLaunchArgs *a,aclrtStream) {
  assert(!failed_acl); ++launch_calls;
  if(mode==6) {
    auto *status=reinterpret_cast<KvafBatchPutCoreStatus*>(a->request.core_status);
    for(uint32_t core=0;core<a->request.aiv_core_count;++core) {
      status[core].magic=KVAF_AIV_BATCH_PUT_STATUS_MAGIC;
      status[core].processed_request_count=uint64_t(a->request.row_count)*(core+1)/a->request.aiv_core_count-
        uint64_t(a->request.row_count)*core/a->request.aiv_core_count;
      status[core].failed_request_index=UINT64_MAX;
    }
  }
  return 0;
}
static void LogBatchPutCompletionDiagnostics(const KvafBatchPutCoreStatus*,uint32_t,uint32_t) {}
struct BatchPutExecutionState { std::vector<KvafBatchPutCoreStatus> host_core_status; };
"""
    source += (
        capture
        + synchronize
        + formatter
        + r"""
int main(int argc,char** argv) {
  assert(argc==2); mode=std::atoi(argv[1]);
  recent_text= mode==2 ? std::string(5000,'x') : "Vector core failure\noriginal SDK context";
  KvafBatchPutTransport transport;
  KvafBatchPutLocalBuffer local;
  local.transport=&transport; local.input_hbm_address=1234; local.input_hbm_bytes=65536;
  std::vector<KvafBatchPutCoreStatus> device(24);
  BatchPutExecutionState execution;
  execution.host_core_status.resize(24);
  for(auto& core:execution.host_core_status) core.magic=0xfeedU;
  KvafExecuteBatchPutKernelArgs args{1234,1,1,1,reinterpret_cast<uint64_t>(device.data()),65536,128,24};
  const int result=KvafExecuteBatchPutSync(&local,&args,reinterpret_cast<void*>(1),
      execution.host_core_status.data(),24);
  if(mode<=5) {
    assert(result==KVAF_BATCH_PUT_RUNTIME_ERROR && recent_calls==1);
    assert(copy_calls==(mode==5 ? 1 : 0));
    for(const auto& core:execution.host_core_status) assert(core.magic==0xfeedU);
    LogBatchPutDeviceStatus(execution,24,result);
  } else if(mode==6) {
    assert(result==0 && copy_calls==1 && recent_calls==0);
  } else {
    assert(result==KVAF_BATCH_PUT_KERNEL_ERROR && copy_calls==1 && recent_calls==0);
    LogBatchPutDeviceStatus(execution,24,result);
  }
  std::puts("PUT_RUNTIME_DIAGNOSTIC_PASS");
}
"""
    )
    cpp = root / "diagnostic.cpp"
    cpp.write_text(source)
    binary = root / "diagnostic"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fsanitize=address,undefined",
            "-I",
            str(root),
            "-I",
            str(DEMO / "include"),
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize("mode", range(8))
def test_actual_sync_failure_never_claims_a_host_buffer_is_device_status(
    runtime_diagnostic_binary, mode
):
    process = subprocess.run(
        [str(runtime_diagnostic_binary), str(mode)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.stdout == "PUT_RUNTIME_DIAGNOSTIC_PASS\n"
    if mode <= 5:
        assert "status_snapshot=unavailable device_status_read=0" in process.stderr
        assert "BATCH_PUT_DEVICE_STATUS_UNAVAILABLE" in process.stderr
        assert "BATCH_PUT_DEVICE_ERROR core=" not in process.stderr
        if mode <= 3:
            assert "phase=synchronize" in process.stderr
            assert (
                "ACL_ERROR_RT_VECTOR_CORE_EXCEPTION"
                if mode == 1
                else "ACL_ERROR_RT_VECTOR_CORE_TIMEOUT"
            ) in process.stderr
        if mode == 2:
            assert "recent_error_bytes=4096 recent_error_truncated=1" in process.stderr
            assert len(process.stderr) < 4800
        elif mode == 3:
            assert "recent_error=(unavailable)" in process.stderr
        else:
            assert (
                "recent_error=Vector core failure original SDK context"
                in process.stderr
            )
    elif mode == 6:
        assert not process.stderr
    else:
        assert process.stderr.count("BATCH_PUT_DEVICE_ERROR core=") == 24
        assert "DEVICE_STATUS_UNAVAILABLE" not in process.stderr
