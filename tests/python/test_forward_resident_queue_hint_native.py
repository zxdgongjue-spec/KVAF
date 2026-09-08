"""Execute the real queue hint/CAS bodies, including stale-hint races.

Host atomics prove ownership logic, not NPU cache visibility or HCCP timing.
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_forward_resident_bulk_descriptor_native import KERNEL, actual_function


class QueueHintNativeTest(unittest.TestCase):
    def test_actual_hint_never_grants_ownership_and_cas_handles_contention(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("host C++ compiler unavailable")
        source = (KERNEL / "kvaf_forward_resident_get_device.h").read_text()
        # Only erase hardware annotations, not function bodies or predicates.
        normalized = source.replace("__simt_callee__", "__aicore__")
        definitions = "\n".join(
            re.search(pattern, source, re.S).group(0)
            for pattern in (
                r"enum KvafForwardResidentQueueLeaseState.*?\n\};",
                r"struct KvafForwardResidentQueueLease \{.*?\n\};",
            )
        )
        functions = "\n".join(actual_function(normalized, name) for name in (
            "IsKvafForwardResidentBatchGetControlValidForAicore",
            "IsKvafForwardResidentBatchGetControlValidForLease",
            "FindKvafForwardResidentAvailableQueueForAicore",
            "TryAcquireKvafForwardResidentQueueFromSimt",
            "ReturnKvafForwardResidentUnpostedQueueFromSimt",
        ))
        stub = r"""
#include <atomic>
#include <array>
#include <cassert>
#include <cstdint>
#include <thread>
#include <vector>
#define __aicore__
#define __gm__
std::atomic<uint32_t> rmws{0};
template<class T> T asc_ldcg(T* p) {return __atomic_load_n(p,__ATOMIC_ACQUIRE);}
template<class T> T ReadGmByPassDCache(T* p) {return asc_ldcg(p);}
template<class T> T asc_atomic_cas(T* p,T old,T desired) {
 ++rmws;
 __atomic_compare_exchange_n(p,&old,desired,false,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE);
 return old;
}
namespace AscendC { namespace Simt {
void ThreadFence(){std::atomic_thread_fence(std::memory_order_seq_cst);}
}}
"""
        cases = r"""
int main() {
 std::array<int32_t,8> owners{};
 KvafForwardResidentBatchGetControl c{};
 c.magic=KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_MAGIC;
 c.version=KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_VERSION;
 c.channel_entity=c.queue_state=c.core_status=c.output=1;
 c.descriptor_scratch=c.descriptor_templates=1;
 c.queue_leases=reinterpret_cast<uint64_t>(owners.data());
 c.queue_count=c.descriptor_template_count=2;
 c.output_rows=c.resident_row_count=1000;c.jettys_per_core=1;c.block_bytes=656;
 uint32_t hint=999;
 owners[0]=11;owners[1]=12;
 KvafForwardResidentQueueLease lease{};
 assert(!FindKvafForwardResidentAvailableQueueForAicore(c,0,&hint));
 assert(hint==999 && rmws==0);
 assert(!TryAcquireKvafForwardResidentQueueFromSimt(c,3,0,&lease));
 assert(rmws==0); // busy queues are never atomic read-modify-written
 owners[0]=0;
 assert(FindKvafForwardResidentAvailableQueueForAicore(c,0,&hint) && hint==0);
 assert(owners[0]==0 && rmws==0); // hint grants NOTHING
 owners[0]=77; // another worker won between the hint and the CAS service
 assert(!TryAcquireKvafForwardResidentQueueFromSimt(c,3,hint,&lease));
 assert(owners[0]==77 && rmws==0);
 owners[1]=0;
 assert(TryAcquireKvafForwardResidentQueueFromSimt(c,3,hint,&lease));
 assert(lease.queue_id==1 && lease.owner_tag==4 && owners[1]==4);
 assert(ReturnKvafForwardResidentUnpostedQueueFromSimt(c,&lease));
 auto invalid=c;invalid.version=0;
 assert(!FindKvafForwardResidentAvailableQueueForAicore(invalid,0,&hint));
 assert(!FindKvafForwardResidentAvailableQueueForAicore(c,0,nullptr));
 for(uint32_t queues:{2U,4U,8U}) {
  c.queue_count=c.descriptor_template_count=queues;owners.fill(0);
  std::array<std::atomic<int>,8> active{};
  std::atomic<uint32_t> completed{0};std::vector<std::thread> workers;
  for(uint32_t worker=0;worker<16;++worker) workers.emplace_back([&,worker]{
   for(uint32_t operation=0;operation<100;++operation) {
    KvafForwardResidentQueueLease held{};
    uint32_t next=worker+operation;
    while(!FindKvafForwardResidentAvailableQueueForAicore(c,next,&next) ||
          !TryAcquireKvafForwardResidentQueueFromSimt(c,worker,next,&held)) {
     ++next;std::this_thread::yield();
    }
    assert(active[held.queue_id].fetch_add(1)==0);
    std::this_thread::yield();
    assert(active[held.queue_id].fetch_sub(1)==1);
    assert(ReturnKvafForwardResidentUnpostedQueueFromSimt(c,&held));
    ++completed;
   }
  });
  for(auto& worker:workers)worker.join();
  assert(completed==1600);
  for(auto owner:owners)assert(owner==0);
 }
}
"""
        with tempfile.TemporaryDirectory(prefix="kvaf_queue_hint_") as temp:
            cpp = Path(temp) / "actual.cpp"
            cpp.write_text(stub + f'\n#include "{KERNEL.parent / "include/batch_get_abi.h"}"\n'
                           + definitions + functions + cases)
            result = subprocess.run(
                [compiler, "-std=c++17", "-O2", "-pthread", str(cpp), "-o", temp + "/test"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([temp + "/test"], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
