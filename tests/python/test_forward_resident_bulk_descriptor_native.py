"""Execute actual bulk descriptor/validation C++ with guarded host DMA stubs.

This proves descriptor values and bounded memory accesses, not CANN domains,
NPU memory ordering, HCCP throughput or real DMA completion.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "access_operator/demo/aiv_batch_get/kernel"


def actual_function(source, name):
    matches = list(
        re.finditer(r"(?:__simt_vf__ )?__aicore__ (?:LAUNCH_BOUND\(32\)\s*)?inline (?:bool|void) " + name + r"\(", source)
    )
    if len(matches) != 1:
        raise AssertionError("missing/duplicate actual function: " + name)
    begin = source.index("{", matches[0].end())
    depth = 0
    for end in range(begin, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[matches[0].start() : end + 1]
    raise AssertionError("unclosed actual function")


PREFIX = r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#define __aicore__
#define __simt_vf__
#define LAUNCH_BOUND(n)
#define __gm__
#define __ubuf__
struct ThreadIndex { uint32_t x; };
thread_local ThreadIndex threadIdx;
struct WarpBarrier {
 std::mutex lock; std::condition_variable cv; uint32_t arrived=0, generation=0;
 void wait() {
  std::unique_lock<std::mutex> guard(lock); const auto observed=generation;
  if(++arrived==32U) { arrived=0; ++generation; cv.notify_all(); }
  else cv.wait(guard,[&]{return generation!=observed;});
 }
} warp_barrier;
std::array<uint32_t,32> warp_values;
void asc_syncthreads() { warp_barrier.wait(); }
uint32_t asc_ballot(int predicate) {
 warp_values[threadIdx.x]=predicate!=0; asc_syncthreads();
 uint32_t bits=0; for(uint32_t i=0;i<32;++i) bits|=warp_values[i]<<i;
 asc_syncthreads(); return bits;
}
uint32_t asc_shfl_down(uint32_t value,uint32_t delta) {
 warp_values[threadIdx.x]=value; asc_syncthreads();
 const uint32_t result=warp_values[std::min(threadIdx.x+delta,31U)];
 asc_syncthreads(); return result;
}
template<class T> T asc_ldcg(T* p) { static_assert(!std::is_const<T>::value); return *p; }
constexpr int EVENT_ID6=6;
alignas(64) std::array<uint8_t,4096> ub;
uint64_t bypass_reads=0, bypass_writes=0, dma_bytes=0;
void* get_imm(uint32_t offset) { assert(offset<ub.size()); return ub.data()+offset; }
template<class T> T ReadGmByPassDCache(T* p) { ++bypass_reads; return *p; }
template<class T> void WriteGmByPassDCache(T* p,T v) { ++bypass_writes; *p=v; }
namespace AscendC {
enum class MemDsbT { DDR }; template<MemDsbT> void DataSyncBarrier() {}
enum class TPosition { VECIN }; struct TBuffAddr { uint8_t logicPos; };
template<class T> struct LocalTensor {
 T* pointer=nullptr; uint32_t capacity=0;
 void SetAddr(TBuffAddr) {}
 void InitBuffer(uint32_t offset,uint32_t n) {
  assert(offset+n*sizeof(T)<=ub.size()); pointer=reinterpret_cast<T*>(get_imm(offset)); capacity=n;
 }
 void SetValue(uint32_t i,T value) { assert(i<capacity); pointer[i]=value; }
};
template<class T> struct GlobalTensor { T* pointer; void SetGlobalBuffer(T* p) { pointer=p; } };
struct DataCopyExtParams { uint32_t blockCount,blockLen,srcStride,dstStride,rsv; };
template<class T> struct DataCopyPadExtParams { bool pad; uint32_t left,right; T value; };
enum class HardEvent { S_MTE2, MTE2_S, S_MTE3, MTE3_S, S_V, V_S };
uint32_t event_state=0;
struct PendingCopy { void* dst; const void* src; uint32_t bytes; };
std::vector<PendingCopy> pending;
template<HardEvent E> void SetFlag(int event) {
 assert(event==6);
 if(E==HardEvent::S_MTE2) { assert(event_state==0); event_state=1; }
 else if(E==HardEvent::MTE2_S) { assert(event_state==3); event_state=4; }
 else if(E==HardEvent::S_MTE3) { assert(event_state==0); event_state=5; }
 else if(E==HardEvent::S_V) { assert(event_state==0); event_state=8; }
 else if(E==HardEvent::V_S) { assert(event_state==9); event_state=10; }
 else { assert(event_state==6 && pending.size()==4); event_state=7; }
}
template<HardEvent E> void WaitFlag(int event) {
 assert(event==6);
 if(E==HardEvent::S_MTE2) { assert(event_state==1); event_state=2; }
 else if(E==HardEvent::MTE2_S) { assert(event_state==4); event_state=0; }
 else if(E==HardEvent::S_MTE3) { assert(event_state==5); event_state=6; }
 else if(E==HardEvent::S_V) { assert(event_state==8); event_state=9; }
 else if(E==HardEvent::V_S) { assert(event_state==10); event_state=0; }
 else {
  assert(event_state==7);
  for(const auto& p:pending) { std::memcpy(p.dst,p.src,p.bytes); dma_bytes+=p.bytes; }
  pending.clear(); event_state=0;
 }
}
namespace Simt {
struct Dim3 { uint32_t x,y,z; };
template<auto Fn,class... Args> void VF_CALL(Dim3 dims,Args... args) {
 assert(dims.x==32 && dims.y==1 && dims.z==1 && event_state==9);
 std::vector<std::thread> threads;
 for(uint32_t lane=0;lane<32;++lane) threads.emplace_back([=]{threadIdx.x=lane; Fn(args...);});
 for(auto& thread:threads) thread.join();
}
}
template<class T> void DataCopyPad(GlobalTensor<T>& global,LocalTensor<T>& local,
 const DataCopyExtParams& p) {
 assert(event_state==6 && p.blockCount==1 && !p.srcStride && !p.dstStride);
 assert((p.blockLen+31U)/32U*32U<=local.capacity*sizeof(T));
 // Delayed completion exposes a missing MTE3->S wait before UB reuse.
 pending.push_back({global.pointer,local.pointer,p.blockLen});
}
template<class T> void DataCopyPad(LocalTensor<T>& local,GlobalTensor<T>& global,
 const DataCopyExtParams& p,const DataCopyPadExtParams<T>& pad) {
 assert(event_state==2 && !pad.pad && p.blockCount==1 && !p.srcStride && !p.dstStride);
 const uint32_t padded_bytes=(p.blockLen+31U)/32U*32U;
 assert(padded_bytes<=local.capacity*sizeof(T));
 // Permit the full aligned UB footprint; never assume the hardware leaves
 // bytes in the trailing partial datablock untouched.
 std::memset(local.pointer,0xfd,padded_bytes);
 std::memcpy(local.pointer,global.pointer,p.blockLen); dma_bytes+=p.blockLen; event_state=3;
}
}
namespace kvaf_fused_get {
constexpr uint32_t kDenseBatchGetReservedWqeUbBytes=4096;
constexpr uint32_t kDenseBatchGetPostOnlyScratchCeiling=135168;
void CleanCacheLines(uint8_t*,uint64_t) {}
}
"""

CASES = r"""
struct Fixture {
 KvafForwardResidentBatchGetControl control{};
 KvafForwardResidentBatchGetDescriptorScratch descriptor{};
 std::array<int32_t,258> indices{},offsets{},outputs{};
 std::array<uint8_t,258> masks{};
 std::array<int32_t,1024> seeds{};
 Fixture() {
  indices.fill(-99); offsets.fill(-99); outputs.fill(-99); masks.fill(99);
  control.magic=KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_MAGIC;
  control.version=KVAF_FORWARD_RESIDENT_BATCH_GET_CONTROL_VERSION;
  control.channel_entity=control.queue_state=control.core_status=control.output=1;
  control.queue_leases=control.descriptor_scratch=control.descriptor_templates=1;
  control.queue_count=control.descriptor_template_count=1;
  control.output_rows=20000; control.resident_row_base=1000; control.resident_row_count=5000;
  control.jettys_per_core=1; control.block_bytes=656;
  descriptor.address_table=descriptor.valid_table=1;
  descriptor.indices=reinterpret_cast<uint64_t>(indices.data()+1);
  descriptor.offsets=reinterpret_cast<uint64_t>(offsets.data()+1);
  descriptor.mask=reinterpret_cast<uint64_t>(masks.data()+1);
  descriptor.output_indices=reinterpret_cast<uint64_t>(outputs.data()+1);
  descriptor.table_capacity=2048; descriptor.request_capacity=256;
  for(uint32_t i=0;i<256;++i) { seeds[i*4+2]=static_cast<int32_t>(i*127+13); seeds[i*4+3]=1000+i*7; }
 }
 bool build(uint32_t n) { return BuildKvafForwardResidentRequestsFromSeedWords(
  control,&descriptor,seeds.data(),n,4,2,3,128); }
 void guards() const { assert(indices.front()==-99 && indices.back()==-99 &&
   offsets.front()==-99 && offsets.back()==-99 && outputs.front()==-99 &&
   outputs.back()==-99 && masks.front()==99 && masks.back()==99); }
};
void equivalence() {
 for(uint32_t n:{1U,2U,14U,63U,64U,65U,255U,256U}) {
  Fixture bulk,legacy; bypass_reads=bypass_writes=dma_bytes=0; ub.fill(0xa7);
  assert(bulk.build(n)); assert(bypass_reads==0 && bypass_writes==0 && dma_bytes==13*n);
  assert(AscendC::pending.empty() && AscendC::event_state==0);
  const auto before_validation=ub;
  assert(ValidateKvafForwardResidentOutputIndices(bulk.control,bulk.descriptor));
  assert(bypass_reads==0 && dma_bytes==17*n && AscendC::event_state==0);
  uint32_t hash_capacity=2;
  while(hash_capacity<2*n) hash_capacity*=2;
  const uint32_t input_end=(4*n+31U)/32U*32U;
  for(uint32_t byte=0;byte<ub.size();++byte)
   if(byte>=input_end && (byte<1024U || byte>=1024U+4U*hash_capacity))
    assert(ub[byte]==before_validation[byte]);
  for(uint32_t byte=3332;byte<ub.size();++byte) assert(ub[byte]==0xa7);
  for(uint32_t i=0;i<n;++i) assert(AppendKvafForwardResidentRequest(
    legacy.control,&legacy.descriptor,legacy.seeds[i*4+2],128,legacy.seeds[i*4+3]));
  assert(bypass_reads==static_cast<uint64_t>(n)*(n-1U)/2U);
  assert(bulk.indices==legacy.indices && bulk.offsets==legacy.offsets &&
    bulk.outputs==legacy.outputs && bulk.masks==legacy.masks);
  assert(bulk.descriptor.active_request_count==n);
  bulk.guards(); legacy.guards();
 }
}
void duplicate_gate() {
 for(uint32_t n=1;n<=256;++n) {
  Fixture batch;
  // Permuted, descending, singleton and full-capacity waves. Validate only
  // private hash scratch is used; GM request order and canaries stay unchanged.
  for(uint32_t i=0;i<n;++i) batch.seeds[i*4+3]=1000+n-1-i;
  if(n%2==0) for(uint32_t i=0;i<n;++i) {
   const uint32_t j=(i*73+19)%n;
   const int32_t swap=batch.seeds[i*4+3];
   batch.seeds[i*4+3]=batch.seeds[j*4+3]; batch.seeds[j*4+3]=swap;
  }
  assert(batch.build(n));
  auto original=batch.outputs;
  assert(ValidateKvafForwardResidentOutputIndices(batch.control,batch.descriptor));
  assert(batch.outputs==original); batch.guards();
  if(n>1) {
   for(uint32_t dup=1;dup<n;++dup) {
    batch.outputs=original; batch.outputs[dup+1]=batch.outputs[1];
    assert(!ValidateKvafForwardResidentOutputIndices(batch.control,batch.descriptor));
    batch.guards();
   }
  }
 }
 Fixture f; f.seeds[63*4+3]=f.seeds[4*4+3];
 assert(f.build(64)); // Sole Publish gate, not Append, owns duplicate rejection.
 assert(!ValidateKvafForwardResidentOutputIndices(f.control,f.descriptor));
 f.outputs[64]=f.outputs[1]-1;
 assert(!ValidateKvafForwardResidentOutputIndices(f.control,f.descriptor));
 f.outputs[64]=6000;
 assert(!ValidateKvafForwardResidentOutputIndices(f.control,f.descriptor));
}
void hash_collisions() {
 Fixture f;
 f.control.resident_row_base=0; f.control.resident_row_count=INT32_MAX;
 f.control.output_rows=INT32_MAX;
 // Deliberately force every key into one bucket, including wrap at table end.
 // Collision resolution must retain all distinct keys and reject every
 // duplicate position without modifying GM or the unused UB tail.
 uint32_t count=0;
 for(uint32_t value=0;count<256;++value) {
  uint32_t hash=value;
  hash^=hash>>16U; hash*=0x9e3779b1U; hash^=hash>>16U;
  if((hash&511U)==511U) f.seeds[(count++)*4+3]=value;
 }
 assert(f.build(256));
 const auto original=f.outputs;
 ub.fill(0xa7);
 assert(ValidateKvafForwardResidentOutputIndices(f.control,f.descriptor));
 assert(original==f.outputs);
 for(uint32_t byte=3072;byte<ub.size();++byte) assert(ub[byte]==0xa7);
 for(uint32_t dup=2;dup<=256;++dup) {
  f.outputs=original; f.outputs[dup]=original[1];
  assert(!ValidateKvafForwardResidentOutputIndices(f.control,f.descriptor));
  f.guards();
 }
 // Empty-table sentinel -1 must never be accepted as an output row.
 f.outputs=original; f.outputs[128]=-1;
 assert(!ValidateKvafForwardResidentOutputIndices(f.control,f.descriptor));
}
void invalid_inputs() {
 for(uint32_t kind=0;kind<6;++kind) {
  Fixture f;
  if(kind==0) f.seeds[7*4+2]=-1;
  if(kind==1) f.seeds[7*4+2]=2048*128;
  if(kind==2) f.seeds[7*4+3]=999;
  if(kind==3) f.seeds[7*4+3]=6000;
  if(kind==4) f.control.magic=0;
  if(kind==5) f.descriptor.active_request_count=1;
  bypass_writes=dma_bytes=0; assert(!f.build(64));
  assert(bypass_writes==0 && dma_bytes==0); f.guards();
 }
 Fixture f;
 assert(!f.build(0) && !f.build(257));
 assert(!BuildKvafForwardResidentRequestsFromSeedWords(f.control,&f.descriptor,
   f.seeds.data(),64,4,4,3,128));
 assert(!BuildKvafForwardResidentRequestsFromSeedWords(f.control,&f.descriptor,
   f.seeds.data(),64,4,2,3,0));
 // Live source seeds inside the borrowed aperture must be rejected.
 std::memcpy(ub.data(),f.seeds.data(),64*4*sizeof(int32_t));
 dma_bytes=0;
 assert(!BuildKvafForwardResidentRequestsFromSeedWords(f.control,&f.descriptor,
   reinterpret_cast<const int32_t*>(ub.data()),64,4,2,3,128));
 assert(dma_bytes==0 && AscendC::pending.empty());
}
void reuse() {
 Fixture f;
 for(uint32_t generation=0;generation<12;++generation) {
  const uint32_t n=generation%2==0 ? 64:14;
  f.descriptor.active_request_count=0;
  for(uint32_t i=0;i<n;++i) f.seeds[i*4+3]=1000+generation*256+i;
  ub.fill(0xff); assert(f.build(n));
  assert(ValidateKvafForwardResidentOutputIndices(f.control,f.descriptor)); f.guards();
 }
}
void native_vectors() {
 for(uint32_t n: {1U,2U,14U,31U,32U,33U,64U}) {
  Fixture direct,legacy;
  ub.fill(0xa7);
  auto* words=reinterpret_cast<int32_t*>(ub.data());
  for(uint32_t i=0;i<n;++i) {
   words[i]=direct.seeds[i*4+2]/128;
   words[64+i]=(direct.seeds[i*4+2]%128)*656;
   words[128+i]=direct.seeds[i*4+3];
   ub[768+i]=1;
  }
  const auto original=ub;
  dma_bytes=0;
  assert(PublishKvafForwardResidentRequestVectors(direct.control,&direct.descriptor,0,64,n));
  assert(ub==original && dma_bytes==13*n && AscendC::pending.empty() && AscendC::event_state==0);
  assert(legacy.build(n));
  assert(direct.indices==legacy.indices && direct.outputs==legacy.outputs &&
         direct.offsets==legacy.offsets && direct.masks==legacy.masks);
  direct.guards();legacy.guards();
 }
 Fixture f;
 for(uint32_t n: {0U,65U}) assert(!PublishKvafForwardResidentRequestVectors(f.control,&f.descriptor,0,64,n));
 assert(!PublishKvafForwardResidentRequestVectors(f.control,&f.descriptor,1,64,1));
 assert(!PublishKvafForwardResidentRequestVectors(f.control,&f.descriptor,0,63,1));
 assert(!PublishKvafForwardResidentRequestVectors(f.control,&f.descriptor,UINT32_MAX-31,64,1));
}

void dense_preflight() {
 std::array<int32_t,256> indices,offsets,outputs;
 std::array<uint64_t,256> addresses;
 std::array<uint8_t,256> valid;
 KvafBatchGetDirectUrmaEntity entity{};
 auto reset=[&] {
  entity.local_hbm_size=256ULL*656;
  entity.remote_host_ddr_address=0x10000000;
  entity.remote_host_ddr_size=256ULL*kAddressTableEntryBytes;
  for(uint32_t i=0;i<256;++i) {
   indices[i]=i;offsets[i]=0;outputs[i]=255-i;valid[i]=1;
   addresses[i]=entity.remote_host_ddr_address+i*kAddressTableEntryBytes;
  }
 };
 auto check=[&](uint32_t n,uint64_t expected,uint64_t request) {
  uint64_t code=99,failed=99;ub.fill(0xa7);
  ValidateDenseBatchGetDescriptors(addresses.data(),valid.data(),indices.data(),offsets.data(),
    outputs.data(),n,256,256,656,&entity,&code,&failed);
  assert(code==expected && failed==request && AscendC::event_state==0);
  for(uint32_t i=8;i<4096;++i) assert(ub[i]==0xa7);
 };
 for(uint32_t n: {0U,1U,31U,32U,33U,64U,255U,256U}) {reset();check(n,kSuccess,UINT64_MAX);}
 for(uint32_t position: {0U,1U,31U,32U,127U,255U}) for(uint32_t fault=0;fault<9;++fault) {
  reset();
  if(fault==0) indices[position]=-1;
  if(fault==1) indices[position]=256;
  if(fault==2) valid[position]=0;
  if(fault==3) offsets[position]=-1;
  if(fault==4) offsets[position]=kAddressTableEntryBytes-kTransferBytes+1;
  if(fault==5) outputs[position]=-1;
  if(fault==6) outputs[position]=256;
  if(fault==7) addresses[position]=entity.remote_host_ddr_address-1;
  if(fault==8) addresses[position]=entity.remote_host_ddr_address+entity.remote_host_ddr_size;
  check(256,(fault==5 || fault==6)?kInvalidOutput:kInvalidLookup,position);
 }
 reset();indices[33]=-1;outputs[31]=-1;check(256,kInvalidOutput,31); // different lanes/errors
 reset();entity.local_hbm_size=655;check(256,kInvalidOutput,0);
 reset();entity.remote_host_ddr_size=kAddressTableEntryBytes-1;check(256,kInvalidLookup,0);
 reset();entity.remote_host_ddr_address=UINT64_MAX-10;addresses[0]=UINT64_MAX-10;offsets[0]=656;
 check(1,kInvalidLookup,0); // wrapped remote address
}
int main(int argc,char**argv) {
 assert(argc==2); std::string name=argv[1];
 if(name=="equivalence") equivalence(); else if(name=="duplicates") duplicate_gate();
 else if(name=="collisions") hash_collisions();
 else if(name=="invalid") invalid_inputs(); else if(name=="reuse") reuse();
 else if(name=="native_vectors") native_vectors(); else if(name=="dense_preflight") dense_preflight(); else return 2;
 std::cout<<"KVAF_BULK_DESCRIPTOR_ACTUAL_CPP_PASS case="<<name<<" npu_proof=0\n";
}
"""


class ForwardResidentBulkDescriptorNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = next(
            (shutil.which(name) for name in ("clang++", "g++") if shutil.which(name)),
            None,
        )
        if compiler is None:
            raise unittest.SkipTest("host C++ compiler unavailable")
        cls.scratch = tempfile.TemporaryDirectory(prefix="kvaf_bulk_descriptor_")
        cls.addClassCleanup(cls.scratch.cleanup)
        source = (KERNEL / "kvaf_forward_resident_get_device.h").read_text()
        functions = [
            actual_function(source, name)
            for name in (
                "IsKvafForwardResidentBatchGetControlValidForAicore",
                "PrepareKvafForwardResidentDescriptorVectorsForPublication",
                "AppendKvafForwardResidentRequest",
                "BuildKvafForwardResidentVectorsSimt",
                "BuildKvafForwardResidentRequestsFromSeedWords",
                "PublishKvafForwardResidentRequestVectors",
                "ValidateKvafForwardResidentOutputIndices",
            )
        ]
        dense = (KERNEL / "execute_batch_get_aiv.asc").read_text()
        functions += [actual_function(dense, name) for name in (
            "ValidateDenseBatchGetDescriptorsSimt", "ValidateDenseBatchGetDescriptors")]
        path = Path(cls.scratch.name) / "actual.cpp"
        path.write_text(
            PREFIX
            + '\n#include "'
            + str(KERNEL.parent / "include/batch_get_abi.h")
            + '"\n'
            + "using namespace AscendC;\n"
            + "constexpr uint32_t kSuccess=0,kInvalidLookup=4,kInvalidOutput=8;\n"
            + "constexpr uint32_t kTransferBytes=KVAF_AIV_BATCH_GET_WQE_DATA_BYTES;\n"
            + "constexpr uint32_t kAddressTableEntryBytes=KVAF_AIV_BATCH_GET_ADDRESS_TABLE_ENTRY_BYTES;\n"
            + "\n".join(functions)
            + CASES
        )
        cls.binary = path.with_suffix("")
        result = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O1",
                "-Wall",
                "-Wextra",
                "-pthread",
                "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
                str(path),
                "-o",
                str(cls.binary),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def test_actual_bulk_descriptor_values_validation_and_reuse(self):
        for name in ("equivalence", "duplicates", "collisions", "invalid", "reuse", "native_vectors", "dense_preflight"):
            with self.subTest(case=name):
                result = subprocess.run(
                    [str(self.binary), name], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
