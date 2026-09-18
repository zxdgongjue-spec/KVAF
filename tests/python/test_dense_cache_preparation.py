"""Check production UB-fed cache ranges without pretending to model DMA timing."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_exact_cache_line_coverage(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = (Path(__file__).resolve().parents[2] /
              "access_operator/demo/aiv_batch_get/kernel/execute_batch_get_aiv.asc").read_text()
    start = source.index("__aicore__ inline void PrepareDenseBatchGetOutputForInboundDma(")
    end = source.index("__aicore__ inline uint32_t WqeControl", start)
    body = source[start:end]
    assert "ReadGmByPassDCache" not in body
    cpp = tmp_path / "cache.cpp"
    cpp.write_text(r'''
#include <cassert>
#include <cstdint>
#include <set>
#include <vector>
#define __aicore__
#define __gm__
#define __ubuf__
enum class MemDsbT { DDR };
unsigned barriers=0,calls=0;
template<MemDsbT> void DataSyncBarrier(){++barriers;}
std::set<uint64_t> lines;
void CleanCacheLines(uint8_t* p,uint64_t n){
 ++calls;uint64_t a=reinterpret_cast<uint64_t>(p);
 for(uint64_t b=a/64;b<=(a+n-1)/64;++b)lines.insert(b);
}
''' + body + r'''
int main(){
 for(unsigned n=0;n<=256;++n)for(unsigned mode=0;mode<4;++mode){
  std::vector<int32_t> rows(n);std::set<uint64_t> expected;
  for(unsigned i=0;i<n;++i){
   rows[i]=mode==0?int(i):mode==1?int(255-i):mode==2?int((i*37)%256):-1;
   if(rows[i]<0)continue;
   uint64_t a=0x100010+uint64_t(rows[i])*656;
   for(uint64_t b=a/64;b<=(a+655)/64;++b)expected.insert(b);
  }
  lines.clear();calls=barriers=0;
  PrepareDenseBatchGetOutputForInboundDma(rows.data(),n,0x100010,656);
  assert(lines==expected);
  if(n)assert(barriers==1);
  if(n && mode==0)assert(calls==1);
 }
}
''')
    executable = tmp_path / "cache"
    subprocess.run([compiler, "-std=c++17", "-fsanitize=address,undefined", str(cpp), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
