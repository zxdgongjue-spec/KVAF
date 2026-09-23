#include "aiv_batch_roundtrip_options.h"

#include <cassert>
#include <iostream>

namespace {

template <std::size_t Size>
bool Accepts(const char* const (&arguments)[Size]) {
  kvaf::aiv_batch_roundtrip::Options options;
  return kvaf::aiv_batch_roundtrip::Parse(Size, arguments, &options);
}

}  // namespace

int main() {
  constexpr const char* defaults[] = {"client"};
  kvaf::aiv_batch_roundtrip::Options options;
  assert(Accepts(defaults));
  assert(options.peer == "127.0.0.1:28999");
  assert(options.put_kernel.find("/bin/kvaf_execute_batch_put_kernels.o") != std::string::npos);
  assert(options.get_kernel.find("/bin/kvaf_execute_batch_get_kernels.o") != std::string::npos);

  constexpr const char* valid[] = {"client", "--device-id", "1", "--blocks", "4",
                                   "--block-bytes", "4096", "--token-bytes", "512",
                                   "--iterations", "3", "--peer", "host:28999"};
  assert(kvaf::aiv_batch_roundtrip::Parse(sizeof(valid) / sizeof(valid[0]), valid, &options));
  assert(options.device_id == 1U && options.blocks == 4U && options.token_bytes == 512U);
  assert(options.iterations == 3U && options.peer == "host:28999");

  constexpr const char* missing[] = {"client", "--device-id"};
  constexpr const char* negative[] = {"client", "--device-id", "-1"};
  constexpr const char* zero[] = {"client", "--blocks", "0"};
  constexpr const char* ratio[] = {"client", "--block-bytes", "4095"};
  constexpr const char* capacity[] = {"client", "--blocks", "1025"};
  constexpr const char* overflow[] = {"client", "--device-id", "4294967296"};
  constexpr const char* unknown[] = {"client", "--npu-eid", "1"};
  assert(!Accepts(missing) && !Accepts(negative) && !Accepts(zero));
  assert(!Accepts(ratio) && !Accepts(capacity) && !Accepts(overflow) && !Accepts(unknown));
  std::cout << "AIV native roundtrip options: PASS\n";
}
