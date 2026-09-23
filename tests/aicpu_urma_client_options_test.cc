#include "aicpu_urma_client_options.h"

#include <cassert>
#include <cstdint>
#include <iostream>

namespace {

template <std::size_t Size>
bool Accepts(const char* const (&arguments)[Size]) {
  kvaf::aicpu_urma_client::Options options;
  return kvaf::aicpu_urma_client::Parse(Size, arguments, &options);
}

}  // namespace

int main() {
  constexpr const char* valid[] = {"client", "--peer", "127.0.0.1:28999", "--kernel-json", "/tmp/kernel.json",
                                   "--blocks", "16", "--block-bytes", "4096", "--token-bytes", "256",
                                   "--warmup", "0", "--iterations", "5", "--seed", "123",
                                   "--ddr-bytes", "1048576"};
  kvaf::aicpu_urma_client::Options options;
  assert(kvaf::aicpu_urma_client::Parse(sizeof(valid) / sizeof(valid[0]), valid, &options));
  assert(options.blocks == 16 && options.token_bytes == 256 && options.warmup == 0);
  assert(options.ddr_bytes == 1048576 && options.seed == 123);
  constexpr const char* invalid_ratio[] = {"client", "--peer", "host:1", "--kernel-json", "file",
                                           "--block-bytes", "257", "--token-bytes", "256"};
  constexpr const char* invalid_count[] = {"client", "--peer", "host:1", "--kernel-json", "file",
                                           "--blocks", "2147483648"};
  constexpr const char* invalid_device[] = {"client", "--peer", "host:1", "--kernel-json", "file",
                                            "--device-id", "4294967295"};
  constexpr const char* invalid_ddr[] = {"client", "--peer", "host:1", "--kernel-json", "file",
                                         "--ddr-bytes", "4095"};
  constexpr const char* missing_value[] = {"client", "--peer"};
  constexpr const char* negative[] = {"client", "--peer", "host:1", "--kernel-json", "file",
                                      "--iterations", "-1"};
  assert(!Accepts(invalid_ratio));
  assert(!Accepts(invalid_count));
  assert(!Accepts(invalid_device));
  assert(!Accepts(invalid_ddr));
  assert(!Accepts(missing_value));
  assert(!Accepts(negative));
  std::cout << "AICPU URMA native client options: PASS\n";
}
