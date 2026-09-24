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
  constexpr const char* defaults[] = {"client"};
  kvaf::aicpu_urma_client::Options default_options;
  assert(kvaf::aicpu_urma_client::Parse(1, defaults, &default_options));
  assert(default_options.peer == "127.0.0.1:28999");
  assert(default_options.token_bytes == default_options.block_bytes);
  assert(default_options.kernel_json.find("/opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json") !=
         std::string::npos);
#if defined(KVAF_AICPU_URMA_CANN_ROOT)
  assert(default_options.kernel_json.rfind(KVAF_AICPU_URMA_CANN_ROOT, 0) == 0);
#endif
  const char* saved_home = std::getenv("ASCEND_HOME_PATH");
  const bool had_home = saved_home != nullptr;
  const std::string previous_home = saved_home == nullptr ? "" : saved_home;
  assert(unsetenv("ASCEND_HOME_PATH") == 0);
  const std::string default_root = kvaf::aicpu_urma_client::DefaultCannRoot();
  assert(kvaf::aicpu_urma_client::ConfigureCannEnvironment());
  assert(std::string(std::getenv("ASCEND_HOME_PATH")) == default_root);
  assert(setenv("ASCEND_HOME_PATH", "/custom/cann", 1) == 0);
  assert(kvaf::aicpu_urma_client::ConfigureCannEnvironment());
  assert(std::string(std::getenv("ASCEND_HOME_PATH")) == "/custom/cann");
  if (!had_home) assert(unsetenv("ASCEND_HOME_PATH") == 0);
  else assert(setenv("ASCEND_HOME_PATH", previous_home.c_str(), 1) == 0);
  constexpr const char* valid[] = {"client", "--peer", "127.0.0.1:28999", "--kernel-json", "/tmp/kernel.json",
                                   "--blocks", "16", "--block-bytes", "4096", "--token-bytes", "256",
                                   "--warmup", "0", "--iterations", "5"};
  kvaf::aicpu_urma_client::Options options;
  assert(kvaf::aicpu_urma_client::Parse(sizeof(valid) / sizeof(valid[0]), valid, &options));
  assert(options.blocks == 16 && options.token_bytes == 256 && options.warmup == 0);
  constexpr const char* block_first[] = {"client", "--block-bytes", "656"};
  assert(kvaf::aicpu_urma_client::Parse(sizeof(block_first) / sizeof(block_first[0]), block_first, &options));
  assert(options.block_bytes == 656 && options.token_bytes == 656);
  constexpr const char* token_first[] = {"client", "--token-bytes", "328", "--block-bytes", "656"};
  assert(kvaf::aicpu_urma_client::Parse(sizeof(token_first) / sizeof(token_first[0]), token_first, &options));
  assert(options.block_bytes == 656 && options.token_bytes == 328);
  constexpr const char* block_then_token[] = {"client", "--block-bytes", "656", "--token-bytes", "328"};
  assert(kvaf::aicpu_urma_client::Parse(sizeof(block_then_token) / sizeof(block_then_token[0]),
                                        block_then_token, &options));
  assert(options.block_bytes == 656 && options.token_bytes == 328);
  constexpr const char* invalid_backend[] = {"client", "--backend", "aiv"};
  constexpr const char* invalid_seed[] = {"client", "--seed", "123"};
  constexpr const char* invalid_ddr_option[] = {"client", "--ddr-bytes", "1048576"};
  constexpr const char* invalid_ratio[] = {"client", "--peer", "host:1", "--kernel-json", "file",
                                           "--block-bytes", "656", "--token-bytes", "256"};
  constexpr const char* invalid_token[] = {"client", "--token-bytes", "0"};
  constexpr const char* invalid_count[] = {"client", "--peer", "host:1", "--kernel-json", "file",
                                           "--blocks", "2147483648"};
  constexpr const char* invalid_device[] = {"client", "--peer", "host:1", "--kernel-json", "file",
                                            "--device-id", "4294967295"};
  constexpr const char* missing_value[] = {"client", "--peer"};
  constexpr const char* negative[] = {"client", "--peer", "host:1", "--kernel-json", "file",
                                      "--iterations", "-1"};
  assert(!Accepts(invalid_backend));
  assert(!Accepts(invalid_seed));
  assert(!Accepts(invalid_ddr_option));
  assert(!Accepts(invalid_ratio));
  assert(!Accepts(invalid_token));
  assert(!Accepts(invalid_count));
  assert(!Accepts(invalid_device));
  assert(!Accepts(missing_value));
  assert(!Accepts(negative));
  std::cout << "AICPU URMA native client options: PASS\n";
}
