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
  assert(default_options.backend == kvaf::aicpu_urma_client::Options::Backend::kAicpu);
  assert(default_options.aiv_get_object.find("/bin/kvaf_execute_batch_get_kernels.o") != std::string::npos);
  assert(default_options.aiv_put_object.find("/bin/kvaf_execute_batch_put_kernels.o") != std::string::npos);
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
                                   "--warmup", "0", "--iterations", "5", "--seed", "123",
                                   "--ddr-bytes", "1048576"};
  kvaf::aicpu_urma_client::Options options;
  assert(kvaf::aicpu_urma_client::Parse(sizeof(valid) / sizeof(valid[0]), valid, &options));
  assert(options.blocks == 16 && options.token_bytes == 256 && options.warmup == 0);
  assert(options.ddr_bytes == 1048576 && options.seed == 123);
  constexpr const char* aiv[] = {"client", "--backend", "aiv", "--peer", "127.0.0.1:28991"};
  assert(kvaf::aicpu_urma_client::Parse(sizeof(aiv) / sizeof(aiv[0]), aiv, &options));
  assert(options.backend == kvaf::aicpu_urma_client::Options::Backend::kAiv);
  constexpr const char* invalid_backend[] = {"client", "--backend", "other"};
  assert(!Accepts(invalid_backend));
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
