#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

namespace kvaf::aicpu_urma_client {

struct Options {
  std::string peer;
  std::string kernel_json;
  std::uint32_t device_id{0};
  std::uint32_t blocks{8};
  std::uint32_t block_bytes{4096};
  std::uint32_t token_bytes{256};
  std::uint32_t warmup{1};
  std::uint32_t iterations{3};
  std::uint64_t seed{0x6b766166U};
  std::uint64_t ddr_bytes{0};
};

inline bool Number(const char* text, std::uint64_t* value) {
  if (text == nullptr || text[0] == '\0' || text[0] == '-') return false;
  char* end = nullptr;
  errno = 0;
  const auto parsed = std::strtoull(text, &end, 0);
  if (errno != 0 || end == text || *end != '\0') return false;
  *value = static_cast<std::uint64_t>(parsed);
  return true;
}

inline bool Parse(int argc, const char* const* argv, Options* options) {
  if (argc < 2 || options == nullptr) return false;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) return false;
    const std::string flag = argv[index];
    const char* value = argv[index + 1];
    if (flag == "--peer") options->peer = value;
    else if (flag == "--kernel-json") options->kernel_json = value;
    else {
      std::uint64_t number = 0;
      if (!Number(value, &number)) return false;
      if (flag == "--seed") options->seed = number;
      else if (flag == "--ddr-bytes") options->ddr_bytes = number;
      else {
        if (number > UINT32_MAX) return false;
        if (flag == "--device-id") options->device_id = static_cast<std::uint32_t>(number);
        else if (flag == "--blocks") options->blocks = static_cast<std::uint32_t>(number);
        else if (flag == "--block-bytes") options->block_bytes = static_cast<std::uint32_t>(number);
        else if (flag == "--token-bytes") options->token_bytes = static_cast<std::uint32_t>(number);
        else if (flag == "--warmup") options->warmup = static_cast<std::uint32_t>(number);
        else if (flag == "--iterations") options->iterations = static_cast<std::uint32_t>(number);
        else return false;
      }
    }
  }
  if (options->peer.empty() || options->kernel_json.empty() || options->blocks == 0 ||
      options->device_id > INT32_MAX || options->token_bytes == 0 ||
      options->block_bytes == 0 || options->iterations == 0 ||
      options->block_bytes % options->token_bytes != 0 ||
      options->block_bytes > INT32_MAX ||
      options->blocks > static_cast<std::uint32_t>(INT32_MAX) /
                            (options->block_bytes / options->token_bytes) ||
      options->blocks > std::numeric_limits<std::size_t>::max() / options->block_bytes) return false;
  const std::uint64_t payload_bytes = static_cast<std::uint64_t>(options->blocks) * options->block_bytes;
  if (payload_bytes > (1ULL << 32U) ||
      (options->ddr_bytes != 0 && (options->ddr_bytes < payload_bytes ||
                                   options->ddr_bytes % options->block_bytes != 0))) return false;
  return true;
}

}  // namespace kvaf::aicpu_urma_client
