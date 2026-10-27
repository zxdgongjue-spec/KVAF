#pragma once
#include <cstddef>
#include <cstdint>

// Diagnostic control traffic only. No KV payload is sent through this socket.
// Both hosts must be little endian and use this exact version of the probe.
namespace ddr_probe {
constexpr std::uint32_t kMagic = 0x44525042, kVersion = 1;
constexpr std::uint64_t kGuard = 4096;
enum Op : std::uint32_t { Info = 1, Arm, Export, Pull, Ack, Close, Fail };
#pragma pack(push, 1)
struct Message {
  std::uint32_t magic{kMagic}, version{kVersion}, op{};
  std::int32_t status{};
  std::uint64_t base{}, bytes{}, round{}, ack{}, tp{};
  std::uint32_t psn{}, token{}, key_bytes{}, segment_bytes{};
  std::uint8_t eid[16]{}, key[64]{}, segment[64]{};
  char error[192]{};
};
#pragma pack(pop)
static_assert(sizeof(Message) == 408);
inline std::uint8_t Pattern(std::uint64_t index, std::uint64_t round) {
  auto x = index ^ (round * UINT64_C(0x9e3779b97f4a7c15));
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return static_cast<std::uint8_t>(x ^ (x >> 31));
}
}  // namespace ddr_probe
