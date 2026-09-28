#include "cpu_gather.h"
#include "peer_protocol.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
  std::array<std::uint8_t, 256> memory{};
  const auto base = reinterpret_cast<std::uint64_t>(memory.data());
  for (std::uint32_t index = 0U; index < 128U; ++index)
    memory[index] = static_cast<std::uint8_t>(index);
  const std::vector<kvaf::peer::GatherSpan> spans{
      {base + 80U, 16U, 0U}, {base + 0U, 16U, 0U}, {base + 48U, 16U, 0U}};
  std::uint32_t total = 0U;
  assert(kvaf::peer::ValidateGather(base, memory.size(), base + 128U, spans, &total));
  assert(total == 48U);
  kvaf::peer::GatherIntoStaging(memory.data(), base, base + 128U, spans);
  for (std::uint32_t index = 0U; index < 16U; ++index) {
    assert(memory[128U + index] == 80U + index);
    assert(memory[144U + index] == index);
    assert(memory[160U + index] == 48U + index);
  }
  assert(!kvaf::peer::ValidateGather(base, memory.size(), base + 64U, spans, &total));
  assert(!kvaf::peer::ValidateGather(base, memory.size(), base + 192U, spans, &total));
  const std::vector<kvaf::peer::GatherSpan> bad{{base + 200U, 8U, 0U}};
  assert(!kvaf::peer::ValidateGather(base, memory.size(), base + 128U, bad, &total));
  assert(kvaf::peer::GatherControlOffset(256U) == 192U);
  assert(sizeof(kvaf::peer::GatherSpan) == sizeof(kvaf::peer::GatherSpanWire));
  std::cout << "CPU gather: PASS\n";
}
