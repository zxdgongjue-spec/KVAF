#include "aicpu_urma/wqe.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

int main() {
  kvaf::aicpu_urma::Endpoint endpoint{};
  endpoint.tp_id = 0x12345678U;
  endpoint.remote_token_id = 0xf1234567U;
  endpoint.remote_token_value = 0xaabbccddU;
  for (std::uint8_t index = 0; index < 16; ++index) endpoint.remote_eid[index] = index;
  const kvaf::aicpu_urma::Transfer transfer{0x1122334455667788ULL,
                                              0x99aabbccddeeff00ULL, 4096, 27};
  for (const auto opcode : {kvaf::aicpu_urma::kReadOpcode,
                            kvaf::aicpu_urma::kWriteOpcode}) {
    std::uint32_t words[16];
    kvaf::aicpu_urma::EncodeWqe(words, endpoint, transfer, opcode, 0, true);
    assert(words[0] == 0x303e0000U);
    assert(words[1] == opcode << 8U);
    assert(words[2] == 0x01345678U && words[3] == 0x00034567U);
    assert(std::memcmp(&words[4], endpoint.remote_eid, 16) == 0);
    assert(words[8] == 0xaabbccddU && words[9] == 0U);
    assert(words[10] == 0x55667788U && words[11] == 0x11223344U);
    assert(words[12] == 4096U && words[13] == 27U);
    assert(words[14] == 0xddeeff00U && words[15] == 0x99aabbccU);
    kvaf::aicpu_urma::EncodeWqe(words, endpoint, transfer, opcode, 0, false);
    assert(words[0] == 0x30150000U);
    kvaf::aicpu_urma::EncodeWqe(words, endpoint, transfer, opcode, 1, true);
    assert(words[0] == 0xb03e0000U);
  }
  assert(kvaf::aicpu_urma::InRange(0x1000, 0x100, 0x1080, 0x80));
  assert(!kvaf::aicpu_urma::InRange(0x1000, 0x100, 0x1080, 0x81));
  assert(!kvaf::aicpu_urma::InRange(0x1000, 0x100, 0xfff, 1));
  std::cout << "AICPU URMA WQE: PASS\n";
}
