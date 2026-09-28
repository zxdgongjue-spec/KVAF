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
  std::uint32_t notify[32];
  kvaf::aicpu_urma::EncodeNotifyWqe(notify, endpoint,
                                    0x1111222233334444ULL, 0x5555666677778888ULL,
                                    0x9999aaaabbbbccccULL, 29U,
                                    0xddddeeeeffff0001ULL, 1U);
  assert(notify[0] == 0xb03e0000U);
  assert(notify[1] == kvaf::aicpu_urma::kWriteWithNotifyOpcode << 8U);
  assert(notify[12] == (endpoint.remote_token_id & 0x000fffffU));
  assert(notify[13] == endpoint.remote_token_value);
  assert(notify[14] == 0x77778888U && notify[15] == 0x55556666U);
  assert(notify[16] == 0xffff0001U && notify[17] == 0xddddeeeeU);
  assert(notify[20] == 8U && notify[21] == 29U);
  assert(notify[22] == 0xbbbbccccU && notify[23] == 0x9999aaaaU);
  assert(notify[24] == 0U && notify[31] == 0U);
  kvaf::aicpu_urma::Transfer reads[16]{};
  const auto count = kvaf::aicpu_urma::PlanContiguousRead(
      reads, 0x100000000ULL, 0x200000000ULL,
      2U * kvaf::aicpu_urma::kMaxReadBytes + 64U, 27U);
  assert(count == 3U);
  assert(reads[0].length == kvaf::aicpu_urma::kMaxReadBytes);
  assert(reads[1].remote_address == 0x100000000ULL + kvaf::aicpu_urma::kMaxReadBytes);
  assert(reads[2].local_address == 0x200000000ULL + 2U * kvaf::aicpu_urma::kMaxReadBytes);
  assert(reads[2].length == 64U && reads[2].local_token_id == 27U);
  assert(kvaf::aicpu_urma::InRange(0x1000, 0x100, 0x1080, 0x80));
  assert(!kvaf::aicpu_urma::InRange(0x1000, 0x100, 0x1080, 0x81));
  assert(!kvaf::aicpu_urma::InRange(0x1000, 0x100, 0xfff, 1));
  std::cout << "AICPU URMA WQE: PASS\n";
}
