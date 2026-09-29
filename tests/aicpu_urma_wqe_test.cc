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
  assert(kvaf::aicpu_urma::ScatterWqebbs(1U) == 1U);
  assert(kvaf::aicpu_urma::ScatterWqebbs(2U) == 2U);
  assert(kvaf::aicpu_urma::ScatterWqebbs(5U) == 2U);
  assert(kvaf::aicpu_urma::ScatterWqebbs(kvaf::kGatherMaxSglSize) == 3U);
  assert(kvaf::aicpu_urma::MaxScatterBatch(2U) == 0U);
  assert(kvaf::aicpu_urma::MaxScatterBatch(64U) == 21U);
  assert(kvaf::aicpu_urma::MaxScatterBatch(8192U) == 64U);
  kvaf::aicpu_urma::ScatterTransfer scatter{};
  scatter.remote_address = 0x1122334455667788ULL;
  scatter.sge_count = kvaf::kGatherMaxSglSize;
  for (std::uint32_t index = 0U; index < scatter.sge_count; ++index) {
    scatter.sges[index] = {0x200000000ULL + index * 4096U, 656U, 27U};
    scatter.bytes += 656U;
  }
  std::uint32_t scatter_words[kvaf::aicpu_urma::kMaxScatterWqebbs * 16U];
  kvaf::aicpu_urma::EncodeScatterReadWqe(scatter_words, endpoint, scatter, 1U, true);
  assert(scatter_words[0] == 0xb03e0000U);
  assert(scatter_words[1] == kvaf::aicpu_urma::kReadOpcode << 8U);
  assert(scatter_words[2] == 0x06345678U);
  assert(scatter_words[10] == 0x55667788U && scatter_words[11] == 0x11223344U);
  for (std::uint32_t index = 0U; index < scatter.sge_count; ++index) {
    const auto base = 12U + index * 4U;
    assert(scatter_words[base] == 656U && scatter_words[base + 1U] == 27U);
    assert(scatter_words[base + 2U] == index * 4096U);
    assert(scatter_words[base + 3U] == 2U);
  }
  for (std::uint32_t index = 36U; index < 48U; ++index)
    assert(scatter_words[index] == 0U);
  scatter.sge_count = 1U;
  kvaf::aicpu_urma::EncodeScatterReadWqe(scatter_words, endpoint, scatter, 0U, true);
  assert(scatter_words[2] == 0x01345678U);
  assert(scatter_words[12] == 656U && scatter_words[13] == 27U);
  std::uint32_t single_read[16];
  kvaf::aicpu_urma::EncodeWqe(single_read, endpoint,
      {scatter.remote_address, scatter.sges[0].address,
       scatter.sges[0].length, scatter.sges[0].token_id},
      kvaf::aicpu_urma::kReadOpcode, 0U, true);
  assert(std::memcmp(scatter_words, single_read, sizeof(single_read)) == 0);
  std::uint32_t compact_words[17];
  for (auto& word : compact_words) word = 0xfeedfaceU;
  kvaf::aicpu_urma::EncodeScatterReadWqe(compact_words, endpoint, scatter, 0U, true);
  assert(std::memcmp(compact_words, single_read, sizeof(single_read)) == 0);
  assert(compact_words[16] == 0xfeedfaceU);
  assert(kvaf::aicpu_urma::InRange(0x1000, 0x100, 0x1080, 0x80));
  assert(!kvaf::aicpu_urma::InRange(0x1000, 0x100, 0x1080, 0x81));
  assert(!kvaf::aicpu_urma::InRange(0x1000, 0x100, 0xfff, 1));
  std::cout << "AICPU URMA WQE: PASS\n";
}
