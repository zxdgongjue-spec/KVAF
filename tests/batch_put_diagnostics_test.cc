#include <cstdio>
#include <cstdint>
#include "batch_put_diagnostics.h"

#define CHECK(expr) do { if (!(expr)) { std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

int main() {
  using namespace kvaf::put_diag;
  CHECK(InRange(0x1020, 64, 0x1000, 128));
  CHECK(!InRange(0xfff, 1, 0x1000, 128));
  CHECK(!InRange(0x1041, 64, 0x1000, 128));
  CHECK(!InRange(UINT64_MAX - 1, 4, UINT64_MAX - 8, 8));
  CHECK(RequestOffset(0, 1, 2) == 32);
  CHECK(RequestOffset(31, 0, 2) == 31);
  CHECK(RequestOffset(32, 0, 2) == 64);
  CHECK(RequestOffset(32, 1, 2) == 96);
  KvafBatchPutDirectUrmaEntity e{};
  e.local_hbm_address = 0x120000001000ULL;
  e.local_hbm_size = 2048;
  e.remote_host_ddr_address = 0xfffe00001000ULL;
  e.remote_host_ddr_size = 4096;
  e.local_token_id = 0x1100;
  e.remote_token_id = 0x76b;
  e.remote_token_value = 0xabcdef;
  e.tp_id = 0x1234;
  for (uint32_t i = 0; i < 16; ++i) e.remote_eid[i] = static_cast<uint8_t>(i);
  uint32_t w[16]{};
  w[2] = e.tp_id | 0x01000000U;
  w[3] = e.remote_token_id;
  w[8] = e.remote_token_value;
  for (uint32_t i = 0; i < 16; ++i) w[4 + i / 4] |= uint32_t(e.remote_eid[i]) << ((i % 4) * 8);
  w[10] = uint32_t(e.remote_host_ddr_address); w[11] = uint32_t(e.remote_host_ddr_address >> 32);
  w[12] = 64; w[13] = e.local_token_id;
  w[14] = uint32_t(e.local_hbm_address); w[15] = uint32_t(e.local_hbm_address >> 32);
  auto good = CheckWqe(w, e);
  CHECK(good.local_range && good.remote_range && good.local_token && good.remote_token && good.remote_eid && good.tpn);
  w[13] <<= 8; CHECK(!CheckWqe(w, e).local_token); w[13] = e.local_token_id;
  w[14] += 2048; CHECK(!CheckWqe(w, e).local_range); w[14] -= 2048;
  w[10] += 4096; CHECK(!CheckWqe(w, e).remote_range); w[10] -= 4096;
  w[8] ^= 1; CHECK(!CheckWqe(w, e).remote_token); w[8] ^= 1;
  w[4] ^= 1; CHECK(!CheckWqe(w, e).remote_eid); w[4] ^= 1;
  w[2] ^= 1; CHECK(!CheckWqe(w, e).tpn);
  return 0;
}
