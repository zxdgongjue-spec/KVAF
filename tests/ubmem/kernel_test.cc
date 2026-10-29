#include "kernel_operator.h"
#include "ubmem/batch_abi.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define __attribute__(...)
#include "../../access_operator/src/ubmem/batch_kernel.asc"
#undef __attribute__
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

using namespace kvaf::ubmem;
template <typename T> std::uint64_t Ptr(T* value) { return reinterpret_cast<std::uintptr_t>(value); }
void Run(BatchArgs args) {
  for (model_core = 0; model_core < args.blocks; ++model_core) KvafUbmemBatchKernel(args);
}
int main() {
  constexpr std::uint64_t remote_base = 0x100000000ULL;
  std::vector<std::uint8_t> remote(32768, 0xCF), local(5000, 0xD7);
  for (std::size_t i = 0; i < remote.size(); ++i) remote[i] = i * 37 + 3;
  std::array<CoreStatus, 3> statuses{};
  std::uint64_t addresses[]{remote_base, remote_base + 8192};
  std::uint8_t valid[]{1, 1};
  std::int32_t indices[]{1, 0, 1, -999}, offsets[]{0, 17, 656, -999}, outputs[]{2, 0, 1, -999};
  std::int32_t count = 3;
  BatchArgs a{};
  a.magic = kBatchMagic; a.remote_base = remote_base; a.remote_bytes = remote.size();
  a.mapped_base = Ptr(remote.data()); a.local_base = Ptr(local.data() + 1); a.local_bytes = 3 * 656;
  a.addresses = Ptr(addresses); a.valid = Ptr(valid); a.indices = Ptr(indices); a.offsets = Ptr(offsets);
  a.output_indices = Ptr(outputs); a.miss_count = Ptr(&count); a.statuses = Ptr(statuses.data());
  a.operation = Operation::kGet; a.blocks = 3; a.capacity = 4; a.table_entries = 2;
  a.token_bytes = 656; a.tokens_per_block = 8;
  Run(a);
  for (unsigned r = 0; r < 3; ++r) {
    CHECK(statuses[r].result == 0 && statuses[r].completed == 1);
    for (unsigned b = 0; b < 656; ++b)
      CHECK(local[1 + outputs[r] * 656 + b] == remote[addresses[indices[r]] - remote_base + offsets[r] + b]);
  }
  CHECK(local.front() == 0xD7 && local[1 + 3 * 656] == 0xD7);
  const auto previous = local;
  count = 0; Run(a);
  CHECK(local == previous);
  for (auto& s : statuses) CHECK(s.result == 0 && s.completed == 0);
  count = -1; Run(a);
  for (auto& s : statuses) CHECK(s.result == 1);
  CHECK(local == previous);
  count = 5; Run(a);
  for (auto& s : statuses) CHECK(s.result == 1);
  count = 3; valid[1] = 0; Run(a); CHECK(statuses[0].result == 2 && statuses[2].result == 2);
  valid[1] = 1; offsets[0] = 5248; Run(a); CHECK(statuses[0].result == 2);
  offsets[0] = 0; outputs[0] = 3; Run(a); CHECK(statuses[0].result == 3);
  outputs[0] = 2; addresses[1] = remote_base + remote.size() - 16; Run(a); CHECK(statuses[0].result == 2);

  std::uint64_t put_offsets[]{1, 20, 680}, lengths[]{1, 655, 4097};
  std::uint64_t destinations[]{remote_base + 1, remote_base + 8193, remote_base + 16384};
  a.operation = Operation::kPut; a.capacity = 3; a.local_base = Ptr(local.data()); a.local_bytes = local.size();
  a.put_offsets = Ptr(put_offsets); a.put_lengths = Ptr(lengths); a.put_destinations = Ptr(destinations);
  Run(a);
  for (unsigned r = 0; r < 3; ++r) {
    CHECK(statuses[r].result == 0);
    for (unsigned b = 0; b < lengths[r]; ++b)
      CHECK(remote[destinations[r] - remote_base + b] == local[put_offsets[r] + b]);
  }
  destinations[2] = remote_base + remote.size() - 10; Run(a); CHECK(statuses[2].result == 2);
  destinations[2] = remote_base; put_offsets[2] = local.size(); Run(a); CHECK(statuses[2].result == 3);
  std::puts("UBmem actual kernel: mapped addressing, 656-byte reads, variable puts, inactive tail, guards, and invalid ranges passed");
}
