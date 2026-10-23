#include "cpu_push_get_layout.h"
#include "cpu_push_pipeline.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace kvaf::cpu_push;
void Require(bool valid, const char* message) { if (!valid) throw std::runtime_error(message); }
int main() {
  try {
    Layout l;
    Require(!MakeLayout(0, 1, &l) && !MakeLayout(656, 0, &l) &&
            !MakeLayout(656, 9, &l) && !MakeLayout(65536, 1, &l), "invalid workspace accepted");
    for (auto bytes : {1U, 63U, 64U, 65U, 656U, 4096U})
      for (auto workers : {1U, 2U, 4U, 8U}) {
        Require(MakeLayout(bytes, workers, &l), "workspace rejected");
        Require(l.staging % 64 == 0 && l.bytes % 64 == 0 &&
                l.descriptor + kMaxRequests * sizeof(kvaf::peer::GatherSpan) <= l.staging,
                "workspace overlap/alignment");
        for (auto count : {0U, 1U, 3U, 1025U, 65535U, 65536U}) {
          std::uint32_t next = 0;
          for (std::uint32_t w = 0; w < workers; ++w) {
            const auto r = WorkerRange(count, bytes, w, workers);
            Require(r.begin == next && r.end >= r.begin && r.end <= count, "partition gap/overlap");
            if (r.begin < count) Require(std::uint64_t{r.begin} * bytes % 64 == 0, "shared staging line");
            next = r.end;
          }
          Require(next == count, "lost tail");
        }
      }
    std::array<std::uint64_t, 4> tokens{Progress(7, 3), Progress(7, 2), Progress(7, 3), Progress(7, 3)};
    std::uint32_t prefix = 0;
    auto read = [&](auto q) { return tokens[q]; };
    Require(ReadyPrefix(7, 3, 0, read, &prefix) == Result::kPending && prefix == 2, "not all Jettys ready");
    tokens[1] = Progress(6, 3);
    Require(ReadyPrefix(7, 3, 1, read, &prefix) == Result::kPending && prefix == 1, "stale ready consumed");
    tokens[1] = Progress(8, 1);
    Require(ReadyPrefix(7, 3, 0, read, &prefix) == Result::kSequenceError, "future sequence accepted");
    tokens[1] = Progress(7, 1);
    Require(ReadyPrefix(7, 3, 2, read, &prefix) == Result::kSequenceError, "regressing progress accepted");
    tokens[1] = Progress(7, 4);
    Require(ReadyPrefix(7, 3, 0, read, &prefix) == Result::kSequenceError, "excess progress accepted");
    Require(SourceRange(100, 20, 110, 10) && !SourceRange(100, 20, 110, 11) &&
            !SourceRange(100, 20, 99, 1), "source bounds");
    std::cout << "CPU push protocol PASS\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
