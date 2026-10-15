#include "output_readback.h"
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace {
using namespace kvaf::aicpu_test;
void Check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void Readback() {
  for (auto payload : {0U, 1U}) {
    Config c; c.rows = 7; c.row_bytes = 65; c.verify_payload = c.verify_each_round = payload;
    Layout l{}; Check(MakeLayout(c, &l), "invalid test layout");
    OutputReadbackVerifier verifier(c, l);
    std::vector<std::uint8_t> data(l.output_bytes, 0xA5);
    for (std::uint64_t seq : {1, 2, 257}) {
      for (std::uint64_t slot = 0; slot < c.rows; ++slot) {
        const auto row = LogicalRowForOutputSlot(c, slot), tag = RowTag(seq, row);
        std::memcpy(data.data() + slot * l.stride, &tag, 8);
        for (std::uint64_t b = 8; b < c.row_bytes; ++b)
          data[slot * l.stride + b] = PayloadPattern(c, seq, row, b);
      }
      OutputMismatch mismatch{};
      Check(verifier.Check(seq, data.data(), &mismatch), "valid payload rejected");
      Check(!verifier.Check(seq + 1, data.data(), &mismatch), "stale row tag accepted");
      Check(verifier.Check(seq, data.data(), &mismatch), "checker did not recover after mismatch");
      for (auto offset : {std::uint64_t{0}, c.row_bytes - 1, l.stride - 1}) {
        data[offset] ^= 1;
        Check(!verifier.Check(seq, data.data(), &mismatch) && mismatch.offset == offset,
              "header/payload/guard corruption was not located");
        data[offset] ^= 1;
      }
    }
  }
}
struct AckIO {
  std::uint64_t clock{}, token{}, published{};
  std::uint64_t Now() { return ++clock; }
  void PublishOutputReady(std::uint64_t seq) { published = seq; }
  Control ReadOutputVerified() { Control c{}; c.sequence = token; return c; }
  void Relax() {}
};
void Acknowledgements() {
  AckIO io; io.token = OutputVerificationToken(3, true);
  Check(VerifyOutputViaUrma(3, 10, io) == Result::kOk && io.published == 3, "valid ACK rejected");
  io.token = OutputVerificationToken(3, false);
  Check(VerifyOutputViaUrma(3, 10, io) == Result::kDataMismatch, "failed ACK ignored");
  io.token = OutputVerificationToken(4, true);
  Check(VerifyOutputViaUrma(3, 10, io) == Result::kSequenceError, "future ACK accepted");
  io.token = OutputVerificationToken(2, true);
  Check(VerifyOutputViaUrma(3, 2, io) == Result::kTimeout, "stale ACK accepted");
}
}
int main() {
  try { Readback(); Acknowledgements(); std::puts("output verification: PASS"); return 0; }
  catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
