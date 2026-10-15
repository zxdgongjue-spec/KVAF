#pragma once

#include "abi.h"

#include <cstring>

namespace kvaf::aicpu_test {
// Control-only handshake. The device must not touch output between publishing
// ready and receiving the external consumer's ACK; buffer reuse follows ACK.
template <class IO>
Result VerifyOutputViaUrma(std::uint64_t sequence, std::uint64_t timeout, IO& io) {
  const auto begin = io.Now();
  io.PublishOutputReady(sequence);
  for (;;) {
    const auto ack = io.ReadOutputVerified();
    const auto ack_sequence = ack.sequence >> 1;
    if (ack_sequence > sequence) return Result::kSequenceError;
    if (ack_sequence == sequence) return (ack.sequence & 1) ? Result::kDataMismatch : Result::kOk;
    if (io.Now() - begin >= timeout) return Result::kTimeout;
    io.Relax();
  }
}

struct OutputMismatch {
  std::uint64_t offset{};
  std::uint8_t expected{}, actual{};
};

}  // namespace kvaf::aicpu_test
