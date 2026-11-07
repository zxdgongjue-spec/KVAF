#include "../runtime/src/ubmem/host_memory.h"

#include <linux/mempolicy.h>
#include <sys/syscall.h>

#include <cerrno>
#include <cstdarg>
#include <iostream>
#include <stdexcept>

namespace {
int expected_node = -1;
unsigned calls = 0;
constexpr unsigned long kBits = sizeof(unsigned long) * 8;
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
}

extern "C" long __wrap_syscall(long number, ...) {
  ++calls;
  va_list args;
  va_start(args, number);
  const auto address = va_arg(args, void*);
  const auto bytes = va_arg(args, std::size_t);
  const auto mode = va_arg(args, int);
  const auto mask = va_arg(args, const unsigned long*);
  const auto maxnode = va_arg(args, unsigned long);
  const auto flags = va_arg(args, unsigned long);
  va_end(args);
  Require(number == SYS_mbind && address == reinterpret_cast<void*>(0x200000) &&
          bytes == 0x200000 && flags == 0, "range-local mbind arguments");
  if (expected_node < 0) {
    Require(mode == MPOL_DEFAULT && mask == nullptr && maxnode == 0, "reset policy ABI");
    return 0;
  }
  Require(mode == MPOL_BIND && mask != nullptr, "explicit memory binding");
  // Linux mm/mempolicy.c:get_nodes decrements maxnode before get_bitmap.
  // Reject the original regression: its only set bit was outside this range.
  if (maxnode <= static_cast<unsigned long>(expected_node) + 1) {
    errno = EINVAL;
    return -1;
  }
  for (unsigned long bit = 0; bit < maxnode - 1; ++bit) {
    const bool set = (mask[bit / kBits] >> (bit % kBits)) & 1UL;
    Require(set == (bit == static_cast<unsigned long>(expected_node)), "exactly the selected node survives");
  }
  return 0;
}

int main() {
  for (const int node : {0, 1, 2, 3, 63, 64, 127, 128, 4095, -1}) {
    expected_node = node;
    Require(kvaf::ubmem_host::BindNuma(reinterpret_cast<void*>(0x200000), 0x200000, node).ok(),
            "NUMA binding must preserve the selected node in the kernel mask");
  }
  const auto before = calls;
  Require(!kvaf::ubmem_host::BindNuma(nullptr, 0, -2).ok(), "reject invalid negative node");
  Require(!kvaf::ubmem_host::BindNuma(nullptr, 0, 4096).ok(), "reject node beyond supported range");
  Require(before == calls, "invalid nodes do not issue a syscall");
  std::cout << "PASS: UBmem Linux NUMA syscall mask boundaries\n";
}
