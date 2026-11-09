#include "cpu_ddr_registration.h"
#include <iostream>
#include <stdexcept>
#define CHECK(c) do { if (!(c)) throw std::runtime_error(#c); } while (false)
namespace {
urma_token_id_t token{};
urma_target_seg_t segment{};
int allocations{}, registrations{}, unregisters{}, frees{};
bool fail_register{}, fail_unregister{}, fail_free{};
}
extern "C" urma_token_id_t* urma_alloc_token_id(urma_context_t*) { ++allocations; token.token_id = 0x12300; return &token; }
extern "C" urma_target_seg_t* urma_register_seg(urma_context_t*, urma_seg_cfg_t* config) {
  ++registrations;
  CHECK(config->flag.value == kvaf::CpuDdrRegistration::Flags().value);
  CHECK(config->flag.bs.access == 14 && config->flag.bs.non_pin && config->flag.bs.token_id_valid);
  if (fail_register) return nullptr;
  segment.seg.ubva.va = config->va + 0x10000;
  segment.seg.len = config->len;
  return &segment;
}
extern "C" urma_status_t urma_unregister_seg(urma_target_seg_t* value) {
  CHECK(value == &segment); ++unregisters; return fail_unregister ? URMA_FAIL : URMA_SUCCESS;
}
extern "C" urma_status_t urma_free_token_id(urma_token_id_t* value) {
  CHECK(value == &token); ++frees; return fail_free ? URMA_FAIL : URMA_SUCCESS;
}
int main() {
  kvaf::CpuDdrRegistration registration;
  auto* context = reinterpret_cast<urma_context_t*>(1);
  auto* memory = reinterpret_cast<void*>(0x2000);
  CHECK(registration.Register(context, memory, 4096, 2).ok());
  const auto descriptor = registration.Describe(123);
  CHECK(descriptor.valid() && descriptor.registered_address == 0x12000 && descriptor.session_id == 123);
  CHECK(descriptor.token_id == 0x12300 && descriptor.token_value == 2);
  CHECK(!registration.Register(context, memory, 4096, 2).ok());
  fail_unregister = true;
  CHECK(!registration.Close().ok() && registration.segment() && frees == 0);
  fail_unregister = false; fail_free = true;
  CHECK(!registration.Close().ok() && !registration.segment() && registration.token_id());
  fail_free = false;
  CHECK(registration.Close().ok() && unregisters == 2 && frees == 2);
  fail_register = true;
  CHECK(!registration.Register(context, memory, 4096, 2).ok() && registration.token_id());
  CHECK(registration.Close().ok() && unregisters == 2 && frees == 3);
  CHECK(registration.Close().ok() && frees == 3);
  std::cout << "CPU DDR registration retry and descriptor passed\n";
}
