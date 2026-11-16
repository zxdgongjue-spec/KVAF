#include "memory/cpu_shared_registration.h"
#include <algorithm>
#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

#define CHECK(c) do { if (!(c)) throw std::runtime_error(#c); } while (false)
namespace {
std::atomic<unsigned> allocations{}, registrations{}, unregisters{}, frees{};
std::atomic<bool> fail_register{}, fail_unregister{}, fail_free{};
}
extern "C" urma_token_id_t* urma_alloc_token_id(urma_context_t*) {
  auto* token = new urma_token_id_t{};
  token->token_id = ++allocations;
  return token;
}
extern "C" urma_target_seg_t* urma_register_seg(urma_context_t*, urma_seg_cfg_t* cfg) {
  ++registrations;
  if (fail_register) return nullptr;
  auto* segment = new urma_target_seg_t{};
  segment->seg.ubva.va = cfg->va + 0x100000;
  segment->seg.len = cfg->len;
  segment->seg.token_id = cfg->token_id->token_id;
  return segment;
}
extern "C" urma_status_t urma_unregister_seg(urma_target_seg_t* segment) {
  ++unregisters;
  if (fail_unregister) return URMA_FAIL;
  delete segment;
  return URMA_SUCCESS;
}
extern "C" urma_status_t urma_free_token_id(urma_token_id_t* token) {
  ++frees;
  if (fail_free) return URMA_FAIL;
  delete token;
  return URMA_SUCCESS;
}
using Pool = kvaf::CpuDdrRegistrationPool;
using Lease = std::shared_ptr<kvaf::CpuDdrRegistration>;
Pool::Key Key() {
  Pool::Key k;
  k.epoch = 10; k.allocation = 20; k.allocation_generation = 30;
  k.context_generation = 40; k.context = 0x1000; k.base = 0x2000; k.bytes = 4096;
  k.candidate = 1; k.eid[0] = 1;
  k.flags = kvaf::CpuDdrRegistration::Flags().value; k.token = 2;
  return k;
}
int main() {
  const auto key = Key();
  {
    Pool pool;
    Lease put, get;
    auto p = std::async(std::launch::async, [&] { return pool.Acquire(key, &put); });
    auto g = std::async(std::launch::async, [&] { return pool.Acquire(key, &get); });
    CHECK(p.get().ok() && g.get().ok());
    CHECK(put == get && allocations == 1 && registrations == 1);
    put.reset();
    CHECK(unregisters == 0 && get->segment()->seg.ubva.va == key.base + 0x100000);
    CHECK(!pool.Close().ok());
    Lease rejected;
    CHECK(!pool.Acquire(key, &rejected).ok());
    get.reset();
    fail_unregister = true;
    CHECK(!pool.Close().ok() && frees == 0);
    fail_unregister = false; fail_free = true;
    CHECK(!pool.Close().ok() && unregisters == 2 && frees == 1);
    fail_free = false;
    CHECK(pool.Close().ok() && unregisters == 2 && frees == 2);
    CHECK(pool.Close().ok());
  }
  {
    Pool pool;
    std::vector<Lease> leases;
    auto acquire = [&](Pool::Key variant) {
      Lease lease;
      CHECK(pool.Acquire(variant, &lease).ok());
      CHECK(std::none_of(leases.begin(), leases.end(), [&](const auto& old) { return old == lease; }));
      leases.push_back(std::move(lease));
    };
    acquire(key);
    auto other = key; ++other.context_generation; acquire(other);
    other = key; ++other.candidate; acquire(other);
    other = key; ++other.eid[0]; acquire(other);
    other = key; ++other.epoch; acquire(other);
    other = key; ++other.allocation_generation; acquire(other);
    other = key; ++other.token; acquire(other);
    other = key; other.flags = 0; Lease invalid;
    CHECK(!pool.Acquire(other, &invalid).ok());
    leases.clear();
    CHECK(pool.Close().ok());
  }
  {
    Pool pool;
    Lease first, second;
    const auto attempts = registrations.load();
    fail_register = true;
    CHECK(!pool.Acquire(key, &first).ok() && first && first->token_id());
    fail_register = false;
    CHECK(!pool.Acquire(key, &second).ok() && second == first);
    CHECK(registrations == attempts + 1);  // No recreation of a failed identity.
    first.reset(); second.reset();
    CHECK(pool.Close().ok());
  }
  std::cout << "CPU DDR shared registration identity, concurrent acquire and close retries passed\n";
}
