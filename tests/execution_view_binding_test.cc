#include "execution_view.h"
#include "direct_hccp/remote_memory_binding.h"
#include <iostream>
#include <map>
#include <stdexcept>

using namespace kvaf;
namespace hccp = kvaf::direct_hccp;
#define CHECK(c) do { if (!(c)) throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #c); } while (false)
namespace {
std::map<void*, void*> handles;
std::uintptr_t next_handle = 100;
int imports{}, unimports{};
bool import_fails{}, close_fails{};
int Import(void* context, hccp::MrImportInfo*, void** out) {
  ++imports;
  *out = reinterpret_cast<void*>(++next_handle);
  handles[*out] = context;
  return import_fails ? -1 : 0; // failure may still create a handle
}
int Unimport(void* context, void* handle) {
  ++unimports;
  CHECK(handles.at(handle) == context);
  if (close_fails) return -1;
  handles.erase(handle);
  return 0;
}
void ViewLifetime() {
  struct Call : ExecutionState {} first, second;
  ExecutionViewState resource;
  RemoteRegionDesc region{0x1000, 0x100, 64, DeviceKind::kCpu};
  resource.Bind(&first);
  CHECK(!resource.View(ExecutionDirection::Put).ready());
  resource.Initialize(region, 12, 34);
  auto put = resource.View(ExecutionDirection::Put);
  CHECK(put.Validate(&first, ExecutionDirection::Put).ok());
  CHECK(!put.Validate(&first, ExecutionDirection::Get).ok());
  CHECK(!put.Validate(&second, ExecutionDirection::Put).ok());
  CHECK(put.region().region_id == 12 && put.region().epoch == 34);
  Address translated{};
  CHECK(put.region().Translate(0x1040, 64, 0x9000, &translated).ok() && translated == 0x9040);
  CHECK(!put.region().Translate(0x10F0, 64, 0x9000, &translated).ok());
  // Independent direction preparation must not invalidate a pending Put.
  resource.Prepare(ExecutionDirection::Get);
  CHECK(put.ready());
  resource.Prepare(ExecutionDirection::Put);
  CHECK(!put.ready());
  auto prepared = resource.View(ExecutionDirection::Put);
  resource.Bind(&second);
  CHECK(!prepared.ready());
  resource.Bind(&first);
  CHECK(!prepared.ready());
  auto old = resource.View(ExecutionDirection::Put);
  resource.Close();
  CHECK(!old.ready());
  resource.Initialize(region, 12, 35); // same address, new allocation epoch
  CHECK(!old.ready());
}
void BindingIdentity() {
  using Binding = hccp::HccpRemoteMemoryBinding;
  UrmaBindingCache<Binding> cache(MemoryDirection::Get), put(MemoryDirection::Put);
  UrmaRemoteMemoryDesc remote;
  remote.epoch = 10; remote.region_id = 20; remote.session_id = 30;
  remote.registered_address = 0x1000; remote.registered_bytes = 0x1000;
  remote.token_id = 0x12300; remote.token_value = 2; remote.key_bytes = 1;
  remote.eid[0] = 1; remote.key[0] = 7;
  CHECK(remote.valid() && remote.wqe_token_id() == 0x123);
  auto* context = reinterpret_cast<void*>(1);
  auto key = UrmaBindingKey::Make(MemoryDirection::Get, context, NewBindingGeneration(), remote, 7);
  auto create = [&](Binding& binding) { return binding.Import(context, Import, Unimport, remote); };
  std::shared_ptr<Binding> first, second;
  CHECK(cache.Acquire(key, create, &first).ok());
  CHECK(cache.Acquire(key, create, &second).ok() && first == second && imports == 1);
  CHECK(!cache.Acquire(key, create, &second).ok());
  CHECK(cache.Release(&first).ok() && !first && unimports == 0);
  std::shared_ptr<Binding> other;
  CHECK(!put.Acquire(key, create, &other).ok() && !other);
  // Each identity component must force a distinct import, including pointer reuse.
  for (unsigned field = 0; field != 10; ++field) {
    auto different = key;
    switch (field) {
      case 0: ++different.context; break;
      case 1: ++different.context_generation; break;
      case 2: ++different.epoch; break;
      case 3: ++different.region; break;
      case 4: ++different.session; break;
      case 5: ++different.token; break;
      case 6: ++different.permissions; break;
      case 7: ++different.base; break;
      case 8: ++different.eid[0]; break;
      case 9: ++different.key[0]; break;
    }
    CHECK(cache.Acquire(different, create, &other).ok() && other != second);
    CHECK(cache.Release(&other).ok());
  }
  close_fails = true;
  CHECK(!cache.Release(&second).ok() && second && handles.size() == 1);
  const auto count = imports;
  CHECK(!cache.Acquire(key, create, &other).ok() && other == second && imports == count);
  CHECK(cache.Release(&other).ok());
  close_fails = false;
  CHECK(cache.Release(&second).ok() && handles.empty());
  import_fails = true;
  CHECK(!cache.Acquire(key, create, &first).ok() && first && handles.size() == 1);
  CHECK(cache.Release(&first).ok() && handles.empty());
  import_fails = false;
  CHECK(!cache.Acquire(key, [&](Binding& binding) -> Status {
    CHECK(create(binding).ok()); throw std::runtime_error("after import");
  }, &first).ok() && first);
  CHECK(cache.Release(&first).ok() && handles.empty());
  CHECK(cache.Acquire(key, create, &first).ok());
  CHECK(cache.Release(&first).ok() && handles.empty());
}
}
int main() { ViewLifetime(); BindingIdentity(); std::cout << "View lifetime and remote binding identity passed\n"; }
