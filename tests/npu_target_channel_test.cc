#include "npu_target_channel.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace hccp = kvaf::direct_hccp;
using kvaf::npu_service_detail::TargetChannel;

static_assert((hccp::MemRegFlags(false) & (1U << 11U)) == 0U,
              "pinned HBM registration must keep non-pin disabled");
static_assert((hccp::MemRegFlags(true) & (1U << 11U)) != 0U,
              "non-pin registration flag ABI changed");

void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

enum class Kind { kCq, kJetty, kPeer };
enum class Failure { kNone, kError, kErrorWithHandle, kNullHandle, kEmptyKey, kOversizedKey };

struct FakeProvider {
  struct Handle {
    FakeProvider* owner;
    Kind kind;
    std::uint32_t id;
    bool live{true};
    Handle* dependency{nullptr};
  };
  struct Event {
    Kind kind;
    std::uint32_t id;
  };

  FakeProvider() {
    api.cq_create = CreateCq;
    api.cq_destroy = DestroyCq;
    api.qp_create = CreateJetty;
    api.qp_destroy = DestroyJetty;
    api.qp_unimport = UnimportPeer;
  }

  Handle* New(Kind kind, Handle* dependency = nullptr) {
    handles.push_back(std::make_unique<Handle>(Handle{
        this, kind, static_cast<std::uint32_t>(handles.size() + 1), true, dependency}));
    return handles.back().get();
  }

  std::size_t LiveCount() const {
    return std::count_if(handles.begin(), handles.end(),
                         [](const auto& handle) { return handle->live; });
  }

  Handle* Find(std::uint32_t id) { return handles.at(id - 1).get(); }

  // Models the provider writing an imported handle even when import later
  // fails validation. Ownership must cover either import result.
  Handle* SetImportedPeer(TargetChannel& channel) {
    Handle* peer = New(Kind::kPeer, Find(channel.jetty_info().ub.id));
    *channel.peer_jetty_output() = peer;
    return peer;
  }

  static int CreateCq(void* context, hccp::CqInfo* info, void** output) {
    auto& fake = *static_cast<FakeProvider*>(context);
    if (fake.cq_failure == Failure::kError) return -1;
    if (fake.cq_failure == Failure::kNullHandle) return 0;
    Handle* cq = fake.New(Kind::kCq);
    *output = cq;
    info->out.id = cq->id;
    return fake.cq_failure == Failure::kErrorWithHandle ? -1 : 0;
  }

  static int CreateJetty(void* context, hccp::QpCreateAttr* attr,
                         hccp::QpCreateInfo* info, void** output) {
    auto& fake = *static_cast<FakeProvider*>(context);
    fake.jetty_attrs.push_back(*attr);
    if (fake.qp_failure == Failure::kError) return -1;
    if (fake.qp_failure == Failure::kNullHandle) return 0;
    Handle* jetty = fake.New(Kind::kJetty, static_cast<Handle*>(attr->scq_handle));
    *output = jetty;
    info->ub.id = jetty->id;
    info->key.size = sizeof(jetty->id);
    std::memcpy(info->key.value, &jetty->id, sizeof(jetty->id));
    if (fake.qp_failure == Failure::kEmptyKey) info->key.size = 0;
    if (fake.qp_failure == Failure::kOversizedKey)
      info->key.size = sizeof(info->key.value) + 1;
    return fake.qp_failure == Failure::kErrorWithHandle ? -1 : 0;
  }

  static int Release(void* opaque, Kind expected) {
    auto* handle = static_cast<Handle*>(opaque);
    auto& fake = *handle->owner;
    // Record violations without throwing from the channel destructor.
    if (!handle->live || handle->kind != expected) ++fake.release_errors;
    for (const auto& other : fake.handles) {
      if (other->live && other->dependency == handle) ++fake.release_errors;
    }
    handle->live = false;
    fake.releases.push_back({handle->kind, handle->id});
    return 0;
  }

  static int DestroyCq(void* context, void* handle) {
    if (context != static_cast<Handle*>(handle)->owner)
      ++static_cast<Handle*>(handle)->owner->release_errors;
    return Release(handle, Kind::kCq);
  }
  static int DestroyJetty(void* handle) { return Release(handle, Kind::kJetty); }
  static int UnimportPeer(void* context, void* handle) {
    if (context != static_cast<Handle*>(handle)->owner)
      ++static_cast<Handle*>(handle)->owner->release_errors;
    return Release(handle, Kind::kPeer);
  }

  hccp::DirectHccpApi api{};
  int token{0};
  Failure cq_failure{Failure::kNone};
  Failure qp_failure{Failure::kNone};
  std::vector<std::unique_ptr<Handle>> handles;
  std::vector<hccp::QpCreateAttr> jetty_attrs;
  std::vector<Event> releases;
  unsigned release_errors{0};
};

kvaf::Status Initialize(TargetChannel& channel, FakeProvider& fake) {
  return channel.Initialize(&fake.token, 0xABCF, 1024, 1024, 16384);
}

void IndependentChannels() {
  FakeProvider fake;
  {
    TargetChannel first(fake.api, &fake);
    Check(Initialize(first, fake).ok(), "first channel initialization failed");
    const auto first_key = first.jetty_info().key;
    const auto first_cq = first.cq_info().out.id;
    fake.SetImportedPeer(first);
    {
      TargetChannel second(fake.api, &fake);
      Check(Initialize(second, fake).ok(), "second channel initialization failed");
      fake.SetImportedPeer(second);
      Check(first.jetty_info().ub.id != second.jetty_info().ub.id,
            "channels must have distinct target Jettys");
      Check(first_cq != second.cq_info().out.id,
            "channels must have distinct completion queues");
      Check(first_key.size == second.jetty_info().key.size &&
                std::memcmp(first_key.value, second.jetty_info().key.value,
                            first_key.size) != 0,
            "channels must export distinct target keys");
      for (const auto& attr : fake.jetty_attrs) {
        Check(attr.scq_handle == attr.rcq_handle && attr.scq_handle == attr.srq_handle,
              "one channel's queues must use its own CQ");
        Check(attr.ub.token_id_handle == &fake.token && attr.ub.token_value == 0xABCF,
              "channels must share the region token");
        Check(attr.sq_depth == 1024 && attr.rq_depth == 1024,
              "target queue depths were lost");
      }
      Check(second.cq_info().in.depth == 16384, "completion depth was lost");
      Check(fake.LiveCount() == 6, "two connected channels must retain six resources");
      Check(!Initialize(second, fake).ok(), "reinitializing a live channel must fail");
      Check(fake.LiveCount() == 6, "reinitialization damaged live resources");
    }
    Check(fake.LiveCount() == 3, "destroying one channel damaged the other");
    Check(fake.Find(first_cq)->live && fake.Find(first.jetty_info().ub.id)->live,
          "first channel resources must survive second channel destruction");
    Check(fake.releases.size() == 3 && fake.releases[0].kind == Kind::kPeer &&
              fake.releases[1].kind == Kind::kJetty && fake.releases[2].kind == Kind::kCq,
          "channel teardown must unimport peer before destroying Jetty then CQ");
  }
  Check(fake.LiveCount() == 0 && fake.release_errors == 0,
        "channel teardown leaked, double-released, or released dependencies too early");
}

void CreationFailuresAndRetry() {
  const std::vector<std::pair<Failure, Failure>> failures{
      {Failure::kError, Failure::kNone},
      {Failure::kErrorWithHandle, Failure::kNone},
      {Failure::kNullHandle, Failure::kNone},
      {Failure::kNone, Failure::kError},
      {Failure::kNone, Failure::kErrorWithHandle},
      {Failure::kNone, Failure::kNullHandle},
      {Failure::kNone, Failure::kEmptyKey},
      {Failure::kNone, Failure::kOversizedKey}};
  for (const auto& failure : failures) {
    FakeProvider fake;
    {
      TargetChannel existing(fake.api, &fake);
      Check(Initialize(existing, fake).ok(), "existing channel initialization failed");
      fake.SetImportedPeer(existing);
      const auto existing_id = existing.jetty_info().ub.id;
      {
        TargetChannel candidate(fake.api, &fake);
        fake.cq_failure = failure.first;
        fake.qp_failure = failure.second;
        Check(!Initialize(candidate, fake).ok(), "provider failure was accepted");
        Check(fake.LiveCount() == 3 && fake.Find(existing_id)->live,
              "failed channel leaked resources or damaged the existing channel");
        Check(fake.release_errors == 0, "partial channel released resources out of order");
        fake.cq_failure = Failure::kNone;
        fake.qp_failure = Failure::kNone;
        Check(Initialize(candidate, fake).ok(), "retry after creation failure failed");
        Check(candidate.jetty_info().ub.id != existing_id,
              "retry reused another live channel's target");
        Check(fake.LiveCount() == 5, "retry did not retain exactly one CQ and Jetty");
      }
      Check(fake.LiveCount() == 3, "retry cleanup damaged existing resources");
    }
    Check(fake.LiveCount() == 0 && fake.release_errors == 0,
          "failure/retry cleanup leaked or released resources incorrectly");
  }
}

void PartialImportFailure() {
  FakeProvider fake;
  {
    TargetChannel candidate(fake.api, &fake);
    Check(Initialize(candidate, fake).ok(), "partial import setup failed");
    fake.SetImportedPeer(candidate);
    // The caller abandons this channel after ImportPeerJetty returns false.
  }
  Check(fake.LiveCount() == 0 && fake.release_errors == 0 && fake.releases.size() == 3,
        "partially imported peer handle must be released with its failed channel");
  Check(fake.releases[0].kind == Kind::kPeer && fake.releases[1].kind == Kind::kJetty &&
            fake.releases[2].kind == Kind::kCq,
        "partial import cleanup must release peer before target queues");
}
}  // namespace

int main() {
  try {
    IndependentChannels();
    CreationFailuresAndRetry();
    PartialImportFailure();
    std::cout << "NPU target channel ownership tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
