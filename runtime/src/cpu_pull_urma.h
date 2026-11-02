#pragma once
#include <urma_api.h>
#include "offload_service.h"
namespace kvaf::offload {
class UrmaDestination;
// One pinned local-write registration per shared DDR pool/context. The pool
// owns it; backends borrow it while their pool leases keep the memory alive.
std::shared_ptr<UrmaDestination> CreateUrmaDestination(urma_context_t*, void*, std::uint64_t, std::uint32_t);
Status CloseUrmaDestination(const std::shared_ptr<UrmaDestination>&);
std::unique_ptr<Backend> CreateUrmaBackend(urma_context_t* context);
// The lease owns the original DDR allocation/context. This backend registers
// that same memory for local writes but never frees or reallocates the pool.
std::unique_ptr<Backend> CreateBorrowedUrmaBackend(urma_context_t*, void* memory,
    std::uint64_t bytes, std::uint32_t token, std::shared_ptr<void> lease,
    std::shared_ptr<UrmaDestination> destination = {});
}
