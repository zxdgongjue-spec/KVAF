#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "status.h"

namespace kvaf {

// Opaque lifetime handle for a locally registered NPU HBM allocation.
// Transport implementations keep their MR and backing-storage owner private.
class HbmRegion {
 public:
  virtual ~HbmRegion() = default;

  // Validate an operation range and pin the registration until the returned
  // lease is released. Async transports keep the lease through CQ completion.
  virtual Status Pin(const void* address, std::size_t bytes,
                     std::shared_ptr<void>* lease) = 0;
  virtual Status Close() = 0;
  virtual bool closed() const = 0;
  virtual std::uint32_t device_id() const = 0;
  virtual std::uintptr_t base() const = 0;
  virtual std::size_t bytes() const = 0;
  virtual const void* transport_identity() const = 0;
};

}  // namespace kvaf
