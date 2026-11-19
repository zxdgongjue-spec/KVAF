#pragma once
#include <functional>
#include <memory>
#include <mutex>
#include <map>
#include <shared_mutex>
#include <unordered_map>
#include "execution_protocol.h"

namespace kvaf::execution {
class Allocation {
 public:
  virtual ~Allocation() = default;
  virtual Status Close() = 0;
  peer::PeerRegion descriptor;
};
class Session {
 public:
  virtual ~Session() = default;
  virtual Status Open(const Request&, Allocation&, peer::PeerRegion*) = 0;
  virtual Status Call(const std::vector<std::uint8_t>&, std::vector<std::uint8_t>*) = 0;
  virtual Status Close() = 0;
};
// Region ownership is separate from session ownership. Slow creation/teardown
// holds only the affected record lock. IDs survive dropped replies until close.
class CacheRegionService {
 public:
  using Allocate = std::function<Status(const Request&, std::unique_ptr<Allocation>*)>;
  using Create = std::function<std::unique_ptr<Session>(const Request&)>;
  CacheRegionService(std::uint64_t capabilities, Allocate, Create);
  ~CacheRegionService();
  Response Handle(const Request&);
  Status Shutdown();

 private:
  struct Record {
    std::mutex mutex;
    Request request;
    Response response;
    std::unique_ptr<Allocation> allocation;
    std::unique_ptr<Session> session;
    std::shared_ptr<Record> parent;
    std::size_t children{0};
    std::map<std::uint64_t, std::pair<std::vector<std::uint8_t>, Response>> calls;
  };
  std::shared_ptr<Record> Lookup(std::uint64_t);
  Response CreateRecord(const Request&);
  Status CloseRecord(const std::shared_ptr<Record>&);
  std::uint64_t capabilities_, epoch_, next_id_{1};
  Allocate allocate_;
  Create create_;
  std::shared_mutex lifetime_;
  std::mutex index_;
  bool closing_{false};
  std::unordered_map<std::uint64_t, std::shared_ptr<Record>> records_;
  // client + create_request_id identifies exactly one immutable request.
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::shared_ptr<Record>> requests_;
};
}  // namespace kvaf::execution
