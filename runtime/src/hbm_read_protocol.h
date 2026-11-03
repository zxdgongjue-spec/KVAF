#pragma once
#include "hbm_read.h"
#include "offload_protocol.h"
#include <mutex>

namespace kvaf::hbm_read {
constexpr std::uint32_t kMagic = 0x4b564852; // KVHR, independent of KVCP/KVRP
constexpr std::uint16_t kVersion = 6; // Receiver listens; Ready publishes an irrevocable batch.
enum class Command : std::uint16_t { Discover=1, Open, Channel, Confirm, Complete=6, Cancel, Close, Memory=10, Retire };
enum class State : std::uint32_t { Registered, Connected, Ready, Done };
using Span = offload::Span;
struct Request {
  peer::MessageHeader header{};
  std::uint32_t token{}, candidate{}, channel{}, count{};
  std::uint64_t source_id{}, epoch{}, client{}, sequence{}, mr_id{1};
  offload::Endpoint endpoint{};
};
struct Response {
  peer::MessageHeader header{};
  std::uint32_t code{}, candidates{}, state{}, reserved{};
  std::uint64_t source_id{}, epoch{}, client{}, sequence{}, checksum{}, mr_id{1}, mr_count{};
  offload::Endpoint endpoint{};
  char detail[256]{};
};
static_assert(sizeof(Request) == 448 && sizeof(Response) == 720, "KVHR wire ABI changed");
inline Request Make(Command c, const HbmReadTicket& t, std::uint64_t client) {
  Request r{};
  r.header = {kMagic, kVersion, static_cast<std::uint16_t>(c), sizeof(Request)};
  r.token=t.token; r.source_id=t.source_id; r.epoch=t.epoch; r.client=client;
  return r;
}
// One reader owns this duplex socket. It never dials the producer. A failed
// exchange invalidates only TCP; the MR/engine/batch owner must resolve it.
class ControlChannel {
 public:
  explicit ControlChannel(int fd) : fd_(fd) {}
  ~ControlChannel();
  Status Exchange(Request, Response*);
  Status Replace(int fd); // Same source/receiver epoch only, checked by listener.
  bool connected();
 private:
  void Close();
  std::mutex mutex_;
  int fd_{-1};
};
// Initial P->D messages. The following mr_count uint64 lengths describe all MRs.
// Join hands the socket to D's RPC client and P's request responder. Leave uses
// a short control connection so it cannot race the responder's recv().
enum class Admission : std::uint16_t { Join=100, Leave=101, Result=102, Resume=103 };
struct Hello {
  peer::MessageHeader header{};
  std::uint64_t receiver_id{}, receiver_epoch{}, source_id{}, source_epoch{};
  std::uint32_t token{}, mr_count{};
};
struct HelloResult {
  peer::MessageHeader header{};
  std::uint32_t code{}, reserved{};
  char detail[256]{};
};
static_assert(sizeof(Hello)==56 && sizeof(HelloResult)==276, "KVHR admission ABI changed");
Status SendHello(int fd, Admission, const HbmReceiverTicket&, const HbmReadTicket&);
Status ReadHello(int fd, Hello*, HbmReadTicket*);
bool SendResult(int fd, const Status&);
Status ReadResult(int fd, const peer::MessageHeader&);
Status Leave(const HbmReceiverTicket&, const HbmReadTicket&);
int Connect(const std::string&);
Status Listen(const std::string&, int* fd, std::uint16_t* port);
}
