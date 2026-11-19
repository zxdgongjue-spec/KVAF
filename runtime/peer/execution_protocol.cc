#include "execution_protocol.h"
#include <cstring>
#include <cerrno>
#include <limits>
#include <netdb.h>
#include <sys/socket.h>
#include <type_traits>
#include <unistd.h>

namespace kvaf::execution {
namespace {
constexpr std::size_t kMaxPacket = 16384;
struct Writer {
  std::vector<std::uint8_t> bytes;
  void U64(std::uint64_t n) {
    for (unsigned i = 0; i < 8; ++i) {
      bytes.push_back(n & 255);
      n >>= 8;
    }
  }
  template <class T>
  void Enum(T n) {
    U64(static_cast<std::uint64_t>(n));
  }
  void Data(const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    if (n) bytes.insert(bytes.end(), b, b + n);
  }
  void Blob(const std::vector<std::uint8_t>& b) {
    U64(b.size());
    Data(b.data(), b.size());
  }
};
struct Reader {
  const std::vector<std::uint8_t>& bytes;
  std::size_t pos{0};
  bool valid{true};
  std::uint64_t U64() {
    std::uint64_t n = 0;
    if (bytes.size() - pos < 8) {
      valid = false;
      return 0;
    }
    for (unsigned i = 0; i < 8; ++i) n |= std::uint64_t(bytes[pos++]) << (8 * i);
    return n;
  }
  template <class T>
  T Enum() {
    auto n = U64();
    using Value = typename std::underlying_type<T>::type;
    if (n > static_cast<std::uint64_t>(std::numeric_limits<Value>::max())) {
      valid = false;
      return T{};
    }
    return static_cast<T>(n);
  }
  void Data(void* p, std::size_t n) {
    if (n > bytes.size() - pos) {
      valid = false;
      return;
    }
    if (n) std::memcpy(p, bytes.data() + pos, n);
    pos += n;
  }
  std::vector<std::uint8_t> Blob() {
    auto n = U64();
    if (n > kMaxPacket || n > bytes.size() - pos) {
      valid = false;
      return {};
    }
    std::vector<std::uint8_t> b(n);
    Data(b.data(), n);
    return b;
  }
  bool Done() const { return valid && pos == bytes.size(); }
};
bool Transfer(int fd, void* data, std::size_t n, bool write) {
  auto* p = static_cast<char*>(data);
  while (n) {
    auto r = write ? send(fd, p, n, MSG_NOSIGNAL) : recv(fd, p, n, 0);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    p += r;
    n -= r;
  }
  return true;
}
}  // namespace
std::vector<std::uint8_t> Encode(const Request& r) {
  Writer w;
  w.U64(kVersion);
  w.Enum(r.op);
  w.U64(r.client);
  w.U64(r.request);
  w.U64(r.region);
  w.U64(r.epoch);
  w.U64(r.session);
  w.Enum(r.mode);
  w.Enum(r.transport);
  w.Enum(r.executor);
  w.Enum(r.algorithm);
  w.U64(r.put);
  w.U64(r.candidate);
  w.U64(r.workers);
  w.U64(r.jettys);
  w.U64(r.request_capacity);
  w.U64(r.token_bytes);
  w.U64(r.cpu_threads);
  w.U64(r.logical_bytes);
  w.U64(r.slot_size);
  w.U64(r.auxiliary_bytes);
  w.Data(r.local_eid.data(), r.local_eid.size());
  w.Blob(r.payload);
  return w.bytes;
}
Status Decode(const std::vector<std::uint8_t>& bytes, Request* out) {
  if (!out || bytes.size() > kMaxPacket) return Status::InvalidArgument("invalid execution record");
  Reader r{bytes};
  Request v;
  if (r.U64() != kVersion) return Status::NotImplemented("execution protocol version mismatch");
  v.op = r.Enum<Op>();
  v.client = r.U64();
  v.request = r.U64();
  v.region = r.U64();
  v.epoch = r.U64();
  v.session = r.U64();
  v.mode = r.Enum<TransportMode>();
  v.transport = r.Enum<Transport>();
  v.executor = r.Enum<Executor>();
  v.algorithm = r.Enum<GetAlgorithm>();
  auto put = r.U64();
  v.put = put == 1;
  auto candidate = r.U64(), workers = r.U64(), jettys = r.U64();
  v.candidate = candidate;
  v.workers = workers;
  v.jettys = jettys;
  auto capacity = r.U64(), token_bytes = r.U64(), cpu_threads = r.U64();
  v.request_capacity = capacity;
  v.token_bytes = token_bytes;
  v.cpu_threads = cpu_threads;
  v.logical_bytes = r.U64();
  v.slot_size = r.U64();
  v.auxiliary_bytes = r.U64();
  r.Data(v.local_eid.data(), v.local_eid.size());
  v.payload = r.Blob();
  if (!r.Done() || capacity > UINT32_MAX || token_bytes > UINT32_MAX || cpu_threads > UINT32_MAX ||
      put > 1 || candidate > UINT32_MAX || workers > UINT32_MAX || jettys > UINT32_MAX ||
      v.op > Op::Fence || v.mode > TransportMode::UrmaAndUbmem || v.transport > Transport::Ubmem ||
      v.executor > Executor::Aicpu || v.algorithm > GetAlgorithm::Scatter)
    return Status::InvalidArgument("malformed execution request");
  *out = std::move(v);
  return Status::OK();
}
std::vector<std::uint8_t> Encode(const Response& r) {
  Writer w;
  w.U64(kVersion);
  w.Enum(r.status.code());
  w.Enum(r.state);
  w.U64(r.capabilities);
  w.U64(r.region);
  w.U64(r.epoch);
  w.U64(r.session);
  w.U64(r.logical_bytes);
  w.U64(r.slot_size);
  auto& b = r.binding;
  w.U64(b.session_id);
  w.U64(b.base);
  w.U64(b.length);
  w.U64(b.token_id);
  w.U64(b.token_value);
  w.U64(b.segment_key_bytes);
  w.Data(b.eid.data(), b.eid.size());
  w.Data(b.segment_key.data(), b.segment_key.size());
  w.Data(b.fabric_handle.data(), b.fabric_handle.size());
  w.U64(b.fabric_mapping_bytes);
  std::vector<std::uint8_t> message(r.status.message().begin(), r.status.message().end());
  w.Blob(message);
  w.Blob(r.payload);
  return w.bytes;
}
Status Decode(const std::vector<std::uint8_t>& bytes, Response* out) {
  if (!out || bytes.size() > kMaxPacket)
    return Status::InvalidArgument("invalid execution response");
  Reader r{bytes};
  Response v;
  if (r.U64() != kVersion) return Status::NotImplemented("execution protocol version mismatch");
  auto code = r.U64();
  v.state = r.Enum<State>();
  v.capabilities = r.U64();
  v.region = r.U64();
  v.epoch = r.U64();
  v.session = r.U64();
  v.logical_bytes = r.U64();
  v.slot_size = r.U64();
  auto& b = v.binding;
  b.session_id = r.U64();
  b.base = r.U64();
  b.length = r.U64();
  auto token = r.U64(), value = r.U64(), keys = r.U64();
  b.token_id = token;
  b.token_value = value;
  b.segment_key_bytes = keys;
  r.Data(b.eid.data(), b.eid.size());
  r.Data(b.segment_key.data(), b.segment_key.size());
  r.Data(b.fabric_handle.data(), b.fabric_handle.size());
  b.fabric_mapping_bytes = r.U64();
  auto message = r.Blob();
  v.payload = r.Blob();
  if (!r.Done() || code > static_cast<unsigned>(StatusCode::kCompletionUnknown) ||
      v.state > State::Closed || token > UINT32_MAX || value > UINT32_MAX ||
      keys > b.segment_key.size())
    return Status::InvalidArgument("malformed execution response");
  v.status =
      Status::Error(static_cast<StatusCode>(code), std::string(message.begin(), message.end()));
  *out = std::move(v);
  return Status::OK();
}
bool Exchange(const std::string& address, const Request& request, Response* response,
              std::string* error) {
  auto fail = [&](const std::string& s) {
    if (error) *error = s;
    return false;
  };
  auto body = Encode(request);
  if (body.size() > kMaxPacket - sizeof(peer::MessageHeader))
    return fail("execution request too large");
  const auto colon = address.rfind(':');
  if (colon == std::string::npos) return fail("expected host:port");
  std::string host = address.substr(0, colon);
  if (host.size() > 1 && host.front() == '[' && host.back() == ']')
    host = host.substr(1, host.size() - 2);
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* list = nullptr;
  if (getaddrinfo(host.c_str(), address.substr(colon + 1).c_str(), &hints, &list) != 0)
    return fail("cannot resolve peer");
  int fd = -1;
  for (auto* a = list; a; a = a->ai_next) {
    fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (fd < 0) continue;
    timeval timeout{30, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
    close(fd);
    fd = -1;
  }
  freeaddrinfo(list);
  if (fd < 0) return fail("cannot connect peer");
  auto h = peer::MakeHeader(static_cast<peer::Command>(kCommand),
                            sizeof(peer::MessageHeader) + body.size());
  bool ok = Transfer(fd, &h, sizeof(h), true) && Transfer(fd, body.data(), body.size(), true) &&
            Transfer(fd, &h, sizeof(h), false);
  if (ok)
    ok = h.magic == peer::kProtocolMagic && h.version == peer::kProtocolVersion &&
         h.command == (kCommand | 0x8000) && h.bytes >= sizeof(h) && h.bytes <= kMaxPacket;
  if (ok) {
    body.resize(h.bytes - sizeof(h));
    ok = Transfer(fd, body.data(), body.size(), false);
  }
  close(fd);
  if (!ok) return fail("execution exchange incomplete; retain request ID for recovery");
  auto decoded = Decode(body, response);
  return decoded.ok() ? true : fail(decoded.message());
}
}  // namespace kvaf::execution
