#include "control.h"
#include "cpu_urma/cpu_urma_transport.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {
using namespace kvaf::aicpu_test;
using kvaf::CpuUrmaTransport;
using Clock = std::chrono::steady_clock;
struct Options {
  Config config;
  kvaf::CpuUrmaTransportOptions transport;
};
Options Parse(int argc, char** argv) {
  Options o;
  o.transport.remote_sgl_size = 16;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    Require(i + 1 < argc, "missing value for " + flag);
    const char* value = argv[++i];
    if (flag == "--peer") { o.transport.peer_address = value; continue; }
    if (flag == "--ub-device") { o.transport.ub_device = value; continue; }
    const auto n = Number(value);
    if (flag == "--rows") o.config.rows = n;
    else if (flag == "--row-bytes") o.config.row_bytes = n;
    else if (flag == "--warmup") o.config.warmup = n;
    else if (flag == "--iterations") o.config.iterations = n;
    else if (flag == "--timeout-ms") o.config.timeout_ms = n;
    else {
      Require(n <= UINT32_MAX, "option exceeds uint32");
      if (flag == "--eid-index") { o.transport.has_eid_index = true; o.transport.eid_index = n; }
      else if (flag == "--remote-sgl-size") o.transport.remote_sgl_size = n;
      else if (flag == "--jettys") o.transport.jetty_count = n;
      else if (flag == "--prepare-threads") o.transport.prepare_thread_count = n;
      else if (flag == "--wqes-per-doorbell") o.transport.wqes_per_doorbell = n;
      else throw std::runtime_error("unknown option " + flag);
    }
  }
  Layout layout{};
  Require(MakeLayout(o.config, &layout) && !o.transport.peer_address.empty() &&
          o.transport.remote_sgl_size > 0 && o.transport.jetty_count > 0 &&
          o.transport.jetty_count <= 16 && o.transport.prepare_thread_count > 0 &&
          o.transport.prepare_thread_count <= 64, "invalid benchmark configuration");
  return o;
}
void Check(const kvaf::Status& status) { Require(status.ok(), status.message()); }
kvaf::peer::PeerRegion Configure(const Options& options) {
  Socket socket(Connect(options.transport.peer_address));
  SocketTimeout(socket.get(), 120000);
  ConfigureRequest request{};
  request.header = kvaf::peer::MakeHeader(kConfigure, sizeof(request));
  request.config = options.config;
  Send(socket.get(), &request, sizeof(request));
  kvaf::peer::AllocateHbmRegionResponse reply{};
  Receive(socket.get(), &reply, sizeof(reply));
  Require(kvaf::peer::IsValidResponseHeader(reply.header, kConfigure, sizeof(reply)) &&
          reply.result == 0 && reply.segment_key_bytes <= kvaf::peer::kSegKeyBytes &&
          reply.jetty_key_bytes <= kvaf::peer::kQpKeyBytes, "NPU configure/export failed");
  kvaf::peer::PeerRegion region{};
  region.session_id = reply.session_id; region.base = reply.hbm_base;
  region.length = reply.hbm_bytes; region.device_memory = true;
  region.token_id = reply.token_id; region.token_value = reply.token_value;
  region.segment_key_bytes = reply.segment_key_bytes;
  region.jetty_key_bytes = reply.jetty_key_bytes;
  std::memcpy(region.eid.data(), reply.peer_eid, sizeof(reply.peer_eid));
  std::memcpy(region.segment_key.data(), reply.segment_key, sizeof(reply.segment_key));
  std::memcpy(region.jetty_key.data(), reply.jetty_key, sizeof(reply.jetty_key));
  return region;
}
void Run(Options options) {
  Layout layout{}; Require(MakeLayout(options.config, &layout), "invalid layout");
  // One page-aligned registered allocation contains source rows and both
  // host-side control buffers. It outlives transport destruction on all paths.
  void* memory = nullptr;
  const auto bytes = (layout.output_bytes + 4096 + 4095) & ~std::uint64_t{4095};
  Require(posix_memalign(&memory, 4096, bytes) == 0, "source allocation failed");
  std::unique_ptr<void, decltype(&std::free)> allocation(memory, std::free);
  std::memset(memory, 0, bytes);
  auto* input = static_cast<std::uint8_t*>(memory) + 4096;
  auto* request = static_cast<Control*>(memory);
  auto* done = reinterpret_cast<Control*>(static_cast<std::uint8_t*>(memory) + 256);
  std::vector<CpuUrmaTransport::HostSge> sources(options.config.rows);
  std::vector<std::uint64_t> offsets(options.config.rows);
  for (std::uint64_t row = 0; row < options.config.rows; ++row) {
    auto* data = input + row * layout.stride;
    for (std::uint64_t b = 8; b < options.config.row_bytes; ++b) data[b] = Pattern(row, b);
    sources[row] = {data, static_cast<std::size_t>(options.config.row_bytes)};
    offsets[row] = kStagingOffset + row * options.config.row_bytes;
  }
  options.transport.external_region = Configure(options);
  options.transport.has_external_region = true;
  options.transport.region_bytes = layout.region_bytes;
  CpuUrmaTransport transport;
  Check(transport.Initialize(options.transport));
  // ValidateLink writes at offset zero. Restore the request record afterward.
  Check(transport.ValidateLink());
  Check(transport.RegisterDirectHostMemory(memory, bytes));
  CpuUrmaTransport::HostSge request_sge{request, sizeof(*request)}, done_sge{done, sizeof(*done)};
  Check(transport.WriteSglBatch(kRequestOffset, &request_sge, 1));
  Check(transport.WriteSglBatch(kDoneOffset, &done_sge, 1));
  Socket control(Connect(options.transport.peer_address));
  SocketTimeout(control.get(), options.config.timeout_ms + 120000);
  auto start = kvaf::peer::MakeHeader(kStart, sizeof(kvaf::peer::MessageHeader));
  Send(control.get(), &start, sizeof(start));
  Ack ack{}; Receive(control.get(), &ack, sizeof(ack));
  Require(ack.magic == kMagic && ack.status == 0, "NPU kernel launch failed");
  long double write_ns = 0, prepare_ns = 0, post_ns = 0, poll_ns = 0;
  std::uint64_t polls = 0, wr_count = 0;
  bool ok = true;
  try {
    for (std::uint64_t seq = 1; seq <= layout.rounds; ++seq) {
      const auto deadline = Clock::now() + std::chrono::milliseconds(options.config.timeout_ms);
      do {
        Check(transport.ReadSglBatch(kRequestOffset, &request_sge, 1));
        ++polls;
        Require(request->sequence <= seq, "unexpected future AICPU request");
        Require(Clock::now() < deadline, "timed out waiting for AICPU request");
      } while (request->sequence != seq);
      for (std::uint64_t row = 0; row < options.config.rows; ++row) {
        const auto tag = RowTag(seq, row);
        std::memcpy(sources[row].data, &tag, sizeof(tag));
      }
      Check(transport.WriteBatch(offsets.data(), sources.data(), sources.size()));
      const auto timing = transport.last_batch_timing();
      if (seq > options.config.warmup) {
        write_ns += timing.total_ns; prepare_ns += timing.prepare_ns;
        post_ns += timing.post_ns; poll_ns += timing.poll_ns; wr_count += timing.wr_count;
      }
      done->sequence = seq;
      Check(transport.WriteSglBatch(kDoneOffset, &done_sge, 1));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "CPU transfer failed: %s\n", e.what()); ok = false;
  }
  // The server must not unregister/free its HBM until the CPU has stopped all
  // URMA operations, including error/timeout paths.
  const auto shutdown = transport.Shutdown();
  Require(shutdown.ok(), "URMA shutdown failed; do not release NPU target");
  Ack drained{}; drained.status = ok ? 0 : 1;
  Send(control.get(), &drained, sizeof(drained));
  Report report{}; Receive(control.get(), &report, sizeof(report));
  Require(ok && report.magic == kMagic && report.status == 0, "benchmark or NPU validation failed");
  const double n = static_cast<double>(options.config.iterations);
  std::printf("engine,rows,row_bytes,iterations,e2e_GBps,mean_ns,p50_ns,p99_ns,wait_ns,scatter_ns,"
              "cpu_write_ns,cpu_prepare_ns,cpu_post_ns,cpu_poll_ns,mean_wrs,control_reads\n");
  std::printf("aicpu_memcpy,%llu,%llu,%llu,%.6f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%llu\n",
              (unsigned long long)options.config.rows, (unsigned long long)options.config.row_bytes,
              (unsigned long long)options.config.iterations, report.gbps, report.mean_ns,
              report.p50_ns, report.p99_ns, report.wait_ns, report.scatter_ns,
              (double)write_ns / n, (double)prepare_ns / n, (double)post_ns / n,
              (double)poll_ns / n, (double)wr_count / n, (unsigned long long)polls);
}
}  // namespace
int main(int argc, char** argv) {
  try { Run(Parse(argc, argv)); return 0; }
  catch (const std::exception& e) {
    std::fprintf(stderr, "AICPU_GET_CPU: %s\n", e.what()); return 1;
  }
}
