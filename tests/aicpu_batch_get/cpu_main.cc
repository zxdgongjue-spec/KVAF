#include "control.h"
#include "pipeline.h"
#include "output_readback.h"
#include "cpu_urma/cpu_urma_link_selection.h"
#include "cpu_urma/cpu_urma_transport.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {
using namespace kvaf::aicpu_test;
using kvaf::CpuUrmaTransport;
using Clock = std::chrono::steady_clock;
void Check(const kvaf::Status& s) { Require(s.ok(), s.message()); }
double Ns(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::nano>(b - a).count();
}
struct Options {
  Config config;
  kvaf::CpuUrmaTransportOptions transport;
  std::uint32_t npu_candidate{UINT32_MAX};
};
Options Parse(int argc, char** argv) {
  Options o; o.transport.jetty_count = kJettyCount;
  o.transport.remote_sgl_size = 16; o.transport.wqes_per_doorbell = 32;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    Require(i + 1 < argc, "missing value for " + flag); const char* value = argv[++i];
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
      if (flag == "--device-id") { Require(n <= INT32_MAX, "invalid NPU device"); o.transport.remote_device_id = n; }
      else if (flag == "--npu-candidate") { Require(n < UINT32_MAX, "invalid candidate"); o.npu_candidate = n; }
      else if (flag == "--eid-index") { o.transport.has_eid_index = true; o.transport.eid_index = n; }
      else throw std::runtime_error("unknown option " + flag);
    }
  }
  Layout layout{};
  Require(MakeLayout(o.config, &layout) && !o.transport.peer_address.empty(), "invalid benchmark configuration");
  return o;
}
struct Work { std::uint32_t worker; std::uint64_t chunk; RowRange rows; };
void SubmitAndWait(CpuUrmaTransport::PreparedWriteSession& session, std::uint64_t id,
                   Clock::time_point deadline) {
  bool submitted = false;
  Check(session.Submit(id, &submitted)); Require(submitted, "drained queue has insufficient credits");
  std::uint64_t polls = 0;
  for (;;) {
    if (++polls % 32 == 0) Require(Clock::now() < deadline, "producer timeout");
    CpuUrmaTransport::PreparedWriteCompletion completed{}; std::size_t count = 0;
    Check(session.Poll(&completed, 1, &count));
    if (count) { Require(count == 1 && completed.id == id, "unexpected completion"); return; }
  }
}
kvaf::peer::PeerRegion Configure(Options& options, std::uint32_t npu_candidate) {
  Socket socket(Connect(options.transport.peer_address));
  SocketTimeout(socket.get(), 120000);
  ConfigureRequest request{};
  request.header = kvaf::peer::MakeHeader(kConfigure, sizeof(request));
  request.device_id = options.transport.remote_device_id;
  request.npu_candidate = npu_candidate;
  request.config = options.config;
  Send(socket.get(), &request, sizeof(request));
  kvaf::peer::AllocateHbmRegionResponse reply{};
  Receive(socket.get(), &reply, sizeof(reply));
  Require(kvaf::peer::IsValidResponseHeader(reply.header, kConfigure, sizeof(reply)) &&
          reply.result == 0 && reply.segment_key_bytes <= kvaf::peer::kSegKeyBytes &&
          reply.jetty_key_bytes <= kvaf::peer::kQpKeyBytes &&
          reply.reserved > 0 && reply.reserved <= kMaxAicpuBlocks,
          "NPU configure/export failed; check device-id and update both test binaries");
  options.config.aicpu_blocks = reply.reserved;
  kvaf::peer::PeerRegion region{};
  region.session_id = reply.session_id; region.base = reply.hbm_base;
  region.length = reply.hbm_bytes; region.device_memory = true;
  region.token_id = reply.token_id; region.token_value = reply.token_value;
  region.segment_key_bytes = reply.segment_key_bytes;
  region.jetty_key_bytes = reply.jetty_key_bytes;
  std::memcpy(region.eid.data(), reply.peer_eid, sizeof(reply.peer_eid));
  std::memcpy(region.route_eid.data(), reply.route_eid, sizeof(reply.route_eid));
  std::memcpy(region.segment_key.data(), reply.segment_key, sizeof(reply.segment_key));
  std::memcpy(region.jetty_key.data(), reply.jetty_key, sizeof(reply.jetty_key));
  return region;
}
void Run(Options options) {
  kvaf::CpuUrmaDiscovery discovery;
  Check(discovery.Initialize(options.transport.ub_device, options.transport.has_eid_index, options.transport.eid_index));
  // Registered source memory must outlive the transport on every exit path.
  std::unique_ptr<void, decltype(&std::free)> allocation(nullptr, std::free);
  std::unique_ptr<CpuUrmaTransport> selected_transport;
  kvaf::peer::PeerLink peer(options.transport.peer_address);
  Check(kvaf::SelectCpuUrmaDeviceLink(peer, options.transport.remote_device_id, discovery.candidates(),
      [&](const kvaf::CpuUrmaCandidate& cpu, std::uint32_t npu_index) -> kvaf::CpuUrmaLinkAttempt {
    auto config = kvaf::BindCpuUrmaEndpoint(options.transport, cpu, options.transport.remote_device_id, npu_index);
    // Use a fresh transport per EID pair to avoid stale selection state.
    auto candidate = std::make_unique<CpuUrmaTransport>();
    kvaf::Status attempt;
    try {
      config.external_region = Configure(options, npu_index);
      config.has_external_region = true;
      config.region_bytes = config.external_region.length;
      attempt = candidate->Initialize(config);
    } catch (const std::exception& e) {
      attempt = kvaf::Status::Internal(e.what());
    }
    // Retry setup only; a failed data probe may have live WRs.
    if (attempt.ok()) {
      attempt = candidate->ValidateLink();
      if (attempt.ok()) selected_transport = std::move(candidate);
      return {attempt, false};
    }
    const auto stopped = candidate->Shutdown();
    if (!stopped.ok()) return {stopped, false};
    candidate.reset();
    Socket socket(Connect(options.transport.peer_address));
    SocketTimeout(socket.get(), 120000);
    const auto header = kvaf::peer::MakeHeader(kResetSetup, sizeof(kvaf::peer::MessageHeader));
    Send(socket.get(), &header, sizeof(header));
    Ack ack{}; Receive(socket.get(), &ack, sizeof(ack));
    if (ack.magic != kMagic || ack.status != 0)
      return {kvaf::Status::Internal("cannot reset failed channel setup"), false};
    return {attempt, true};
  }, options.npu_candidate));
  CpuUrmaTransport& transport = *selected_transport;
  Layout layout{};
  Require(MakeLayout(options.config, &layout) && layout.region_bytes == transport.region_bytes(),
          "NPU layout does not match configured AICPU blocks");
  std::vector<Work> work;
  ForEachPipelineChunk(options.config, [&](auto worker, auto chunk, auto rows) { work.push_back({worker, chunk, rows}); });
  const auto ready_base = Align64(kStagingOffset + options.config.rows * layout.stride);
  Require(work.size() <= (kMaxAllocation - ready_base) / sizeof(Control), "too many ready buffers");
  const auto verify_base = ready_base + work.size() * sizeof(Control);
  Require(layout.output_bytes + kGuardBytes <= kMaxAllocation - verify_base, "host buffers exceed allocation limit");
  const auto bytes = (verify_base + layout.output_bytes + kGuardBytes + 4095) & ~std::uint64_t{4095};
  Require(bytes <= kMaxAllocation, "rounded host buffers exceed allocation limit");
  void* host = nullptr;
  Require(posix_memalign(&host, 4096, bytes) == 0, "source allocation failed");
  allocation.reset(host);
  std::memset(host, 0, bytes);
  auto* memory = static_cast<std::uint8_t*>(host);
  auto* input = memory + kStagingOffset;
  auto* request = reinterpret_cast<Control*>(memory + kRequestOffset);
  auto* done = reinterpret_cast<Control*>(memory + kDoneOffset);
  auto* output_ready = reinterpret_cast<Control*>(memory + kOutputReadyOffset);
  auto* output_verified = reinterpret_cast<Control*>(memory + kOutputVerifiedOffset);
  auto* tokens = reinterpret_cast<Control*>(memory + ready_base);
  auto* readback = memory + verify_base;
  std::vector<CpuUrmaTransport::HostSge> sources(options.config.rows), read_sges;
  std::vector<std::uint64_t> offsets(options.config.rows);
  for (std::uint64_t row = 0; row < options.config.rows; ++row)
    for (std::uint64_t b = 8; b < options.config.row_bytes; ++b)
      input[row * layout.stride + b] = Pattern(row, b);
  for (std::uint64_t slot = 0; slot < options.config.rows; ++slot) {
    sources[slot] = {input + LogicalRowForOutputSlot(options.config, slot) * layout.stride,
                    static_cast<std::size_t>(options.config.row_bytes)};
    offsets[slot] = kStagingOffset + slot * options.config.row_bytes;
  }
  Check(transport.RegisterDirectHostMemory(memory, bytes));
  CpuUrmaTransport::HostSge request_sge{request, sizeof(Control)}, done_sge{done, sizeof(Control)};
  CpuUrmaTransport::HostSge ready_sge{output_ready, sizeof(Control)}, ack_sge{output_verified, sizeof(Control)};
  Check(transport.WriteSglBatch(kRequestOffset, &request_sge, 1));
  Check(transport.WriteSglBatch(kDoneOffset, &done_sge, 1));
  std::vector<CpuUrmaTransport::WriteChunk> chunks(work.size());
  std::vector<std::array<std::uint64_t, kJettyCount>> markers(work.size());
  for (std::size_t i = 0; i < work.size(); ++i) {
    const auto& w = work[i];
    for (std::uint32_t q = 0; q < kJettyCount; ++q)
      markers[i][q] = ChainReadyOffset(layout, w.worker, q);
    chunks[i] = {offsets.data() + w.rows.begin, sources.data() + w.rows.begin,
        static_cast<std::size_t>(w.rows.end - w.rows.begin), markers[i].data(), &tokens[i].sequence};
  }
  CpuUrmaTransport::PreparedWriteId data_id{}, done_id{};
  Check(transport.PrepareWrite(chunks.data(), chunks.size(), &data_id));
  const std::uint64_t done_offset = kDoneOffset;
  const CpuUrmaTransport::WriteChunk done_chunk{&done_offset, &done_sge, 1};
  Check(transport.PrepareWrite(&done_chunk, 1, &done_id));
  CpuUrmaTransport::PreparedWriteInfo info{}; Check(transport.DescribePreparedWrite(data_id, &info));
  for (std::uint64_t offset = 0; offset < layout.output_bytes + kGuardBytes;) {
    const auto length = std::min(transport.max_write_bytes(), layout.output_bytes + kGuardBytes - offset);
    Require(length != 0, "invalid read message limit");
    read_sges.push_back({readback + offset, static_cast<std::size_t>(length)}); offset += length;
  }
  OutputReadbackVerifier verifier(options.config, layout);
  std::fprintf(stderr, "AICPU_GET_CPU CONFIG ready=ordered_write copy=securec_memcpy aicpu_blocks=%llu workers=%u jettys=4 "
      "chunk_rows=1024 doorbell=32 cq_batch=8 deadline_polls=32 "
      "data_wrs=%llu marker_wrs=%llu tail_cqes=%llu revision=%llu\n",
      (unsigned long long)options.config.aicpu_blocks, ActiveThreads(options.config),
      (unsigned long long)(info.wr_count - info.marker_wrs), (unsigned long long)info.marker_wrs,
      (unsigned long long)(info.tail_cqes + 1), (unsigned long long)kKernelRevision);
  Socket control(Connect(options.transport.peer_address)); SocketTimeout(control.get(), options.config.timeout_ms + 120000);
  const auto start = kvaf::peer::MakeHeader(kStart, sizeof(kvaf::peer::MessageHeader));
  Send(control.get(), &start, sizeof(start)); Ack launched{}; Receive(control.get(), &launched, sizeof(launched));
  Require(launched.magic == kMagic && !launched.status, "NPU launch failed");
  double data_ns = 0, producer_ns = 0, round_ns = 0;
  std::uint64_t reads = 0;
  bool ok = true;
  CpuUrmaTransport::PreparedAsyncStats stats{};
  std::unique_ptr<CpuUrmaTransport::PreparedWriteSession> session;
  try {
    for (std::uint64_t seq = 1; seq <= layout.rounds; ++seq) {
      if (seq == options.config.warmup + 1) Check(transport.ReadPreparedAsyncStats(&stats, true));
      const bool measured = seq > options.config.warmup;
      const auto round_begin = Clock::now();
      const auto deadline = round_begin + std::chrono::milliseconds(options.config.timeout_ms);
      do {
        Check(transport.ReadSglBatch(kRequestOffset, &request_sge, 1));
        if (measured) ++reads;
        Require(request->sequence <= seq, "future request sequence");
        Require(Clock::now() < deadline, "request timeout");
      } while (request->sequence != seq);
      for (std::uint64_t row = 0; row < options.config.rows; ++row) {
        const auto tag = RowTag(seq, row); std::memcpy(input + row * layout.stride, &tag, 8);
      }
      for (std::size_t i = 0; i < work.size(); ++i)
        tokens[i].sequence = PipelineProgressToken(seq, work[i].chunk + 1);
      Check(transport.BeginPreparedWriteSession(&session));
      const auto producer_begin = Clock::now();
      SubmitAndWait(*session, data_id, deadline);
      const auto data_end = Clock::now();
      done->sequence = seq; SubmitAndWait(*session, done_id, deadline);
      session.reset();
      if (measured) {
        data_ns += Ns(producer_begin, data_end);
        producer_ns += Ns(producer_begin, Clock::now());
      }
      if (seq == layout.rounds) {
        const auto verify_deadline = Clock::now() + std::chrono::milliseconds(options.config.timeout_ms);
        do {
          Check(transport.ReadSglBatch(kOutputReadyOffset, &ready_sge, 1));
          Require(output_ready->sequence <= seq, "future output-ready sequence");
          Require(Clock::now() < verify_deadline, "output-ready timeout");
        } while (output_ready->sequence != seq);
        Check(transport.ReadSglBatch(RegisteredOutputOffset(layout), read_sges.data(), read_sges.size()));
        std::uint64_t mismatch_offset{}; bool matches = verifier.Check(seq, readback, &mismatch_offset);
        for (std::uint64_t i = 0; i < kGuardBytes; ++i)
          if (readback[layout.output_bytes + i] != 0xA5) {
            if (matches) mismatch_offset = layout.output_bytes + i;
            matches = false; break;
          }
        output_verified->sequence = OutputVerificationToken(seq, matches);
        Check(transport.WriteSglBatch(kOutputVerifiedOffset, &ack_sge, 1));
        if (!matches) std::fprintf(stderr, "AICPU_OUTPUT_VERIFY FAIL sequence=%llu offset=%llu\n",
            (unsigned long long)seq, (unsigned long long)mismatch_offset);
        Require(matches, "output payload or guard mismatch");
      }
      if (measured) round_ns += Ns(round_begin, Clock::now());
    }
  } catch (const std::exception& e) { std::fprintf(stderr, "CPU transfer failed: %s\n", e.what()); ok = false; }
  session.reset(); Check(transport.ReadPreparedAsyncStats(&stats));
  const auto stopped = transport.Shutdown();
  if (!stopped.ok()) {
    std::fprintf(stderr, "Shutdown failed: %s; sources retained. Stop CPU before releasing target.\n", stopped.message().c_str());
    for (;;) pause();
  }
  Ack drained{}; drained.status = ok ? 0 : 1; Send(control.get(), &drained, sizeof(drained));
  Report report{}; Receive(control.get(), &report, sizeof(report));
  Require(ok && report.magic == kMagic && !report.status, "benchmark or validation failed");
  const auto n = static_cast<double>(options.config.iterations);
  std::printf("engine,rows,row_bytes,iterations,active_threads,e2e_GBps,mean_ns,p50_ns,p99_ns,"
      "first_ready_ns,pipeline_ns,cpu_data_span_ns,cpu_data_GBps,cpu_producer_ns,cpu_round_ns,cpu_cycle_GBps,"
      "npu_prepare_ns,npu_verify_ns,mean_wrs,mean_marker_wrs,mean_tail_cqes,mean_control_reads\n");
  std::printf("aicpu_sgl_ordered_write,%llu,%llu,%llu,%llu,%.6f,%.1f,%.1f,%.1f,%.1f,%.1f,"
      "%.1f,%.6f,%.1f,%.1f,%.6f,%.1f,%.1f,%llu,%llu,%.1f,%.1f\n",
      (unsigned long long)options.config.rows, (unsigned long long)options.config.row_bytes,
      (unsigned long long)options.config.iterations,
      (unsigned long long)report.active_threads, report.gbps, report.mean_ns, report.p50_ns, report.p99_ns,
      report.wait_ns, report.scatter_ns, data_ns / n, layout.payload_bytes * n / data_ns,
      producer_ns / n, round_ns / n, layout.payload_bytes * n / round_ns, report.prepare_ns, report.verify_ns,
      (unsigned long long)(info.wr_count - info.marker_wrs), (unsigned long long)info.marker_wrs,
      stats.tail_cqes / n, reads / n);
}
}  // namespace
int main(int argc, char** argv) {
  try { Run(Parse(argc, argv)); return 0; }
  catch (const std::exception& e) { std::fprintf(stderr, "AICPU_GET_CPU: %s\n", e.what()); return 1; }
}
