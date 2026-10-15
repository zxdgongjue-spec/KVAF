#include "control.h"
#include "pipeline.h"
#include "host_memory.h"
#include "output_readback.h"
#include "cpu_urma/cpu_urma_link_selection.h"
#include "cpu_urma/cpu_urma_transport.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
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
  HostMemoryOptions memory;
  std::uint32_t npu_candidate{UINT32_MAX};
};
Options Parse(int argc, char** argv) {
  Options o; o.transport.remote_sgl_size = 16; o.transport.wqes_per_doorbell = 32;
  o.memory.numa_node = "auto";
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--verify-payload") { o.config.verify_payload = o.config.verify_each_round = 1; continue; }
    Require(i + 1 < argc, "missing value for " + flag); const char* value = argv[++i];
    if (flag == "--peer") { o.transport.peer_address = value; continue; }
    if (flag == "--ub-device") { o.transport.ub_device = value; continue; }
    if (flag == "--host-numa-node") { o.memory.numa_node = value; continue; }
    if (flag == "--payload-cache-maintenance") {
      const std::string mode = value; Require(mode == "on" || mode == "off", "payload cache must be on or off");
      o.config.payload_cache_maintenance = mode == "on"; continue;
    }
    if (flag == "--output-verify") {
      const std::string mode = value; Require(mode == "final" || mode == "urma", "output verify must be final or urma");
      o.config.verify_each_round = mode == "urma"; continue;
    }
    const auto n = Number(value);
    if (flag == "--rows") o.config.rows = n;
    else if (flag == "--row-bytes") o.config.row_bytes = n;
    else if (flag == "--warmup") o.config.warmup = n;
    else if (flag == "--iterations") o.config.iterations = n;
    else if (flag == "--timeout-ms") o.config.timeout_ms = n;
    else if (flag == "--scatter-threads") o.config.scatter_threads = n;
    else if (flag == "--pipeline-chunk-rows") o.config.pipeline_chunk_rows = n;
    else if (flag == "--host-cpu") { Require(n < CPU_SETSIZE, "invalid host CPU"); o.memory.cpu = static_cast<int>(n); }
    else {
      Require(n <= UINT32_MAX, "option exceeds uint32");
      if (flag == "--device-id") { Require(n <= INT32_MAX, "invalid NPU device"); o.transport.remote_device_id = n; }
      else if (flag == "--npu-candidate") { Require(n < UINT32_MAX, "invalid candidate"); o.npu_candidate = n; }
      else if (flag == "--eid-index") { o.transport.has_eid_index = true; o.transport.eid_index = n; }
      else if (flag == "--jettys") o.transport.jetty_count = n;
      else if (flag == "--remote-sgl-size") o.transport.remote_sgl_size = n;
      else if (flag == "--prepare-threads") o.transport.prepare_thread_count = n;
      else if (flag == "--wqes-per-doorbell") o.transport.wqes_per_doorbell = n;
      else throw std::runtime_error("unknown option " + flag);
    }
  }
  o.config.pipeline_jettys = o.transport.jetty_count;
  Layout layout{};
  Require(MakeLayout(o.config, &layout) && !o.transport.peer_address.empty() &&
      o.transport.remote_sgl_size && o.transport.prepare_thread_count && o.transport.prepare_thread_count <= 64 &&
      o.transport.wqes_per_doorbell <= 8192, "invalid benchmark configuration");
  Require(!o.config.verify_payload || o.config.verify_each_round, "payload validation requires --output-verify urma");
  if (o.memory.numa_node != "auto" && o.memory.numa_node != "none")
    Require(Number(o.memory.numa_node.c_str()) < 4096, "invalid NUMA node");
  return o;
}
struct Work { std::uint32_t worker; std::uint64_t chunk; RowRange rows; };
struct PollWork { double post{}, poll{}; };
void SubmitAndWait(CpuUrmaTransport::PreparedWriteSession& session, std::uint64_t id,
                   Clock::time_point deadline, PollWork& work) {
  bool submitted = false; auto begin = Clock::now();
  Check(session.Submit(id, &submitted)); Require(submitted, "drained queue has insufficient credits");
  work.post += Ns(begin, Clock::now());
  std::uint64_t polls = 0;
  for (;;) {
    if (++polls % 32 == 0) Require(Clock::now() < deadline, "producer timeout");
    CpuUrmaTransport::PreparedWriteCompletion completed{}; std::size_t count = 0;
    begin = Clock::now(); Check(session.Poll(&completed, 1, &count)); work.poll += Ns(begin, Clock::now());
    if (count) { Require(count == 1 && completed.id == id, "unexpected completion"); return; }
  }
}
kvaf::peer::PeerRegion Configure(const Options& options, std::uint32_t npu_candidate) {
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
          reply.jetty_key_bytes <= kvaf::peer::kQpKeyBytes,
          "NPU configure/export failed; check device-id and update both test binaries");
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
  Layout layout{}; Require(MakeLayout(options.config, &layout), "invalid layout");
  std::vector<Work> work;
  ForEachPipelineChunk(options.config, [&](auto worker, auto chunk, auto rows) { work.push_back({worker, chunk, rows}); });
  const auto ready_base = Align64(kStagingOffset + options.config.rows * layout.source_stride);
  Require(work.size() <= (kMaxAllocation - ready_base) / sizeof(Control), "too many ready buffers");
  const auto verify_base = ready_base + work.size() * sizeof(Control);
  Require(layout.output_bytes + kGuardBytes <= kMaxAllocation - verify_base, "host buffers exceed allocation limit");
  HostMemory allocation; // Must outlive transport and every posted descriptor.
  options.transport.region_bytes = layout.region_bytes;
  std::unique_ptr<CpuUrmaTransport> selected_transport;
  std::string selected_device;
  kvaf::peer::PeerLink peer(options.transport.peer_address);
  Check(kvaf::SelectCpuUrmaDeviceLink(peer, options.transport.remote_device_id, discovery.candidates(),
      [&](const kvaf::CpuUrmaCandidate& cpu, std::uint32_t npu_index) -> kvaf::CpuUrmaLinkAttempt {
    auto config = kvaf::BindCpuUrmaEndpoint(options.transport, cpu, options.transport.remote_device_id, npu_index);
    // Match Runtime: each pair gets a fresh transport. Shutdown does not reset
    // every selection field (e.g. device_), so reinitializing one object could
    // silently retain the previous CPU EID.
    auto candidate = std::make_unique<CpuUrmaTransport>();
    kvaf::Status attempt;
    try {
      config.external_region = Configure(options, npu_index);
      config.has_external_region = true;
      attempt = candidate->Initialize(config);
    } catch (const std::exception& e) {
      attempt = kvaf::Status::Internal(e.what());
    }
    // Unlike Runtime's allocator, this test does not recycle a target after a
    // failed data probe. It retries setup only, before any data WR is posted.
    if (attempt.ok()) {
      attempt = candidate->ValidateLink();
      if (attempt.ok()) { selected_device = cpu.device_name; selected_transport = std::move(candidate); }
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
  allocation.Allocate(verify_base + layout.output_bytes + kGuardBytes, options.memory, selected_device);
  auto* memory = static_cast<std::uint8_t*>(allocation.data());
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
      input[row * layout.source_stride + b] = Pattern(row, b);
  for (std::uint64_t slot = 0; slot < options.config.rows; ++slot) {
    sources[slot] = {input + LogicalRowForOutputSlot(options.config, slot) * layout.source_stride,
                    static_cast<std::size_t>(options.config.row_bytes)};
    offsets[slot] = kStagingOffset + slot * options.config.row_bytes;
  }
  Check(transport.RegisterDirectHostMemory(memory, allocation.bytes()));
  CpuUrmaTransport::HostSge request_sge{request, sizeof(Control)}, done_sge{done, sizeof(Control)};
  CpuUrmaTransport::HostSge ready_sge{output_ready, sizeof(Control)}, ack_sge{output_verified, sizeof(Control)};
  Check(transport.WriteSglBatch(kRequestOffset, &request_sge, 1));
  Check(transport.WriteSglBatch(kDoneOffset, &done_sge, 1));
  std::vector<CpuUrmaTransport::WriteChainChunk> chunks(work.size());
  std::vector<std::array<std::uint64_t, 16>> markers(work.size());
  for (std::size_t i = 0; i < work.size(); ++i) {
    const auto& w = work[i];
    for (std::uint32_t q = 0; q < options.transport.jetty_count; ++q)
      markers[i][q] = ChainReadyOffset(options.config, layout, w.worker, q);
    chunks[i] = {offsets.data() + w.rows.begin, sources.data() + w.rows.begin,
        static_cast<std::size_t>(w.rows.end - w.rows.begin), markers[i].data(), &tokens[i].sequence};
  }
  CpuUrmaTransport::PreparedWriteId data_id{}, done_id{};
  Check(transport.PrepareWriteChain(chunks.data(), chunks.size(), &data_id));
  const std::uint64_t done_offset = kDoneOffset;
  Check(transport.PrepareWriteBatch(&done_offset, &done_sge, 1, &done_id));
  CpuUrmaTransport::PreparedWriteInfo info{}; Check(transport.DescribePreparedWrite(data_id, &info));
  for (std::uint64_t offset = 0; offset < layout.output_bytes + kGuardBytes;) {
    const auto length = std::min(transport.max_write_bytes(), layout.output_bytes + kGuardBytes - offset);
    Require(length != 0, "invalid read message limit");
    read_sges.push_back({readback + offset, static_cast<std::size_t>(length)}); offset += length;
  }
  OutputReadbackVerifier verifier(options.config, layout);
  std::fprintf(stderr, "AICPU_GET_CPU CONFIG ready=ordered_write copy=securec_memcpy workers=%u jettys=%u "
      "chunk_rows=%llu doorbell=%u cq_batch=8 deadline_polls=32 payload_cache=%s output_verify=%s "
      "data_wrs=%llu marker_wrs=%llu tail_cqes=%llu revision=%llu\n",
      ActiveThreads(options.config), options.transport.jetty_count,
      (unsigned long long)options.config.pipeline_chunk_rows, options.transport.wqes_per_doorbell,
      options.config.payload_cache_maintenance ? "on" : "off", options.config.verify_each_round ? "urma" : "final",
      (unsigned long long)(info.wr_count - info.marker_wrs), (unsigned long long)info.marker_wrs,
      (unsigned long long)(info.tail_cqes + 1), (unsigned long long)kKernelRevision);
  Socket control(Connect(options.transport.peer_address)); SocketTimeout(control.get(), options.config.timeout_ms + 120000);
  const auto start = kvaf::peer::MakeHeader(kStart, sizeof(kvaf::peer::MessageHeader));
  Send(control.get(), &start, sizeof(start)); Ack launched{}; Receive(control.get(), &launched, sizeof(launched));
  Require(launched.magic == kMagic && !launched.status, "NPU launch failed");
  double data_ns = 0, producer_ns = 0, post_ns = 0, poll_ns = 0, done_ns = 0, scheduler_ns = 0;
  double source_ns = 0, prepare_ns = 0, request_ns = 0, round_ns = 0, verify_ns = 0;
  std::uint64_t reads = 0, checks = 0;
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
      auto begin = Clock::now(); if (measured) request_ns += Ns(round_begin, begin);
      for (std::uint64_t row = 0; row < options.config.rows; ++row) {
        const auto tag = RowTag(seq, row); std::memcpy(input + row * layout.source_stride, &tag, 8);
        if (options.config.verify_payload)
          for (std::uint64_t b = 8; b < options.config.row_bytes; ++b)
            input[row * layout.source_stride + b] = PayloadPattern(options.config, seq, row, b);
      }
      const auto source_end = Clock::now(); if (measured) source_ns += Ns(begin, source_end);
      for (std::size_t i = 0; i < work.size(); ++i)
        tokens[i].sequence = PipelineProgressToken(seq, work[i].chunk + 1);
      Check(transport.BeginPreparedWriteSession(&session));
      const auto producer_begin = Clock::now(); if (measured) prepare_ns += Ns(source_end, producer_begin);
      PollWork data{}; const auto data_begin = Clock::now();
      SubmitAndWait(*session, data_id, deadline, data);
      const auto data_end = Clock::now(); const auto done_begin = data_end;
      done->sequence = seq; PollWork notify{}; SubmitAndWait(*session, done_id, deadline, notify);
      const auto done_end = Clock::now(); session.reset(); const auto producer_end = Clock::now();
      if (measured) {
        data_ns += Ns(data_begin, data_end); post_ns += data.post; poll_ns += data.poll;
        const auto notification = Ns(done_begin, done_end); done_ns += notification;
        const auto elapsed = Ns(producer_begin, producer_end); producer_ns += elapsed;
        scheduler_ns += elapsed - data.post - data.poll - notification;
      }
      if (ShouldVerifyOutput(options.config, seq)) {
        const auto verify_begin = Clock::now();
        const auto verify_deadline = verify_begin + std::chrono::milliseconds(options.config.timeout_ms);
        do {
          Check(transport.ReadSglBatch(kOutputReadyOffset, &ready_sge, 1));
          Require(output_ready->sequence <= seq, "future output-ready sequence");
          Require(Clock::now() < verify_deadline, "output-ready timeout");
        } while (output_ready->sequence != seq);
        Check(transport.ReadSglBatch(RegisteredOutputOffset(options.config, layout), read_sges.data(), read_sges.size()));
        OutputMismatch mismatch{}; bool matches = verifier.Check(seq, readback, &mismatch);
        for (std::uint64_t i = 0; i < kGuardBytes; ++i)
          if (readback[layout.output_bytes + i] != 0xA5) {
            if (matches) mismatch = {layout.output_bytes + i, 0xA5, readback[layout.output_bytes + i]};
            matches = false; break;
          }
        output_verified->sequence = OutputVerificationToken(seq, matches);
        Check(transport.WriteSglBatch(kOutputVerifiedOffset, &ack_sge, 1));
        if (measured) { ++checks; verify_ns += Ns(verify_begin, Clock::now()); }
        if (!matches) std::fprintf(stderr, "AICPU_OUTPUT_VERIFY FAIL sequence=%llu offset=%llu\n",
            (unsigned long long)seq, (unsigned long long)mismatch.offset);
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
  std::printf("engine,rows,row_bytes,iterations,scatter_threads,active_threads,e2e_GBps,mean_ns,p50_ns,p99_ns,"
      "first_ready_ns,pipeline_ns,cpu_data_span_ns,cpu_data_GBps,cpu_producer_ns,cpu_data_submit_ns,"
      "cpu_cq_poll_work_ns,cpu_done_ns,cpu_scheduler_ns,cpu_source_update_ns,cpu_producer_prepare_ns,"
      "cpu_request_wait_ns,cpu_round_ns,cpu_cycle_GBps,npu_prepare_ns,npu_verify_ns,mean_wrs,mean_marker_wrs,"
      "mean_tail_cqes,mean_control_reads,payload_cache_maintenance,output_verify,verify_payload,output_verify_checks,cpu_output_verify_ns\n");
  std::printf("aicpu_sgl_ordered_write,%llu,%llu,%llu,%llu,%llu,%.6f,%.1f,%.1f,%.1f,%.1f,%.1f,"
      "%.1f,%.6f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.6f,%.1f,%.1f,%llu,%llu,%.1f,%.1f,%s,%s,%llu,%llu,%.1f\n",
      (unsigned long long)options.config.rows, (unsigned long long)options.config.row_bytes,
      (unsigned long long)options.config.iterations, (unsigned long long)options.config.scatter_threads,
      (unsigned long long)report.active_threads, report.gbps, report.mean_ns, report.p50_ns, report.p99_ns,
      report.wait_ns, report.scatter_ns, data_ns / n, layout.payload_bytes * n / data_ns,
      producer_ns / n, post_ns / n, poll_ns / n, done_ns / n, scheduler_ns / n, source_ns / n, prepare_ns / n,
      request_ns / n, round_ns / n, layout.payload_bytes * n / round_ns, report.prepare_ns, report.verify_ns,
      (unsigned long long)(info.wr_count - info.marker_wrs), (unsigned long long)info.marker_wrs,
      stats.tail_cqes / n, reads / n, options.config.payload_cache_maintenance ? "on" : "off",
      options.config.verify_each_round ? "urma" : "final", (unsigned long long)options.config.verify_payload,
      (unsigned long long)checks, verify_ns / n);
}
}  // namespace
int main(int argc, char** argv) {
  try { Run(Parse(argc, argv)); return 0; }
  catch (const std::exception& e) { std::fprintf(stderr, "AICPU_GET_CPU: %s\n", e.what()); return 1; }
}
