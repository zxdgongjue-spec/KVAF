#include "control.h"
#include "transfer_profile.h"
#include "cpu_urma/cpu_urma_link_selection.h"
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
std::uint64_t ClockNs() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now().time_since_epoch()).count());
}
struct Options {
  Config config;
  kvaf::CpuUrmaTransportOptions transport;
  std::string transfer_csv;
};
Options Parse(int argc, char** argv) {
  Options o;
  o.transport.remote_sgl_size = 16;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--verify-payload") { o.config.verify_payload = 1; continue; }
    if (flag == "--profile-transfer") { o.config.profile_transfer = 1; continue; }
    Require(i + 1 < argc, "missing value for " + flag);
    const char* value = argv[++i];
    if (flag == "--peer") { o.transport.peer_address = value; continue; }
    if (flag == "--ub-device") { o.transport.ub_device = value; continue; }
    if (flag == "--profile-transfer-csv") {
      o.transfer_csv = value; o.config.profile_transfer = 1; continue;
    }
    if (flag == "--transfer-mode") {
      const std::string mode = value;
      Require(mode == "scatter" || mode == "direct_output" || mode == "scatter_pipeline",
              "--transfer-mode must be scatter, direct_output or scatter_pipeline");
      o.config.transfer_mode = mode == "direct_output" ? TransferMode::kDirectOutput :
          mode == "scatter_pipeline" ? TransferMode::kScatterPipeline : TransferMode::kScatter;
      continue;
    }
    const auto n = Number(value);
    if (flag == "--rows") o.config.rows = n;
    else if (flag == "--scatter-threads") o.config.scatter_threads = n;
    else if (flag == "--row-bytes") o.config.row_bytes = n;
    else if (flag == "--warmup") o.config.warmup = n;
    else if (flag == "--iterations") o.config.iterations = n;
    else if (flag == "--timeout-ms") o.config.timeout_ms = n;
    else if (flag == "--direct-output-stride") o.config.direct_output_stride = n;
    else {
      Require(n <= UINT32_MAX, "option exceeds uint32");
      if (flag == "--eid-index") { o.transport.has_eid_index = true; o.transport.eid_index = n; }
      else if (flag == "--device-id") {
        Require(n <= INT32_MAX, "invalid NPU device-id"); o.transport.remote_device_id = n;
      }
      else if (flag == "--remote-sgl-size") o.transport.remote_sgl_size = n;
      else if (flag == "--jettys") o.transport.jetty_count = n;
      else if (flag == "--prepare-threads") o.transport.prepare_thread_count = n;
      else if (flag == "--wqes-per-doorbell") o.transport.wqes_per_doorbell = n;
      else throw std::runtime_error("unknown option " + flag);
    }
  }
  Layout layout{};
  Require(!o.config.profile_transfer || o.config.transfer_mode == TransferMode::kScatter,
          "--profile-transfer requires --transfer-mode scatter (no overlapping pipeline timings)");
  Require(MakeLayout(o.config, &layout) && !o.transport.peer_address.empty() &&
          o.transport.remote_sgl_size > 0 && o.transport.jetty_count > 0 &&
          o.transport.jetty_count <= 16 && o.transport.prepare_thread_count > 0 &&
          o.transport.prepare_thread_count <= 64, "invalid benchmark configuration");
  o.transport.profile_write_queues = o.config.profile_transfer != 0;
  return o;
}
void Check(const kvaf::Status& status) { Require(status.ok(), status.message()); }
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
  std::memcpy(region.segment_key.data(), reply.segment_key, sizeof(reply.segment_key));
  std::memcpy(region.jetty_key.data(), reply.jetty_key, sizeof(reply.jetty_key));
  return region;
}
void Run(Options options) {
  std::unique_ptr<FILE, decltype(&std::fclose)> transfer_csv(nullptr, &std::fclose);
  if (!options.transfer_csv.empty()) {
    transfer_csv.reset(std::fopen(options.transfer_csv.c_str(), "w"));
    Require(transfer_csv != nullptr, "cannot open --profile-transfer-csv: " + options.transfer_csv);
  }
  kvaf::CpuUrmaDiscovery discovery;
  Check(discovery.Initialize(options.transport.ub_device, options.transport.has_eid_index,
                             options.transport.eid_index));
  Layout layout{}; Require(MakeLayout(options.config, &layout), "invalid layout");
  // One page-aligned registered allocation contains source rows and all
  // host-side control buffers. It outlives transport destruction on all paths.
  void* memory = nullptr;
  const auto bytes = (options.config.rows * layout.source_stride + 4096 + 4095) & ~std::uint64_t{4095};
  Require(posix_memalign(&memory, 4096, bytes) == 0, "source allocation failed");
  std::unique_ptr<void, decltype(&std::free)> allocation(memory, std::free);
  std::memset(memory, 0, bytes);
  auto* input = static_cast<std::uint8_t*>(memory) + 4096;
  auto* request = static_cast<Control*>(memory);
  auto* done = reinterpret_cast<Control*>(static_cast<std::uint8_t*>(memory) + 256);
  std::vector<CpuUrmaTransport::HostSge> sources(options.config.rows);
  std::vector<std::uint64_t> offsets(options.config.rows);
  for (std::uint64_t row = 0; row < options.config.rows; ++row) {
    auto* data = input + row * layout.source_stride;
    for (std::uint64_t b = 8; b < options.config.row_bytes; ++b) data[b] = Pattern(row, b);
  }
  // Order local sources by physical output slot in all modes. Scatter writes
  // packed staging; direct output uses the final output's remote stride.
  for (std::uint64_t slot = 0; slot < options.config.rows; ++slot) {
    const auto row = LogicalRowForOutputSlot(options.config, slot);
    sources[slot] = {input + row * layout.source_stride,
                     static_cast<std::size_t>(options.config.row_bytes)};
    offsets[slot] = kStagingOffset + slot *
        (IsDirectOutput(options.config) ? layout.stride : options.config.row_bytes);
  }
  std::fprintf(stderr,
               "AICPU_GET_CPU SGL_LAYOUT=source_by_output_slot transfer_mode=%s "
               "write_target=%s source_stride=%llu output_stride=%llu\n",
               TransferModeName(options.config), IsDirectOutput(options.config) ? "output" : "staging",
               (unsigned long long)layout.source_stride, (unsigned long long)layout.stride);
  options.transport.region_bytes = layout.region_bytes;
  std::unique_ptr<CpuUrmaTransport> selected_transport;
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
  }));
  CpuUrmaTransport& transport = *selected_transport;
  // ValidateLink writes at offset zero. Restore the request record afterward.
  Check(transport.RegisterDirectHostMemory(memory, bytes));
  CpuUrmaTransport::HostSge request_sge{request, sizeof(*request)}, done_sge{done, sizeof(*done)};
  Check(transport.WriteSglBatch(kRequestOffset, &request_sge, 1));
  Check(transport.WriteSglBatch(kDoneOffset, &done_sge, 1));
  const bool pipeline = IsScatterPipeline(options.config);
  const bool profile_transfer = options.config.profile_transfer != 0;
  // Allocate once before launch; never allocate/log detailed rows in the loop.
  std::vector<CpuTransferSample> transfer_samples(profile_transfer ? options.config.iterations : 0);
  const auto workers = ActiveThreads(options.config);
  std::vector<CpuUrmaTransport::HostSge> ready_sges;
  if (pipeline) {
    ready_sges.resize(workers);
    for (std::uint32_t worker = 0; worker < workers; ++worker) {
      ready_sges[worker] = {static_cast<std::uint8_t*>(memory) + PipelineReadyOffset(worker), sizeof(Control)};
      Check(transport.WriteSglBatch(PipelineReadyOffset(worker), &ready_sges[worker], 1));
    }
    std::fprintf(stderr, "[AICPU_PIPELINE] CPU ranges=%u scheduling=range_order "
        "ready=after_range_write_batch staging=disjoint_full_batch\n", workers);
  }
  if (options.config.verify_payload)
    std::fprintf(stderr, "AICPU_GET_CPU VALIDATION payload=sequence_pattern "
        "source_generation=inside_request_wait performance=validation_only\n");
  if (profile_transfer)
    std::fprintf(stderr, "[AICPU_TRANSFER] ENABLED mode=scatter warmup_excluded=1 "
        "trace=after_drain detailed_csv=%s\n",
        options.transfer_csv.empty() ? "disabled" : options.transfer_csv.c_str());
  Socket control(Connect(options.transport.peer_address));
  SocketTimeout(control.get(), options.config.timeout_ms + 120000);
  auto start = kvaf::peer::MakeHeader(kStart, sizeof(kvaf::peer::MessageHeader));
  Send(control.get(), &start, sizeof(start));
  Ack ack{}; Receive(control.get(), &ack, sizeof(ack));
  Require(ack.magic == kMagic && ack.status == 0, "NPU kernel launch failed");
  long double write_ns = 0, prepare_ns = 0, post_ns = 0, poll_ns = 0;
  long double notify_ns = 0;
  std::uint64_t polls = 0, wr_count = 0;
  bool ok = true;
  try {
    for (std::uint64_t seq = 1; seq <= layout.rounds; ++seq) {
      const bool measure_transfer = profile_transfer && seq > options.config.warmup;
      CpuTransferSample* transfer = measure_transfer
          ? &transfer_samples[seq - options.config.warmup - 1] : nullptr;
      const auto request_poll_begin = measure_transfer ? ClockNs() : 0;
      const auto polls_before = polls;
      std::uint64_t observed = 0, source_end = 0, write_begin = 0, write_end = 0;
      std::uint64_t notify_begin = 0, notify_end = 0;
      const auto deadline = Clock::now() + std::chrono::milliseconds(options.config.timeout_ms);
      do {
        Check(transport.ReadSglBatch(kRequestOffset, &request_sge, 1));
        ++polls;
        if (request->sequence > seq)
          throw std::runtime_error("unexpected future AICPU request: expected=" + std::to_string(seq) +
                                   " observed=" + std::to_string(request->sequence));
        if (Clock::now() >= deadline)
          throw std::runtime_error("timed out waiting for AICPU request: expected=" + std::to_string(seq) +
                                   " observed=" + std::to_string(request->sequence) +
                                   " last_done=" + std::to_string(done->sequence));
      } while (request->sequence != seq);
      if (measure_transfer) observed = ClockNs();
      for (std::uint64_t row = 0; row < options.config.rows; ++row) {
        const auto tag = RowTag(seq, row);
        std::memcpy(input + row * layout.source_stride, &tag, sizeof(tag));
        if (options.config.verify_payload)
          for (std::uint64_t b = 8; b < options.config.row_bytes; ++b)
            input[row * layout.source_stride + b] = PayloadPattern(options.config, seq, row, b);
      }
      if (measure_transfer) {
        source_end = ClockNs();
        transfer->sequence = seq;
        transfer->request_poll_ns = observed - request_poll_begin;
        transfer->request_reads = polls - polls_before;
        transfer->request = transport.last_batch_timing(); // Matching READ only.
      }
      const auto write_range = [&](RowRange range) {
        if (measure_transfer) write_begin = ClockNs();
        Check(transport.WriteBatch(offsets.data() + range.begin, sources.data() + range.begin,
                                   range.end - range.begin));
        if (measure_transfer) write_end = ClockNs();
        const auto timing = transport.last_batch_timing();
        if (measure_transfer) {
          transfer->data = timing;
        }
        if (seq > options.config.warmup) {
          write_ns += timing.total_ns; prepare_ns += timing.prepare_ns;
          post_ns += timing.post_ns; poll_ns += timing.poll_ns; wr_count += timing.wr_count;
        }
      };
      const auto publish = [&](std::uint64_t offset, CpuUrmaTransport::HostSge& sge) {
        static_cast<Control*>(sge.data)->sequence = seq;
        Clock::time_point before{};
        if (pipeline && seq > options.config.warmup) before = Clock::now();
        if (measure_transfer) notify_begin = ClockNs();
        // A ready WRITE is submitted only after this range's WriteBatch has
        // waited for every participating Jetty. Never infer cross-Jetty
        // completion from a single queue's tail. Provider HBM visibility
        // uses the same completion/cache contract as the serial path.
        Check(transport.WriteSglBatch(offset, &sge, 1));
        if (measure_transfer) {
          notify_end = ClockNs();
          transfer->notify = transport.last_batch_timing();
        }
        if (pipeline && seq > options.config.warmup)
          notify_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - before).count();
      };
      if (pipeline) {
        for (std::uint32_t worker = 0; worker < workers; ++worker) {
          write_range(ScatterRange(options.config, worker, workers));
          publish(PipelineReadyOffset(worker), ready_sges[worker]);
        }
      } else {
        write_range({0, options.config.rows});
      }
      publish(kDoneOffset, done_sge);
      if (measure_transfer) {
        FinishCpuTransferSample(*transfer, observed, source_end, write_begin, write_end,
                                notify_begin, notify_end);
        // This snapshot survives control WRs; copy it after publishing done
        // so the larger diagnostic record does not delay staging consumption.
        transfer->queues = transport.last_write_queue_profile();
      }
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
  if (profile_transfer) {
    TransferProfileHeader header{};
    Receive(control.get(), &header, sizeof(header));
    Require(header.magic == kMagic && header.revision == kKernelRevision &&
            header.rounds == options.config.iterations && header.frequency > 0 &&
            header.sample_bytes == sizeof(Sample), "incompatible device transfer profile");
    std::vector<Sample> device_samples(header.rounds);
    Receive(control.get(), device_samples.data(), device_samples.size() * sizeof(Sample));
    PrintTransferProfile(stderr, options.config, header.frequency, transfer_samples, device_samples);
    if (transfer_csv)
      PrintTransferCsv(transfer_csv.get(), options.config, header.frequency, transfer_samples, device_samples);
  }
  const double n = static_cast<double>(options.config.iterations);
  std::printf("engine,rows,row_bytes,iterations,scatter_threads,active_threads,e2e_GBps,mean_ns,p50_ns,p99_ns,%s,%s,"
              "cpu_write_ns,cpu_prepare_ns,cpu_post_ns,cpu_poll_ns,mean_wrs,control_reads",
              pipeline ? "first_ready_ns" : "wait_ns", pipeline ? "pipeline_ns" : "scatter_ns");
  if (pipeline) std::printf(",cpu_notify_ns,mean_data_batches,mean_notify_wrs");
  std::printf("\n");
  std::printf("%s,%llu,%llu,%llu,%llu,%llu,%.6f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%llu",
              IsDirectOutput(options.config) ? "aicpu_urma_direct_output" :
              pipeline ? "aicpu_memcpy_sgl_reordered_pipeline" :
              (report.active_threads > 1 ? "aicpu_memcpy_sgl_reordered_mt" :
                                           "aicpu_memcpy_sgl_reordered"),
              (unsigned long long)options.config.rows, (unsigned long long)options.config.row_bytes,
              (unsigned long long)options.config.iterations,
              (unsigned long long)options.config.scatter_threads,
              (unsigned long long)report.active_threads, report.gbps, report.mean_ns,
              report.p50_ns, report.p99_ns, report.wait_ns, report.scatter_ns,
              (double)write_ns / n, (double)prepare_ns / n, (double)post_ns / n,
              (double)poll_ns / n, (double)wr_count / n, (unsigned long long)polls);
  if (pipeline) std::printf(",%.1f,%u,%u", (double)notify_ns / n, workers, workers + 1);
  std::printf("\n");
}
}  // namespace
int main(int argc, char** argv) {
  try { Run(Parse(argc, argv)); return 0; }
  catch (const std::exception& e) {
    std::fprintf(stderr, "AICPU_GET_CPU: %s\n", e.what()); return 1;
  }
}
