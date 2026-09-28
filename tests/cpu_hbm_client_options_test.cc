#include <initializer_list>

#define main kvaf_cpu_hbm_client_main
#include "cpu_hbm_client.cc"
#undef main

#include "cpu_urma/sgl_batch_builder.h"

namespace {
std::size_t gather_posts = 0;
std::size_t single_posts = 0;
std::size_t read_posts = 0;
std::uint32_t initialized_prepare_threads = 0;
bool corrupt_gather = false;
bool fail_first_probe = false;
bool fail_release = false;
std::uint64_t active_session = 0;
std::uint64_t next_session = 1;
std::size_t live_link_transports = 0;
std::vector<std::uint32_t> allocated_candidates;
std::vector<kvaf::CpuUrmaTransportOptions> initialized_endpoints;
}

// Exercise the real client parser and routing without liburma or hardware.
namespace kvaf {
CpuUrmaDiscovery::~CpuUrmaDiscovery() { Shutdown(); }
Status CpuUrmaDiscovery::Initialize(const std::string& device, bool has_index,
                                    std::uint32_t index) {
  candidates_ = {{"udmac0d1e2", 0U, {}}};
  candidates_[0].eid[15] = 23U;
  if ((!device.empty() && device != candidates_[0].device_name) ||
      (has_index && index != candidates_[0].eid_index)) {
    candidates_.clear();
    return Status::NotFound("mock CPU endpoint not found");
  }
  return Status::OK();
}
void CpuUrmaDiscovery::Shutdown() { candidates_.clear(); }

namespace peer {
PeerLink::PeerLink(std::string address) : address_(std::move(address)) {}
Status PeerLink::GetHbmEidCandidates(
    std::uint32_t, std::vector<std::array<std::uint8_t, kEidBytes>>* candidates) const {
  candidates->resize(2);
  return Status::OK();
}
Status PeerLink::AllocateHbmRegion(std::uint64_t bytes, std::uint64_t,
                                  std::uint32_t, std::uint32_t eid_index,
                                  PeerRegion* region) {
  if (active_session != 0U || live_link_transports != 0U) {
    return Status::Internal("previous attempt was not cleaned before retry");
  }
  allocated_candidates.push_back(eid_index);
  *region = {};
  region->session_id = active_session = next_session++;
  region->base = 0x1000U;
  region->length = bytes;
  region->device_memory = true;
  return Status::OK();
}
Status PeerLink::ReleaseHbmRegion(std::uint64_t session) {
  if (live_link_transports != 0U || session != active_session) {
    return Status::Internal("HBM released before transport shutdown or with wrong session");
  }
  if (fail_release) return Status::Internal("mock HBM release failed");
  active_session = 0U;
  return Status::OK();
}
}  // namespace peer

struct CpuUrmaTransport::Impl {
  std::vector<std::uint8_t> remote;
  std::uint32_t remote_sgl_size{1};
  bool external{false};
  std::uint32_t remote_eid_index{0};
  BatchTiming timing;
};
CpuUrmaTransport::CpuUrmaTransport() : impl_(std::make_unique<Impl>()) {}
CpuUrmaTransport::~CpuUrmaTransport() { (void)Shutdown(); }
Status CpuUrmaTransport::Initialize(const CpuUrmaTransportOptions& options) {
  impl_->remote.resize(options.region_bytes);
  impl_->remote_sgl_size = std::min(options.remote_sgl_size, 13U);
  initialized_prepare_threads = options.prepare_thread_count;
  impl_->external = options.has_external_region;
  impl_->remote_eid_index = options.remote_eid_index;
  if (impl_->external) {
    ++live_link_transports;
    initialized_endpoints.push_back(options);
  }
  return Status::OK();
}
Status CpuUrmaTransport::ValidateLink() {
  return fail_first_probe && impl_->remote_eid_index == 0U
      ? Status::Internal("mock status=4") : Status::OK();
}
Status CpuUrmaTransport::RegisterDirectHostMemory(void*, std::size_t) {
  return Status::OK();
}
Status CpuUrmaTransport::WriteSglBatch(std::uint64_t offset, const HostSge* buffers,
                                     std::size_t count) {
  ++single_posts;
  for (std::size_t i = 0; i < count; ++i) {
    std::memcpy(impl_->remote.data() + offset, buffers[i].data, buffers[i].bytes);
    offset += buffers[i].bytes;
  }
  impl_->timing.wr_count = count;
  return Status::OK();
}
Status CpuUrmaTransport::WriteBatch(const std::uint64_t* offsets, const HostSge* buffers,
                                  std::size_t count) {
  ++gather_posts;
  std::size_t wr_count = 0U;
  const auto status = cpu_urma_detail::BuildWriteGatherShard(
      buffers, offsets, count, 0U, count, impl_->remote_sgl_size,
      impl_->remote.size(), 65536U,
      [](std::size_t, std::size_t, const HostSge&) {},
      [&](std::size_t, const cpu_urma_detail::WriteGatherGroup& group) {
        std::uint64_t destination = offsets[group.first];
        for (std::size_t row = group.first;
             row < group.first + group.count; ++row) {
          const auto* source = buffers[corrupt_gather ? group.first : row].data;
          std::memcpy(impl_->remote.data() + destination,
                      source, buffers[row].bytes);
          destination += buffers[row].bytes;
        }
      },
      &wr_count);
  if (!status.ok()) return status;
  impl_->timing.wr_count = wr_count;
  return Status::OK();
}
Status CpuUrmaTransport::ReadSglBatch(std::uint64_t offset, const HostSge* buffers,
                                    std::size_t count) {
  ++read_posts;
  for (std::size_t i = 0; i < count; ++i) {
    std::memcpy(buffers[i].data, impl_->remote.data() + offset, buffers[i].bytes);
    offset += buffers[i].bytes;
  }
  impl_->timing.wr_count = count;
  return Status::OK();
}
Status CpuUrmaTransport::Shutdown() {
  if (impl_->external) { --live_link_transports; impl_->external = false; }
  return Status::OK();
}
std::uint64_t CpuUrmaTransport::region_bytes() const { return impl_->remote.size(); }
CpuUrmaTransport::BatchTiming CpuUrmaTransport::last_batch_timing() const { return impl_->timing; }
}  // namespace kvaf

namespace {
bool ParseArgs(std::initializer_list<const char*> args, Options* options) {
  std::vector<std::string> storage(args.begin(), args.end());
  std::vector<char*> argv;
  for (auto& arg : storage) argv.push_back(arg.data());
  return Parse(static_cast<int>(argv.size()), argv.data(), options);
}

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

void EndpointSelection() {
  Options options;
  options.peer = "127.0.0.1:28111";
  options.device_id = 6;
  options.remote_sgl_size = 12;
  options.wqes_per_doorbell = 32;
  options.prepare_thread_count = 8;
  fail_first_probe = true;
  {
    HbmConnection connection(options.peer);
    Expect(connection.Initialize(options, 4096).ok(),
           "client must retry NPU candidate 1 after candidate 0 fails its data probe");
    Expect(allocated_candidates == std::vector<std::uint32_t>{0, 1} &&
               initialized_endpoints.size() == 2 && live_link_transports == 1,
           "failed probe must release HBM and destroy its transport before retry");
    const auto& selected = initialized_endpoints.back();
    Expect(selected.ub_device == "udmac0d1e2" && selected.eid_index == 0 &&
               selected.has_eid_index && selected.has_preferred_eid &&
               selected.preferred_eid[15] == 23U && selected.remote_device_id == 6 &&
               selected.remote_eid_index == 1 && selected.remote_hbm &&
               selected.jetty_count == 4 && selected.prepare_thread_count == 8 &&
               selected.remote_sgl_size == 12 && selected.wqes_per_doorbell == 32 &&
               selected.external_region.session_id == active_session,
           "selected endpoints and benchmark options must reach the new transport");
  }
  Expect(active_session == 0 && live_link_transports == 0,
         "successful connection must release transport and HBM on destruction");

  allocated_candidates.clear();
  initialized_endpoints.clear();
  Options pinned;
  Expect(ParseArgs({"client", "127.0.0.1:28111", "--device-id", "6",
                    "--ub-device", "udmac0d1e2", "--eid-index", "0",
                    "--remote-eid-index", "1"}, &pinned),
         "independent local and remote EID options must parse");
  {
    HbmConnection connection(pinned.peer);
    Expect(connection.Initialize(pinned, 4096).ok() &&
               allocated_candidates == std::vector<std::uint32_t>{1} &&
               initialized_endpoints.back().eid_index == 0 &&
               initialized_endpoints.back().remote_eid_index == 1,
           "pinning remote candidate 1 must preserve CPU EID 0 and skip remote 0");
  }

  allocated_candidates.clear();
  pinned.remote_eid_index = 0;
  {
    HbmConnection connection(pinned.peer);
    Expect(!connection.Initialize(pinned, 4096).ok() &&
               allocated_candidates == std::vector<std::uint32_t>{0} && active_session == 0,
           "explicit failed remote candidate must not silently fall back");
  }

  allocated_candidates.clear();
  fail_release = true;
  {
    HbmConnection connection(options.peer);
    const auto status = connection.Initialize(options, 4096);
    Expect(!status.ok() && status.message() == "mock HBM release failed" &&
               allocated_candidates == std::vector<std::uint32_t>{0},
           "cleanup failure must stop candidate retries");
    fail_release = false;
  }
  fail_first_probe = false;
  Expect(active_session == 0 && live_link_transports == 0, "connection cleanup leaked resources");
}
}  // namespace

int main() {
  Options options;
  Expect(ParseArgs({"client", "127.0.0.1:28111", "--blocks", "2048",
                    "--block-bytes", "4096", "--host-space-bytes", "100G",
                    "--device-id", "4", "--jettys", "8", "--doorbell-wqes", "128",
                    "--warmup", "20", "--iterations", "200", "--seed", "1",
                    "--ub-device", "udmac0", "--eid-index", "1",
                    "--remote-sgl-size", "8", "--prepare-threads", "32"}, &options),
         "retained options must parse");
  Expect(options.blocks == 2048 && options.block_bytes == 4096 &&
             options.host_space_bytes == (100ULL << 30U) && options.device_id == 4 &&
             options.jetty_count == 8 && options.wqes_per_doorbell == 128 &&
             options.warmup == 20 && options.iterations == 200 && options.seed == 1 &&
             options.ub_device == "udmac0" && options.has_eid_index && options.eid_index == 1 &&
             options.remote_sgl_size == 8 && options.prepare_thread_count == 32,
         "retained option values must be preserved");
  for (const char* flag : {"--window-bytes", "--stream-depth", "--cq-mod", "--bytes"}) {
    Options removed;
    Expect(!ParseArgs({"client", "127.0.0.1:28111", flag, "1"}, &removed),
           "removed options must be rejected");
  }
  Options defaults;
  Expect(ParseArgs({"client", "127.0.0.1", "28111"}, &defaults) &&
             defaults.peer == "127.0.0.1:28111" && defaults.jetty_count == 4 &&
             defaults.host_space_bytes == 0 && defaults.wqes_per_doorbell == 0 &&
             defaults.remote_sgl_size == 1 && defaults.prepare_thread_count == 4 &&
             defaults.remote_eid_index == UINT32_MAX,
         "peer syntax and defaults must be preserved");
  for (const char* count : {"0", "17"}) {
    Options invalid;
    Expect(!ParseArgs({"client", "127.0.0.1:28111", "--jettys", count}, &invalid),
           "invalid Jetty counts must be rejected");
  }
  for (const char* size : {"0", "-1", "4294967296", "invalid"}) {
    Options invalid;
    Expect(!ParseArgs({"client", "127.0.0.1:28111", "--remote-sgl-size", size}, &invalid),
           "invalid remote SGL sizes must be rejected");
  }
  for (const char* count : {"0", "65", "-1", "4294967296", "invalid"}) {
    Options invalid;
    Expect(!ParseArgs({"client", "127.0.0.1:28111", "--prepare-threads", count}, &invalid),
           "invalid prepare thread counts must be rejected");
  }

  for (const char* index : {"-1", "4294967295", "4294967296", "invalid"}) {
    Options invalid;
    Expect(!ParseArgs({"client", "127.0.0.1:28111", "--remote-eid-index", index}, &invalid),
           "invalid remote EID candidate indices must be rejected");
  }
  EndpointSelection();

  kvaf::CpuUrmaTransport transport;
  kvaf::CpuUrmaTransportOptions config;
  config.region_bytes = 64;
  config.remote_sgl_size = 2;
  Expect(transport.Initialize(config).ok(), "mock transport must initialize");
  std::vector<std::uint8_t> storage(64);
  for (std::size_t i = 0; i < storage.size(); ++i) storage[i] = static_cast<std::uint8_t>(i);
  std::vector<HostBlock> blocks{{storage.data()}, {storage.data() + 16},
                                {storage.data() + 32}, {storage.data() + 48}};
  const std::vector<std::size_t> order{2, 0, 3, 1};
  std::vector<std::uint8_t> expected(64);
  for (std::size_t row = 0U; row < order.size(); ++row) {
    std::memcpy(expected.data() + row * 16U, blocks[order[row]].data, 16U);
  }
  Stats write;
  Expect(RunBatchPass(&transport, true, &blocks, order, 16, 2, &write),
         "gather WRITE must succeed");
  Expect(gather_posts == 2 && single_posts == 0 && write.wrs == 4 && write.bytes == 128,
         "gather WRITE must use local SGE grouping and actual WR accounting");
  std::vector<std::uint8_t> readback(64);
  std::vector<HostBlock> read_blocks{{readback.data()}, {readback.data() + 16},
                                     {readback.data() + 32}, {readback.data() + 48}};
  const std::vector<std::size_t> read_order{0, 1, 2, 3};
  Stats read;
  Expect(RunBatchPass(&transport, false, &read_blocks, read_order, 16, 1, &read) &&
             readback == expected && read_posts == 1 && read.wrs == 4,
         "independent packed READ must preserve gathered row order");
  kvaf::CpuUrmaTransport single_sge_transport;
  config.remote_sgl_size = 1;
  Expect(single_sge_transport.Initialize(config).ok(),
         "single-SGE mock transport must initialize");
  Stats baseline;
  Expect(RunBatchPass(
             &single_sge_transport, true, &blocks, order, 16, 1, &baseline) &&
             single_posts == 0 && gather_posts == 3 && baseline.wrs == 4,
         "single-SGE WRITE must use the same gather path");

  std::vector<std::uint8_t> sparse_storage(128, 0xEE);
  std::vector<HostBlock> sparse{{sparse_storage.data()}, {sparse_storage.data() + 32},
                               {sparse_storage.data() + 64}, {sparse_storage.data() + 96}};
  for (std::size_t row = 0U; row < sparse.size(); ++row) {
    std::fill_n(sparse[row].data, 16U, static_cast<std::uint8_t>(row + 1U));
    std::fill_n(expected.data() + row * 16U, 16U,
                static_cast<std::uint8_t>(order[row] + 1U));
  }
  Stats sparse_write;
  Expect(RunBatchPass(&transport, true, &sparse, order, 16, 1, &sparse_write) &&
             sparse_write.wrs == 2,
         "scattered host rows must gather into two local-SGE WRs");
  std::fill(readback.begin(), readback.end(), 0);
  Stats sparse_read;
  Expect(RunBatchPass(&transport, false, &read_blocks, read_order, 16, 1, &sparse_read) &&
             readback == expected,
         "gathered remote data must omit source gaps and preserve row order");

  (void)setenv("OMP_WAIT_POLICY", "ACTIVE", 1);
  std::vector<std::string> client_storage{
      "client", "127.0.0.1:28111", "--blocks", "4", "--block-bytes", "16",
      "--warmup", "0", "--iterations", "1", "--remote-sgl-size", "2",
      "--prepare-threads", "32"};
  std::vector<char*> client_argv;
  for (auto& arg : client_storage) client_argv.push_back(arg.data());
  Expect(kvaf_cpu_hbm_client_main(
             static_cast<int>(client_argv.size()), client_argv.data()) == 0 &&
             initialized_prepare_threads == 32U,
         "client main must forward prepare threads to the transport");
  corrupt_gather = true;
  Expect(kvaf_cpu_hbm_client_main(
             static_cast<int>(client_argv.size()), client_argv.data()) == 1,
         "packed READ must detect a provider that repeats the first local SGE");
  return 0;
}
