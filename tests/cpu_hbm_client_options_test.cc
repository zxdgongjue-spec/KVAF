#include <initializer_list>

#define main kvaf_cpu_hbm_client_main
#include "cpu_hbm_client.cc"
#undef main

#include "cpu_urma/sgl_batch_plan.h"

namespace {
std::size_t scatter_posts = 0;
std::size_t single_posts = 0;
std::size_t read_posts = 0;
}

// Exercise the real client parser and routing without liburma or hardware.
namespace kvaf {
struct CpuUrmaTransport::Impl {
  std::vector<std::uint8_t> remote;
  std::uint32_t remote_sgl_size{1};
  BatchTiming timing;
};
CpuUrmaTransport::CpuUrmaTransport() : impl_(std::make_unique<Impl>()) {}
CpuUrmaTransport::~CpuUrmaTransport() = default;
Status CpuUrmaTransport::Initialize(const CpuUrmaTransportOptions& options) {
  impl_->remote.resize(options.region_bytes);
  impl_->remote_sgl_size = options.remote_sgl_size;
  return Status::OK();
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
  ++scatter_posts;
  std::vector<cpu_urma_detail::SglGroup> groups;
  const auto status = cpu_urma_detail::PlanWriteScatterGroups(
      buffers, offsets, count, impl_->remote_sgl_size, impl_->remote.size(), &groups);
  if (!status.ok()) return status;
  for (std::size_t i = 0; i < count; ++i) {
    std::memcpy(impl_->remote.data() + offsets[i], buffers[i].data, buffers[i].bytes);
  }
  impl_->timing.wr_count = groups.size();
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
Status CpuUrmaTransport::Shutdown() { return Status::OK(); }
std::uint64_t CpuUrmaTransport::region_bytes() const { return 0; }
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
}  // namespace

int main() {
  Options options;
  Expect(ParseArgs({"client", "127.0.0.1:28111", "--blocks", "2048",
                    "--block-bytes", "4096", "--host-space-bytes", "100G",
                    "--device-id", "4", "--jettys", "8", "--doorbell-wqes", "128",
                    "--warmup", "20", "--iterations", "200", "--seed", "1",
                    "--ub-device", "udmac0", "--eid-index", "1",
                    "--remote-sgl-size", "8"}, &options),
         "retained options must parse");
  Expect(options.blocks == 2048 && options.block_bytes == 4096 &&
             options.host_space_bytes == (100ULL << 30U) && options.device_id == 4 &&
             options.jetty_count == 8 && options.wqes_per_doorbell == 128 &&
             options.warmup == 20 && options.iterations == 200 && options.seed == 1 &&
             options.ub_device == "udmac0" && options.has_eid_index && options.eid_index == 1 &&
             options.remote_sgl_size == 8,
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
             defaults.remote_sgl_size == 1,
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

  kvaf::CpuUrmaTransport transport;
  kvaf::CpuUrmaTransportOptions config;
  config.region_bytes = 64;
  config.remote_sgl_size = 2;
  Expect(transport.Initialize(config).ok(), "mock transport must initialize");
  std::vector<std::uint8_t> storage(64);
  for (std::size_t i = 0; i < storage.size(); ++i) storage[i] = static_cast<std::uint8_t>(i);
  const auto expected = storage;
  std::vector<HostBlock> blocks{{storage.data()}, {storage.data() + 16},
                                {storage.data() + 32}, {storage.data() + 48}};
  const std::vector<std::size_t> order{2, 0, 3, 1};
  Stats write;
  Expect(RunBatchPass(&transport, true, &blocks, order, 16, 2, &write, 2),
         "scatter WRITE must succeed");
  Expect(scatter_posts == 2 && single_posts == 0 && write.wrs == 4 && write.bytes == 128,
         "scatter WRITE must use grouping and actual WR accounting");
  std::fill(storage.begin(), storage.end(), 0);
  Stats read;
  Expect(RunBatchPass(&transport, false, &blocks, order, 16, 1, &read, 2) &&
             storage == expected && read_posts == 1 && read.wrs == 4,
         "READ must preserve the original shuffled mapping and single-SGE routing");
  Stats baseline;
  Expect(RunBatchPass(&transport, true, &blocks, order, 16, 1, &baseline) &&
             single_posts == 1 && scatter_posts == 2 && baseline.wrs == 4,
         "default WRITE must preserve the single-SGE baseline");

  std::vector<std::uint8_t> sparse_storage(128, 7);
  std::vector<HostBlock> sparse{{sparse_storage.data()}, {sparse_storage.data() + 32},
                               {sparse_storage.data() + 64}, {sparse_storage.data() + 96}};
  Stats sparse_write;
  Expect(RunBatchPass(&transport, true, &sparse, order, 16, 1, &sparse_write, 2) &&
             sparse_write.wrs == 4,
         "sparse host gaps must prevent grouping");
  return 0;
}
