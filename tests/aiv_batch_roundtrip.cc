#include "aiv_batch_roundtrip_options.h"

#include "acl/acl.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "kvaf_aiv_batch_get.h"
#include "kvaf_aiv_batch_put.h"
#include "npu_eid_discovery.h"
#include "peer_link.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using kvaf::aiv_batch_roundtrip::Options;

void RequireAcl(aclError result, const char* operation) {
  if (result != ACL_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed acl=" + std::to_string(result));
}

void RequireResult(int32_t result, const char* operation) {
  if (result != 0)
    throw std::runtime_error(std::string(operation) + " failed result=" + std::to_string(result));
}

void RequireStatus(const kvaf::Status& status, const char* operation) {
  if (!status.ok()) throw std::runtime_error(std::string(operation) + ": " + status.message());
}

struct AclContext {
  explicit AclContext(std::uint32_t device_id) : device_id(device_id) {
    RequireAcl(aclInit(nullptr), "aclInit");
    try {
      RequireAcl(aclrtSetDevice(device_id), "aclrtSetDevice");
      active = true;
      RequireAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
    } catch (...) {
      if (active) (void)aclrtResetDevice(device_id);
      (void)aclFinalize();
      throw;
    }
  }
  ~AclContext() {
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    if (active) (void)aclrtResetDevice(device_id);
    (void)aclFinalize();
  }
  AclContext(const AclContext&) = delete;
  AclContext& operator=(const AclContext&) = delete;
  std::uint32_t device_id;
  bool active{false};
  aclrtStream stream{nullptr};
};

struct DeviceBuffer {
  explicit DeviceBuffer(std::size_t bytes) : bytes(bytes) {
    RequireAcl(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    if (data == nullptr) throw std::runtime_error("aclrtMalloc returned null");
  }
  ~DeviceBuffer() { if (data != nullptr) (void)aclrtFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void Upload(const void* source) const {
    RequireAcl(aclrtMemcpy(data, bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
  }
  void Download(void* destination) const {
    RequireAcl(aclrtMemcpy(destination, bytes, data, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
  }
  void Clear() const { RequireAcl(aclrtMemset(data, bytes, 0, bytes), "aclrtMemset"); }

  void* data{nullptr};
  std::size_t bytes;
};

struct Buffers {
  explicit Buffers(const Options& options)
      : payload_bytes(static_cast<std::size_t>(options.blocks) * options.block_bytes),
        token_count(payload_bytes / options.token_bytes),
        input(payload_bytes), output(payload_bytes),
        sources(sizeof(KvafBatchPutSourceDescriptor)),
        source_indices(options.blocks * sizeof(std::uint32_t)),
        input_offsets(options.blocks * sizeof(std::uint64_t)),
        lengths(options.blocks * sizeof(std::uint64_t)),
        destinations(options.blocks * sizeof(std::uint64_t)),
        put_status(sizeof(KvafBatchPutCoreStatus)),
        addresses(options.blocks * sizeof(std::uint64_t)), valid(options.blocks),
        indices(token_count * sizeof(std::int32_t)),
        offsets(token_count * sizeof(std::int32_t)), mask(token_count),
        output_indices(token_count * sizeof(std::int32_t)),
        get_status(sizeof(KvafBatchGetCoreStatus)),
        get_queue(sizeof(KvafBatchGetQueueState)),
        host_input(payload_bytes), host_output(payload_bytes) {
    const std::uint32_t tokens_per_block = options.block_bytes / options.token_bytes;
    std::vector<std::uint32_t> host_source_indices(options.blocks, 0U);
    std::vector<std::uint64_t> host_input_offsets(options.blocks);
    std::vector<std::uint64_t> host_lengths(options.blocks, options.block_bytes);
    std::vector<std::int32_t> host_indices(token_count);
    std::vector<std::int32_t> host_offsets(token_count);
    std::vector<std::int32_t> host_output_indices(token_count);
    std::vector<std::uint8_t> host_valid(options.blocks, 1U);
    std::vector<std::uint8_t> host_mask(token_count, 1U);
    for (std::uint32_t block = 0U; block < options.blocks; ++block)
      host_input_offsets[block] = static_cast<std::uint64_t>(block) * options.block_bytes;
    for (std::size_t row = 0U; row < token_count; ++row) {
      host_indices[row] = static_cast<std::int32_t>(row / tokens_per_block);
      host_offsets[row] = static_cast<std::int32_t>((row % tokens_per_block) * options.token_bytes);
      host_output_indices[row] = static_cast<std::int32_t>(token_count - 1U - row);
    }
    source_indices.Upload(host_source_indices.data());
    input_offsets.Upload(host_input_offsets.data());
    lengths.Upload(host_lengths.data());
    valid.Upload(host_valid.data());
    indices.Upload(host_indices.data());
    offsets.Upload(host_offsets.data());
    mask.Upload(host_mask.data());
    output_indices.Upload(host_output_indices.data());
  }

  void SetRemote(std::uint64_t base, const Options& options) const {
    std::vector<std::uint64_t> host_addresses(options.blocks);
    for (std::uint32_t block = 0U; block < options.blocks; ++block)
      host_addresses[block] = base + static_cast<std::uint64_t>(block) * options.block_bytes;
    addresses.Upload(host_addresses.data());
    destinations.Upload(host_addresses.data());
  }

  void Fill(std::uint32_t round) {
    for (std::size_t byte = 0U; byte < payload_bytes; ++byte)
      host_input[byte] = static_cast<std::uint8_t>((byte * 131U + (byte >> 8U) * 17U + round * 29U) & 0xffU);
    input.Upload(host_input.data());
    output.Clear();
  }

  bool Verify(const Options& options, std::string* error) {
    output.Download(host_output.data());
    for (std::size_t row = 0U; row < token_count; ++row) {
      const std::size_t source_offset = (token_count - 1U - row) * options.token_bytes;
      const std::size_t output_offset = row * options.token_bytes;
      for (std::size_t byte = 0U; byte < options.token_bytes; ++byte) {
        if (host_output[output_offset + byte] != host_input[source_offset + byte]) {
          *error = "payload mismatch row=" + std::to_string(row) +
                   " byte=" + std::to_string(byte);
          return false;
        }
      }
    }
    return true;
  }

  std::size_t payload_bytes;
  std::size_t token_count;
  DeviceBuffer input, output, sources, source_indices, input_offsets, lengths, destinations, put_status;
  DeviceBuffer addresses, valid, indices, offsets, mask, output_indices, get_status, get_queue;
  std::vector<std::uint8_t> host_input, host_output;
};

struct Region {
  explicit Region(kvaf::peer::PeerLink& link) : link(link) {}
  ~Region() {
    if (value.base != 0U) {
      const kvaf::Status status = value.device_memory
          ? link.ReleaseHbmRegion(value.session_id) : link.ReleaseRegion(value.base);
      if (!status.ok()) std::fprintf(stderr, "[KVAF_AIV_TEST] RELEASE_FAILED reason=%s\n", status.message().c_str());
    }
  }
  kvaf::peer::PeerLink& link;
  kvaf::peer::PeerRegion value{};
};

struct PutTransport {
  ~PutTransport() { if (value != nullptr) KvafBatchPutTransportDestroy(value); }
  KvafBatchPutTransport* value{nullptr};
};

struct GetTransport {
  ~GetTransport() { if (value != nullptr) KvafBatchGetTransportDestroy(value); }
  KvafBatchGetTransport* value{nullptr};
};

struct PutLocalBuffer {
  ~PutLocalBuffer() { if (value != nullptr) KvafBatchPutLocalBufferRelease(value); }
  KvafBatchPutLocalBuffer* value{nullptr};
};

struct GetLocalBuffer {
  ~GetLocalBuffer() { if (value != nullptr) KvafBatchGetLocalBufferRelease(value); }
  KvafBatchGetLocalBuffer* value{nullptr};
};

std::string EidHex(const std::array<std::uint8_t, kvaf::peer::kEidBytes>& eid) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(eid.size() * 2U);
  for (const std::uint8_t byte : eid) {
    result.push_back(digits[byte >> 4U]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

bool Attempt(kvaf::peer::PeerLink& link, std::uint32_t peer_candidate,
             const std::array<std::uint8_t, kvaf::peer::kEidBytes>& local_eid,
             const Options& options, Buffers& buffers, aclrtStream stream) {
  Region region(link);
  const kvaf::Status allocated = options.peer_device == "npu"
      ? link.AllocateHbmRegion(buffers.payload_bytes, options.block_bytes,
                               options.peer_device_id, peer_candidate, &region.value)
      : link.AllocateRegion(buffers.payload_bytes, options.block_bytes, peer_candidate, &region.value);
  if (!allocated.ok()) {
    std::fprintf(stderr, "[KVAF_AIV_TEST] ALLOCATE_FAILED reason=%s\n", allocated.message().c_str());
    return false;
  }
  PutTransport put;
  GetTransport get;
  const int32_t put_create = KvafBatchPutTransportCreateWithPeerLink(
      reinterpret_cast<void*>(region.value.base), region.value.length, 1U, 1U,
      &link, local_eid.data(), &put.value);
  const int32_t get_create = KvafBatchGetTransportCreateWithPeerLink(
      reinterpret_cast<void*>(region.value.base), region.value.length, 1U, 1U,
      &link, local_eid.data(), &get.value);
  if (put_create != 0 || get_create != 0 || put.value == nullptr || get.value == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV_TEST] TRANSPORT_FAILED put=%d get=%d\n", put_create, get_create);
    return false;
  }
  PutLocalBuffer source;
  GetLocalBuffer target;
  std::uint64_t get_entity = 0U;
  const int32_t source_result = KvafBatchPutLocalBufferAcquire(
      put.value, buffers.input.data, buffers.payload_bytes, &source.value);
  const int32_t target_result = KvafBatchGetLocalBufferAcquire(
      get.value, buffers.output.data, buffers.payload_bytes, &target.value, &get_entity);
  if (source_result != 0 || target_result != 0 || source.value == nullptr ||
      target.value == nullptr || get_entity == 0U) {
    std::fprintf(stderr, "[KVAF_AIV_TEST] HBM_REGISTER_FAILED put=%d get=%d\n",
                 source_result, target_result);
    return false;
  }
  KvafBatchPutSourceDescriptor descriptor{};
  RequireResult(KvafBatchPutLocalBufferDescribe(source.value, &descriptor), "describe Put HBM");
  buffers.sources.Upload(&descriptor);
  buffers.SetRemote(region.value.base, options);
  buffers.get_queue.Clear();

  KvafExecuteBatchPutKernelArgs put_args{};
  put_args.source_descriptors = reinterpret_cast<std::uint64_t>(buffers.sources.data);
  put_args.source_indices = reinterpret_cast<std::uint64_t>(buffers.source_indices.data);
  put_args.input_offsets = reinterpret_cast<std::uint64_t>(buffers.input_offsets.data);
  put_args.lengths = reinterpret_cast<std::uint64_t>(buffers.lengths.data);
  put_args.destination_addresses = reinterpret_cast<std::uint64_t>(buffers.destinations.data);
  put_args.core_status = reinterpret_cast<std::uint64_t>(buffers.put_status.data);
  put_args.row_count = options.blocks;
  put_args.aiv_core_count = 1U;

  KvafExecuteBatchGetKernelArgs get_args{};
  get_args.address_table = reinterpret_cast<std::uint64_t>(buffers.addresses.data);
  get_args.valid_table = reinterpret_cast<std::uint64_t>(buffers.valid.data);
  get_args.indices = reinterpret_cast<std::uint64_t>(buffers.indices.data);
  get_args.offsets = reinterpret_cast<std::uint64_t>(buffers.offsets.data);
  get_args.mask = reinterpret_cast<std::uint64_t>(buffers.mask.data);
  get_args.output_indices = reinterpret_cast<std::uint64_t>(buffers.output_indices.data);
  get_args.output = reinterpret_cast<std::uint64_t>(buffers.output.data);
  get_args.core_status = reinterpret_cast<std::uint64_t>(buffers.get_status.data);
  get_args.channel_entity = get_entity;
  get_args.queue_state = reinterpret_cast<std::uint64_t>(buffers.get_queue.data);
  get_args.table_capacity = options.blocks;
  get_args.request_capacity = static_cast<std::uint32_t>(buffers.token_count);
  get_args.output_rows = static_cast<std::uint32_t>(buffers.token_count);
  get_args.token_bytes = options.token_bytes;
  get_args.tokens_per_block = options.block_bytes / options.token_bytes;
  get_args.block_count = 1U;
  get_args.jettys_per_core = 1U;

  for (std::uint32_t round = 0U; round < options.iterations; ++round) {
    buffers.Fill(round);
    KvafBatchPutCoreStatus put_status{};
    const int32_t put_result = KvafExecuteBatchPutSync(put.value, &put_args, stream, &put_status, 1U);
    std::fprintf(stderr,
                 "[KVAF_AIV_TEST] PUT round=%u result=%d magic=0x%llx device_result=%llu processed=%llu "
                 "cq_status=%llu cq_sub_status=%llu cqe_word=0x%llx sq_pi=%llu cq_ci=%llu\n",
                 round, put_result, static_cast<unsigned long long>(put_status.magic),
                 static_cast<unsigned long long>(put_status.result),
                 static_cast<unsigned long long>(put_status.processed_request_count),
                 static_cast<unsigned long long>(put_status.cq_status),
                 static_cast<unsigned long long>(put_status.cq_sub_status),
                 static_cast<unsigned long long>(put_status.reserved[0]),
                 static_cast<unsigned long long>(put_status.sq_producer_index),
                 static_cast<unsigned long long>(put_status.cq_consumer_index));
    if (put_result != 0 && put_result != KVAF_BATCH_PUT_KERNEL_ERROR) return false;

    buffers.get_status.Clear();
    const int32_t get_result = KvafExecuteBatchGetLaunch(&get_args, stream);
    if (get_result != 0) {
      std::fprintf(stderr, "[KVAF_AIV_TEST] GET_LAUNCH_FAILED round=%u result=%d\n", round, get_result);
      return false;
    }
    RequireAcl(aclrtSynchronizeStream(stream), "BatchGet synchronize");
    KvafBatchGetCoreStatus get_status{};
    buffers.get_status.Download(&get_status);
    const bool get_ok = get_status.magic == KVAF_AIV_BATCH_GET_STATUS_MAGIC &&
                        get_status.result == 0U &&
                        get_status.processed_request_count == buffers.token_count &&
                        get_status.completed_batch_count != 0U;
    std::fprintf(stderr,
                 "[KVAF_AIV_TEST] GET round=%u magic=0x%llx result=%llu processed=%llu failed_request=%llu "
                 "sq_pi=%llu cq_ci=%llu last_wqe=0x%llx\n",
                 round, static_cast<unsigned long long>(get_status.magic),
                 static_cast<unsigned long long>(get_status.result),
                 static_cast<unsigned long long>(get_status.processed_request_count),
                 static_cast<unsigned long long>(get_status.failed_request_index),
                 static_cast<unsigned long long>(get_status.sq_producer_index),
                 static_cast<unsigned long long>(get_status.cq_consumer_index),
                 static_cast<unsigned long long>(get_status.last_wqe_protocol));
    if (put_result != 0 || !get_ok) return false;
    std::string verification_error;
    if (!buffers.Verify(options, &verification_error)) {
      std::fprintf(stderr, "[KVAF_AIV_TEST] VERIFY_FAILED round=%u reason=%s\n",
                   round, verification_error.c_str());
      return false;
    }
    std::fprintf(stderr, "[KVAF_AIV_TEST] GET_VERIFIED round=%u bytes=%zu\n",
                 round, buffers.payload_bytes);
  }
  return true;
}

int Run(const Options& options) {
  AclContext acl(options.device_id);
  std::uint32_t physical_device_id = 0U;
  if (!kvaf::direct_hccp::GetCurrentNpuPhysicalDeviceId(&physical_device_id))
    throw std::runtime_error("cannot resolve NPU physical device id");
  std::vector<std::array<std::uint8_t, kvaf::peer::kEidBytes>> local_eids;
  std::string discovery_error;
  if (!kvaf::npu_eid_detail::DiscoverDeviceEids(physical_device_id, &local_eids, &discovery_error))
    throw std::runtime_error("NPU EID discovery failed: " + discovery_error);
  RequireResult(KvafExecuteBatchPutLoadKernel(options.put_kernel.c_str()), "load AIV BatchPut kernel");
  RequireResult(KvafExecuteBatchGetLoadKernel(options.get_kernel.c_str()), "load AIV BatchGet kernel");
  kvaf::peer::PeerLink link(options.peer);
  std::uint32_t peer_count = 0U;
  if (options.peer_device == "npu") {
    std::vector<std::array<std::uint8_t, kvaf::peer::kEidBytes>> candidates;
    RequireStatus(link.GetHbmEidCandidates(options.peer_device_id, &candidates), "NPU candidate discovery");
    peer_count = static_cast<std::uint32_t>(candidates.size());
  } else {
    RequireStatus(link.GetCandidateCount(&peer_count), "CPU candidate discovery");
  }
  Buffers buffers(options);
  for (std::uint32_t peer_candidate = 0U; peer_candidate < peer_count; ++peer_candidate) {
    for (std::size_t npu_candidate = 0U; npu_candidate < local_eids.size(); ++npu_candidate) {
      std::fprintf(stderr,
                   "[KVAF_AIV_TEST] ATTEMPT peer_candidate=%u/%u npu_candidate=%zu/%zu npu_eid=%s\n",
                   peer_candidate, peer_count, npu_candidate, local_eids.size(),
                   EidHex(local_eids[npu_candidate]).c_str());
      if (Attempt(link, peer_candidate, local_eids[npu_candidate], options, buffers, acl.stream)) {
        std::printf("AIV_BATCH_ROUNDTRIP_PASS device=%u peer_candidate=%u npu_candidate=%zu "
                    "blocks=%u block_bytes=%u token_bytes=%u iterations=%u peer_device=%s peer_device_id=%u\n",
                    options.device_id, peer_candidate, npu_candidate, options.blocks,
                    options.block_bytes, options.token_bytes, options.iterations,
                    options.peer_device.c_str(), options.peer_device_id);
        return 0;
      }
    }
  }
  std::fprintf(stderr, "[KVAF_AIV_TEST] FAIL no peer/local NPU EID pair passed Put and Get\n");
  return 1;
}

void Usage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s [--device-id N] [--peer HOST:PORT] [--blocks N] "
               "[--peer-device cpu|npu] [--peer-device-id N] "
               "[--block-bytes N] [--token-bytes N] [--iterations N] "
               "[--put-kernel PATH] [--get-kernel PATH]\n"
               "Defaults: device=0 peer=127.0.0.1:28999 blocks=1 block-bytes=4096 "
               "token-bytes=256 iterations=1 peer-device=cpu peer-device-id=0. "
               "Local and peer EIDs are discovered automatically.\n",
               program);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    Usage(argv[0]);
    return 0;
  }
  Options options;
  const std::vector<const char*> arguments(argv, argv + argc);
  if (!kvaf::aiv_batch_roundtrip::Parse(argc, arguments.data(), &options)) {
    Usage(argv[0]);
    return 2;
  }
  try {
    if (!kvaf::aiv_batch_roundtrip::ConfigureCannEnvironment())
      throw std::runtime_error("cannot configure ASCEND_HOME_PATH");
    return Run(options);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[KVAF_AIV_TEST] FAIL %s\n", error.what());
    return 1;
  }
}
