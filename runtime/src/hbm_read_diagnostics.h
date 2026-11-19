#pragma once
#include <cstdint>
#include <sstream>
#include <string>
#include "batch_get_raw_addr_abi.h"

namespace kvaf::raw_read_debug {
inline const char* ResultName(std::uint64_t result) {
  switch (result) {
    case 0: return "success";
    case 1: return "invalid_args";
    case 2: return "invalid_entity";
    case 3: return "invalid_queue";
    case 4: return "invalid_source_or_length";
    case 5: return "queue_full";
    case 6: return "cq_timeout";
    case 7: return "cq_error";
    case 8: return "invalid_output";
    default: return "unknown";
  }
}

inline bool AllRequestsCompleted(const KvafExecuteBatchGetRawAddrKernelArgs& args,
                                 const KvafRawReadCoreStatus* statuses) {
  for (std::uint32_t core = 0; core < args.block_count; ++core) {
    const auto& status = statuses[core];
    const auto expected = std::uint64_t(args.request_count) * (core + 1) / args.block_count -
                          std::uint64_t(args.request_count) * core / args.block_count;
    if (status.execution.magic != KVAF_AIV_BATCH_GET_STATUS_MAGIC || status.execution.result != 0 ||
        status.execution.processed_request_count != expected || status.posted_wqes != expected ||
        status.total_wqes != args.request_count) return false;
  }
  return args.block_count != 0;
}

// Snapshots are copied only after stream completion in diagnostic mode.
// Printing a successful status does not establish payload correctness.
inline std::string Format(const KvafExecuteBatchGetRawAddrKernelArgs& args,
                          const KvafRawReadCoreStatus* statuses,
                          const KvafBatchGetQueueState* queues,
                          const KvafBatchGetDirectUrmaEntity* entities) {
  std::ostringstream out;
  out << "[KVAF_NPU] HBM_READ_KERNEL_DIAGNOSTIC input_mode=" << args.input_mode
      << " requests=" << args.request_count << " cores=" << args.block_count
      << " jettys_per_core=" << args.jettys_per_core << " validate_ranges=" << args.validate_ranges
      << " output=0x" << std::hex << args.output << " entity_table=0x" << args.channel_entity
      << " src_metadata=0x" << args.source_addresses << " dst_metadata=0x" << args.destination_addresses
      << " lengths_metadata=0x" << args.lengths << std::dec << '\n';
  for (std::uint32_t core = 0; core < args.block_count; ++core) {
    const auto& status = statuses[core];
    const auto& execution = status.execution;
    const auto expected = std::uint64_t(args.request_count) * (core + 1) / args.block_count -
                          std::uint64_t(args.request_count) * core / args.block_count;
    out << "[KVAF_NPU] HBM_READ_CORE_DIAGNOSTIC core=" << core
        << " status_written=" << (execution.magic == KVAF_AIV_BATCH_GET_STATUS_MAGIC)
        << " result=" << execution.result << " result_name=" << ResultName(execution.result)
        << " processed=" << execution.processed_request_count << " expected=" << expected
        << " total_wqes=" << status.total_wqes << " posted_wqes=" << status.posted_wqes
        << " failed_request=" << execution.failed_request_index
        << " sq_pi=" << execution.sq_producer_index << " cq_ci=" << execution.cq_consumer_index << '\n';
    for (std::uint32_t jetty = 0; jetty < args.jettys_per_core; ++jetty) {
      const auto index = core * args.jettys_per_core + jetty;
      const auto& queue = queues[index];
      const auto& entity = entities[index];
      out << "[KVAF_NPU] HBM_READ_QUEUE_DIAGNOSTIC core=" << core << " jetty=" << jetty
          << " initialized=" << queue.initialized << " poisoned=" << queue.poisoned
          << " last_error=" << queue.last_error << " sq_pi=" << queue.sq_producer_index
          << " cq_ci=" << queue.cq_consumer_index << " completed_batches=" << queue.completed_batch_count << '\n';
      out << "[KVAF_NPU] HBM_READ_ENTITY_DIAGNOSTIC core=" << core << " jetty=" << jetty
          << " magic=0x" << std::hex << entity.magic << std::dec << " version=" << entity.version
          << " output_matches=" << (entity.local_hbm_address == args.output)
          << " local_base=0x" << std::hex << entity.local_hbm_address << std::dec
          << " local_bytes=" << entity.local_hbm_size
          << " remote_base=0x" << std::hex << entity.remote_host_ddr_address << std::dec
          << " remote_bytes=" << entity.remote_host_ddr_size
          << " sq_depth=" << entity.sq_depth << " cq_depth=" << entity.cq_depth
          << " local_token_id=" << entity.local_token_id << " remote_token_id=" << entity.remote_token_id << '\n';
    }
  }
  return out.str();
}
}  // namespace kvaf::raw_read_debug
