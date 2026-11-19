#pragma once
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include "batch_get_raw_addr_abi.h"

namespace kvaf::raw_read_debug {
// Raw CQE fields are not urma_cr_status_t enum values. This mapping is from
// UMDK udma_u_jfc.c, udma_u_get_cr_status(), indexed by status/substatus.
inline const char* UdmaStatusName(std::uint32_t status,std::uint32_t sub_status) {
  return status==2 && sub_status==2 ? "URMA_CR_LOC_ACCESS_ERR" : "unmapped";
}

inline std::string FormatWqe(std::uint32_t core,std::uint32_t jetty,std::uint32_t entry,
                             const KvafBatchGetDirectUrmaEntity& entity,const std::uint32_t* words) {
  const auto local=std::uint64_t(words[14]) | (std::uint64_t(words[15])<<32);
  const auto remote=std::uint64_t(words[10]) | (std::uint64_t(words[11])<<32);
  const auto bytes=words[12];
  auto range=[&](std::uint64_t base,std::uint64_t size,std::uint64_t address) {
    return bytes && address>=base && bytes<=size && address-base<=size-bytes;
  };
  std::ostringstream out;
  out << "[KVAF_NPU] HBM_READ_WQE_DIAGNOSTIC core=" << core << " jetty=" << jetty
      << " entry=" << entry << " snapshot_source=aiv_status opcode=" << ((words[1]>>8)&255) << " sge_count=" << (words[2]>>24)
      << " bytes=" << bytes << " local_address=0x" << std::hex << local
      << " local_sge_token_id=0x" << words[13] << " expected_local_token_id=0x" << entity.local_token_id
      << " remote_address=0x" << remote << " remote_token_id=0x" << (words[3]&0xfffffU) << std::dec
      << " local_range_valid=" << range(entity.local_hbm_address,entity.local_hbm_size,local)
      << " local_token_matches=" << (words[13]==entity.local_token_id)
      << " remote_range_valid=" << range(entity.remote_host_ddr_address,entity.remote_host_ddr_size,remote)
      << " remote_token_matches=" << ((words[3]&0xfffffU)==entity.remote_token_id)
      << " remote_eid_matches=" << (std::memcmp(words+4,entity.remote_eid,16)==0)
      << " raw_words=" << std::hex << std::setfill('0');
  for(unsigned word=0;word<16;++word) {if(word)out << ',';out << std::setw(8) << words[word];}
  out << std::dec << std::setfill(' ') << '\n';
  return out.str();
}

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
    if (status.reserved[0] == KVAF_RAW_CQE_DIAGNOSTIC_MAGIC ||
        status.reserved[0] == KVAF_RAW_CQE_WQE_DIAGNOSTIC_MAGIC) {
      const auto word0 = static_cast<std::uint32_t>(status.reserved[2]);
      out << "[KVAF_NPU] HBM_READ_CQE_DIAGNOSTIC core=" << core << " jetty=" << status.reserved[1]
          << " status=" << ((word0 >> 24U) & 255U) << " sub_status=" << ((word0 >> 16U) & 255U)
          << " udma_status_name=" << UdmaStatusName((word0>>24)&255,(word0>>16)&255)
          << " raw_words=" << std::hex << std::setfill('0');
      for (std::uint32_t word = 0; word < 8; ++word) {
        if (word) out << ',';
        out << std::setw(8) << static_cast<std::uint32_t>(status.reserved[2 + word / 2] >> (32 * (word % 2)));
      }
      out << std::dec << std::setfill(' ') << '\n';
      const auto jetty = status.reserved[1];
      const auto entry = static_cast<std::uint32_t>(status.reserved[2] >> 32U) & 0xffffU;
      if (status.reserved[0] == KVAF_RAW_CQE_WQE_DIAGNOSTIC_MAGIC && jetty < args.jettys_per_core) {
        std::uint32_t words[16];
        for (unsigned word=0;word<16;++word)
          words[word]=static_cast<std::uint32_t>(status.execution.profile.jetty_completion_wait_cycles[word/2] >> (32*(word%2)));
        out << FormatWqe(core,static_cast<std::uint32_t>(jetty),entry,entities[core*args.jettys_per_core+jetty],words);
      } else {
        out << "[KVAF_NPU] HBM_READ_WQE_DIAGNOSTIC core=" << core << " jetty=" << jetty
            << " entry=" << entry << " snapshot_unavailable=1 snapshot_source=aiv_status hint=check_matched_raw_kernel_build\n";
      }
    } else if (execution.result == 7) {
      out << "[KVAF_NPU] HBM_READ_CQE_DIAGNOSTIC core=" << core
          << " snapshot_written=0 hint=check_matched_raw_kernel_build\n";
    }
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
          << " local_token_id=" << entity.local_token_id << " remote_token_id=" << entity.remote_token_id
          << " jetty_id=" << entity.jetty_id << " tpn=" << entity.tp_id << " wqe_remote_eid="
          << std::hex << std::setfill('0');
      for (auto byte : entity.remote_eid) out << std::setw(2) << static_cast<unsigned>(byte);
      out << std::dec << std::setfill(' ') << '\n';
    }
  }
  return out.str();
}
}  // namespace kvaf::raw_read_debug
