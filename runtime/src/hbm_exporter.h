#pragma once
#include "types.h"
#include "status.h"
#include "offload_client.h"
#include "peer_link.h"
#include "npu_eid_discovery.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "direct_hccp/target_channel.h"
#include <acl/acl.h>
#include <chrono>
#include <cstdio>
#include <random>
#include <sstream>
#include <thread>

namespace kvaf::detail {
namespace h = direct_hccp;
namespace p = offload;
inline Status Error(const std::string& text) { return Status::Internal("Offload: " + text); }
inline std::string EidHex(const std::uint8_t* eid) {
  constexpr char digits[] = "0123456789abcdef";
  std::string text;
  text.reserve(32);
  for (std::size_t i = 0; i < 16; ++i) {
    text.push_back(digits[eid[i] >> 4]);
    text.push_back(digits[eid[i] & 15]);
  }
  return text;
}
inline std::uint64_t Nonce() {
  std::random_device r;
  const auto n = (std::uint64_t(r()) << 32) ^ r();
  return n ? n : 1;
}
inline bool ParseEid(const std::string& text, std::uint8_t* eid) {
  std::string hex;
  for (char c : text) if (c != ':') hex.push_back(c);
  if (hex.size() != 32) return false;
  auto digit = [](char c) -> int {
    if (c >= '0' && c <= '9') return c-'0';
    if (c >= 'a' && c <= 'f') return c-'a'+10;
    if (c >= 'A' && c <= 'F') return c-'A'+10;
    return -1;
  };
  bool nonzero = false;
  for (std::size_t i = 0; i < 16; ++i) {
    const int a = digit(hex[2*i]), b = digit(hex[2*i+1]);
    if (a < 0 || b < 0) return false;
    eid[i] = static_cast<std::uint8_t>(a*16+b); nonzero |= eid[i] != 0;
  }
  return nonzero;
}

class Exporter {
 public:
  p::Endpoint descriptor{};
  ~Exporter() { if (!Close().ok()) std::fprintf(stderr, "OFFLOAD: quarantined HBM export\n"); }
  Status Open(const ConstBufferView& input, const std::string& eid, const p::Endpoint& cpu,
              bool npu_peer = false) {
    std::uint32_t physical{};
    const auto input_base = reinterpret_cast<std::uint64_t>(input.data);
    if (!input.valid() || input.device != DeviceKind::kNpu || input.bytes > UINT64_MAX-input_base)
      return Status::InvalidArgument("invalid HBM export range");
    // Match BatchGet's HCCP registration: Tensor suballocations/views need
    // not start on a page boundary. Register the containing range, but keep
    // the public endpoint restricted to the Tensor itself.
    constexpr std::uint64_t kPageBytes = 4096;
    const auto registration_base = input_base & ~(kPageBytes-1);
    const auto input_offset = input_base-registration_base;
    const auto registration_bytes = input_offset+input.bytes;
    if (!cpu.token) return Error("CPU Allocate omitted region token; rebuild both Offload peers");
    if (!ParseEid(eid, descriptor.eid)) return Status::InvalidArgument("npu_eid must contain 32 hex digits");
    if (!h::LoadDirectHccp(&api_)) return Error("load Direct HCCP API");
    if (!h::GetCurrentNpuPhysicalDeviceId(&physical)) return Error("resolve NPU physical device");
    if (!h::AcquireDirectHccpSession(physical, 18, &session_))
      return Error("initialize Direct HCCP session: physical=" + std::to_string(physical));
    h::CtxInitCfg cfg{}; cfg.mode = h::kOfflineHdc;
    h::CtxInitAttr attr{}; attr.phy_id = physical;
    std::memcpy(attr.ub.eid.raw, descriptor.eid, 16);
    const int context_ret = api_.ctx_init(&cfg, &attr, &context_);
    if (context_ret || !context_) {
      std::ostringstream message;
      message << "create HBM context: RaCtxInit ret=" << context_ret
              << " context_nonnull=" << (context_ != nullptr) << " mode=" << cfg.mode
              << " physical=" << physical << " eid_index=" << attr.ub.eid_index
              << " npu_eid=" << EidHex(descriptor.eid);
      return Error(message.str());
    }
    h::TokenId token{};
    if (api_.token_alloc(context_, &token, &token_) || !token_) return Error("allocate source token");
    descriptor.token = cpu.token;
    h::MrRegInfo mr{};
    mr.in.mem.addr = registration_base; mr.in.mem.size = registration_bytes;
    // Match cpu_push_host.cc's deployed HBM contract (0x29c1).
    // Offload still submits only READ operations.
    mr.in.ub.flags = h::MemRegFlags(true);
    mr.in.ub.token_value = descriptor.token; mr.in.ub.token_id_handle = token_;
    const int register_ret = api_.lmem_register(context_, &mr, &mr_);
    if (register_ret || !mr_) {
      std::ostringstream message;
      message << "register source HBM: RaCtxLmemRegister ret=" << register_ret
              << " handle_nonnull=" << (mr_ != nullptr) << " physical=" << physical
              << " addr=0x" << std::hex << mr.in.mem.addr << " flags=0x" << mr.in.ub.flags
              << " input_addr=0x" << input_base
              << std::dec << " bytes=" << registration_bytes << " input_bytes=" << input.bytes
              << " input_offset=" << input_offset;
      return Error(message.str());
    }
    if (mr.out.key.size != 48)
      return Error("unsupported HBM segment ABI: key_bytes=" + std::to_string(mr.out.key.size) +
                   " expected=48 physical=" + std::to_string(physical));
    descriptor.segment_bytes = mr.out.key.size;
    std::memcpy(descriptor.segment, mr.out.key.value, mr.out.key.size);
    // urma_seg_t: packed UBVA {eid[16], uasid[4], va[8]}, aligned len at 32.
    std::uint64_t exported_base{}, exported_bytes{};
    std::memcpy(&exported_base, descriptor.segment + 20, 8);
    std::memcpy(&exported_bytes, descriptor.segment + 32, 8);
    const bool zero_base = exported_base == 0;
    const bool short_segment = exported_bytes < registration_bytes;
    const bool address_overflow = exported_base > UINT64_MAX-registration_bytes;
    // Match aicpu-urma's CPU-push export: the configured EID selects a CTP
    // route, while the segment carries the provider's memory identity. Do not
    // require equality or replace either EID with the other one.
    if (zero_base || short_segment || address_overflow) {
      std::ostringstream message;
      message << "invalid exported HBM range: zero_base=" << zero_base
              << " short_segment=" << short_segment << " address_overflow=" << address_overflow
              << " physical=" << physical
              << " eid_index=" << attr.ub.eid_index << " key_bytes=" << unsigned(mr.out.key.size)
              << " requested_base=0x" << std::hex << registration_base
              << " exported_base=0x" << exported_base << std::dec
              << " requested_bytes=" << registration_bytes << " exported_bytes=" << exported_bytes
              << " npu_eid=" << EidHex(descriptor.eid)
              << " segment_eid=" << EidHex(descriptor.segment);
      return Error(message.str());
    }
    descriptor.base = exported_base+input_offset;
    descriptor.bytes = input.bytes;
    std::fprintf(stderr, "[KVAF_NPU] HBM_EXPORT_RANGE input_base=0x%llx input_bytes=%llu "
        "registered_base=0x%llx registered_bytes=%llu input_offset=%llu exported_base=0x%llx "
        "exported_bytes=%llu source_base=0x%llx\n",
        (unsigned long long)input_base, (unsigned long long)input.bytes,
        (unsigned long long)registration_base, (unsigned long long)registration_bytes,
        (unsigned long long)input_offset, (unsigned long long)exported_base,
        (unsigned long long)exported_bytes, (unsigned long long)descriptor.base);
    descriptor.psn = 1; // cpu_push_host.cc initializes npu_psn to 1.
    tp_cfg_.flag = h::CtpFlag(); tp_cfg_.transport_mode = h::kReliableMessage;
    std::memcpy(tp_cfg_.local_eid.raw, descriptor.eid, 16);
    std::memcpy(tp_cfg_.peer_eid.raw, cpu.eid, 16);
    if (api_.get_tp_info_async(context_, &tp_cfg_, &tp_info_, &tp_count_, &async_) || !async_)
      return Error("submit source CTP query");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (async_) {
      auto status = PollTp();
      if (status.code() != StatusCode::kBusy) {
        if (!status.ok()) return status;
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) return Status::InDoubt("source CTP query is pending");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Query CTP before creating the target channels, as in the reference.
    for (std::size_t index = 0; index < channels_.size(); ++index) {
      auto& channel = channels_[index];
      channel = std::make_unique<h::TargetChannel>(api_, context_);
      const auto status = channel->Initialize(token_, descriptor.token, 1024, 1024, 16384);
      if (!status.ok()) return status;
      const auto& key = channel->jetty_info().key;
      if (key.size != 32) return Error("unsupported NPU Jetty ABI");
      descriptor.key_bytes = key.size;
      std::memcpy(descriptor.key[index], key.value, key.size);
    }
    std::fprintf(stderr, "[KVAF_NPU] %s commit=8bc5a2a protocol=%u jettys=%u "
        "physical=%u npu_eid=%s cpu_eid=%s hbm_flags=0x%x psn=%u\n",
        npu_peer ? "HBM_READ_EXPORT" : "CPU_PULL_REFERENCE",
        p::kVersion, p::kChannelCount, physical, EidHex(descriptor.eid).c_str(), EidHex(cpu.eid).c_str(),
        mr.in.ub.flags, descriptor.psn);
    return Status::OK();
  }
  Status Bind(const p::Endpoint& cpu) {
    if (cpu.token != descriptor.token) return Error("CPU Jetty token differs from allocated region token");
    if (cpu.key_bytes != 28 || !cpu.tp || !cpu.psn) return Error("invalid CPU initiator");
    for (std::size_t index = 0; index < channels_.size(); ++index) {
      h::QpImportInfoT info{};
      if (!h::ImportPeerJetty(api_, context_, cpu.key[index], cpu.key_bytes,
          cpu.token, descriptor.tp, cpu.tp, descriptor.psn, cpu.psn, "cpu_pull",
          &info, channels_[index]->peer_jetty_output())) return Error("import CPU initiator");
    }
    return Status::OK();
  }
  // NPU initiators export 32-byte keys, unlike the CPU's 28-byte keys.
  // Keep both provider identities opaque and use route EIDs only for CTP.
  Status BindNpuChannel(std::size_t index, const p::Endpoint& peer) {
    if (index >= channels_.size() || peer.key_bytes != 32 ||
        peer.token != descriptor.token || !peer.tp || !peer.psn)
      return Error("invalid NPU READ initiator");
    // A's BatchGet requests one CTP for every initiator Jetty. Keep the
    // corresponding B CTP per channel rather than reusing the CPU-pull TP.
    if (!npu_tp_handles_[index]) {
      if (index == 0) {
        npu_tp_handles_[index] = descriptor.tp;
      } else {
        tp_info_ = {}; tp_count_ = 1;
        if (api_.get_tp_info_async(context_, &tp_cfg_, &tp_info_, &tp_count_, &async_) || !async_)
          return Error("submit NPU channel CTP query");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (async_) {
          auto status = PollTp();
          if (status.code() != StatusCode::kBusy) {
            if (!status.ok()) return status;
            break;
          }
          if (std::chrono::steady_clock::now() >= deadline)
            return Status::InDoubt("NPU channel CTP query pending");
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        npu_tp_handles_[index] = descriptor.tp;
      }
    }
    h::QpImportInfoT info{};
    if (!h::ImportPeerJetty(api_, context_, peer.key[0], peer.key_bytes,
        peer.token, npu_tp_handles_[index], peer.tp, descriptor.psn, peer.psn,
        "npu_raw_read_target", &info, channels_[index]->peer_jetty_output()))
      return Error("import NPU READ initiator");
    return Status::OK();
  }
  p::Endpoint NpuChannelDescriptor(std::size_t index) const {
    auto result = descriptor;
    if (index < npu_tp_handles_.size()) result.tp = npu_tp_handles_[index];
    return result;
  }
  static Status Ready(void* stream) {
    if (!stream) return Status::InvalidArgument("Offload requires a producer ACL stream");
    aclrtEvent event{};
    if (aclrtCreateEvent(&event) != ACL_SUCCESS) return Error("create producer event");
    const auto record = aclrtRecordEvent(event, static_cast<aclrtStream>(stream));
    const auto wait = record == ACL_SUCCESS ? aclrtSynchronizeEvent(event) : record;
    const auto destroy = aclrtDestroyEvent(event);
    return wait == ACL_SUCCESS && destroy == ACL_SUCCESS ? Status::OK() : Error("wait producer event");
  }
  Status Close() {
    if (async_) { auto s = PollTp(); if (s.code() == StatusCode::kBusy) return Status::InDoubt("source CTP pending"); }
#define RELEASE(handle, operation) \
    if (handle) { if (api_.operation(context_, handle)) return Status::InDoubt(#operation); handle = nullptr; }
    for (auto& channel : channels_) if (channel) {
      const auto status = channel->Shutdown();
      if (!status.ok()) return Status::InDoubt(status.message());
      channel.reset();
    }
    RELEASE(mr_, lmem_unregister)
    RELEASE(token_, token_free)
#undef RELEASE
    if (context_) { if (api_.ctx_deinit(context_)) return Status::InDoubt("destroy source context"); context_ = nullptr; }
    if (session_) { h::ReleaseDirectHccpSession(session_); session_ = nullptr; }
    h::CloseDirectHccp(&api_);
    return Status::OK();
  }
 private:
  Status PollTp() {
    int result{}; const int rc = api_.get_async_req_result(async_, &result);
    if (rc == 128301) return Status::Busy("CTP pending");
    async_ = nullptr;
    if (rc || result || !tp_count_ || !tp_info_.tp_handle) return Error("source CTP query failed");
    descriptor.tp = tp_info_.tp_handle; return Status::OK();
  }
  h::DirectHccpApi api_{};
  h::DirectHccpSession* session_{};
  void *context_{}, *token_{}, *mr_{}, *async_{};
  std::array<std::unique_ptr<h::TargetChannel>, p::kChannelCount> channels_;
  std::array<std::uint64_t, p::kChannelCount> npu_tp_handles_{};
  h::GetTpCfg tp_cfg_{};
  h::TpInfo tp_info_{};
  std::uint32_t tp_count_{1};
};

}
