// Real CPU SGL WRITE transport with a synchronous fake URMA provider.
// Requires the public UMDK headers, no URMA library or hardware.
#include "cpu_service/push/cpu_push_get_transport.h"
#include "urma_device_abi.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
using kvaf::urma::Sge;
std::uint32_t device_sges = 13, queue_sges{}, data_wrs{}, data_sges{}, segments{};
std::uint64_t output_base{}, output_bytes{}, done_address{};
std::uint32_t ready_wrs{}, data_tails{};
bool delay_output_import{};
std::function<void()> publish_suffix;
std::uint64_t head_bytes{}, head_written{}, suffix_descriptor{};
bool saw_async_suffix_read{};
std::map<urma_jfc_t*, std::deque<std::uint64_t>> completions;
int QueryTp(urma_context_t*, kvaf::urma::TpQuery*, std::uint32_t* count, kvaf::urma::TpInfo* info) {
  *count = 1; info->tp_handle = 1; return 0;
}
urma_target_jetty_t* ImportJetty(urma_context_t*, kvaf::urma::RemoteJettyEx*, urma_token_t*,
                                kvaf::urma::ActiveTp*) {
  return new urma_target_jetty_t{};
}
}  // namespace

extern "C" {
urma_device_t** urma_get_device_list(int*) { return nullptr; }
void urma_free_device_list(urma_device_t**) {}
urma_eid_info_t* urma_get_eid_list(urma_device_t*, std::uint32_t*) { return nullptr; }
void urma_free_eid_list(urma_eid_info_t*) {}
urma_status_t urma_query_device(urma_device_t*, urma_device_attr_t* attributes) {
  auto& value = *reinterpret_cast<kvaf::urma::UbDeviceAttributes*>(attributes);
  value = {};
  value.dev_cap.max_jfs_sge = device_sges;
  value.dev_cap.max_msg_size = 65536;
  return URMA_SUCCESS;
}
urma_token_id_t* urma_alloc_token_id(urma_context_t*) { return new urma_token_id_t{}; }
urma_status_t urma_free_token_id(urma_token_id_t* token) { delete token; return URMA_SUCCESS; }
urma_target_seg_t* urma_register_seg(urma_context_t*, urma_seg_cfg_t* config) {
  auto* segment = new urma_target_seg_t{};
  segment->seg.ubva.va = config->va;
  segment->seg.len = config->len;
  ++segments;
  return segment;
}
urma_status_t urma_unregister_seg(urma_target_seg_t* segment) {
  --segments; delete segment; return URMA_SUCCESS;
}
urma_target_seg_t* urma_import_seg(urma_context_t*, urma_seg_t* value, urma_token_t*,
                                  std::uint64_t, urma_import_seg_flag_t) {
  if (delay_output_import && value->ubva.va == output_base)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  auto* segment = new urma_target_seg_t{};
  segment->seg = *value;
  ++segments;
  return segment;
}
urma_status_t urma_unimport_seg(urma_target_seg_t* segment) {
  if (delay_output_import && segment->seg.ubva.va == output_base)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  --segments; delete segment; return URMA_SUCCESS;
}
urma_jfc_t* urma_create_jfc(urma_context_t*, urma_jfc_cfg_t*) {
  auto* cq = new urma_jfc_t{};
  completions[cq] = {};
  return cq;
}
urma_status_t urma_delete_jfc(urma_jfc_t* cq) {
  assert(completions[cq].empty()); completions.erase(cq); delete cq; return URMA_SUCCESS;
}
urma_jfr_t* urma_create_jfr(urma_context_t*, urma_jfr_cfg_t*) { return new urma_jfr_t{}; }
urma_status_t urma_delete_jfr(urma_jfr_t* queue) { delete queue; return URMA_SUCCESS; }
urma_jetty_t* urma_create_jetty(urma_context_t*, urma_jetty_cfg_t* config) {
  queue_sges = config->jfs_cfg.max_sge;
  auto* queue = new urma_jetty_t{};
  queue->jetty_cfg = *config;
  return queue;
}
urma_status_t urma_delete_jetty(urma_jetty_t* queue) { delete queue; return URMA_SUCCESS; }
urma_status_t urma_unimport_jetty(urma_target_jetty_t* queue) { delete queue; return URMA_SUCCESS; }
urma_status_t urma_post_jetty_send_wr(urma_jetty_t* queue, urma_jfs_wr_t* wr, urma_jfs_wr_t**) {
  for (; wr; wr = wr->next) {
    const auto* src = reinterpret_cast<const Sge*>(wr->rw.src.sge);
    const auto* dst = reinterpret_cast<const Sge*>(wr->rw.dst.sge);
    assert(wr->rw.dst.num_sge == 1 && wr->rw.src.num_sge <= queue_sges);
    assert(dst->addr >= dst->tseg->seg.ubva.va);
    assert(dst->len <= dst->tseg->seg.len - (dst->addr - dst->tseg->seg.ubva.va));
    if (suffix_descriptor && wr->opcode == URMA_OPC_READ && src[0].addr == suffix_descriptor) {
      std::size_t pending_data = 0;
      for (const auto& cq : completions) pending_data += cq.second.size();
      assert(!publish_suffix && pending_data > 0);  // Head CQEs have not been drained.
      saw_async_suffix_read = true;
    }
    std::uint32_t bytes = 0;
    for (std::uint32_t i = 0; i < wr->rw.src.num_sge; ++i) {
      assert(src[i].addr >= src[i].tseg->seg.ubva.va);
      assert(src[i].len <= src[i].tseg->seg.len - (src[i].addr - src[i].tseg->seg.ubva.va));
      std::memcpy(reinterpret_cast<void*>(dst->addr + bytes), reinterpret_cast<void*>(src[i].addr), src[i].len);
      bytes += src[i].len;
    }
    assert(bytes == dst->len && bytes <= 65536);
    if (wr->opcode == URMA_OPC_WRITE && dst->addr >= output_base && dst->addr < output_base + output_bytes) {
      assert(wr->rw.src.num_sge <= queue_sges);
      ++data_wrs; data_sges += wr->rw.src.num_sge;
      if (publish_suffix) {
        assert(dst->addr + bytes <= output_base + head_bytes);
        head_written += bytes;
        if (head_written == head_bytes) {
          auto publish = std::move(publish_suffix);
          publish_suffix = {};
          publish();
        }
      }
      if (wr->flag.bs.complete_enable) {
        assert(wr->flag.bs.place_order == URMA_STRONG_ORDER);
        ++data_tails;
      }
    }
    if (wr->opcode == URMA_OPC_WRITE && dst->addr >= done_address + sizeof(kvaf::cpu_push::Completion)
        && dst->addr < done_address + sizeof(kvaf::cpu_push::Completion) + 4 * sizeof(kvaf::cpu_push::Ready))
      ++ready_wrs;
    if (wr->opcode == URMA_OPC_WRITE && dst->addr == done_address)
      assert(segments == 2);  // Workspace + control; output is unimported before done.
    if (wr->flag.bs.complete_enable)
      completions[queue->jetty_cfg.jfs_cfg.jfc].push_back(wr->user_ctx);
  }
  return URMA_SUCCESS;
}
int urma_poll_jfc(urma_jfc_t* cq, int capacity, urma_cr_t* result) {
  auto& pending = completions[cq];
  int count = 0;
  while (count < capacity && !pending.empty()) {
    result[count] = {};
    result[count].status = URMA_CR_SUCCESS;
    result[count++].user_ctx = pending.front();
    pending.pop_front();
  }
  return count;
}
}  // extern "C"

namespace {
void CheckProfile(const kvaf::cpu_push::Transport& transport,
                  const kvaf::CpuSglWriteSample& sample, std::uint32_t sequence,
                  bool payload_only, bool profiling, bool pipeline = false) {
  auto* capture = std::tmpfile();
  assert(capture);
  std::fflush(stderr);
  const auto saved = dup(fileno(stderr));
  assert(saved >= 0 && dup2(fileno(capture), fileno(stderr)) >= 0);
  transport.PrintProfile();
  std::fflush(stderr);
  assert(dup2(saved, fileno(stderr)) >= 0);
  close(saved);
  std::rewind(capture);
  std::string log;
  char buffer[4096];
  while (const auto bytes = std::fread(buffer, 1, sizeof(buffer), capture)) log.append(buffer, bytes);
  std::fclose(capture);
  if (!profiling) { assert(log.empty()); return; }
  if (pipeline) {
    assert(log.find("CPU_SGL_PIPELINE_PROFILE sequence=" + std::to_string(sequence)) == 0);
    assert(log.find("ready_wrs=0") != std::string::npos);
    return;
  }
  if (!payload_only) {
    assert(log.find("CPU_PUSH_PROFILE ") != std::string::npos);
    assert(log.find("CPU_SGL_WRITE_STAGES ") == std::string::npos);
    return;
  }
  assert(log.find("CPU_SGL_WRITE_SAMPLE sequence=" + std::to_string(sequence)) == 0);
  const auto stages = log.find("CPU_SGL_WRITE_STAGES ");
  assert(stages != std::string::npos);
  std::istringstream fields(log.substr(stages));
  std::map<std::string, double> values;
  std::string field;
  while (fields >> field) {
    const auto equals = field.find('=');
    if (equals != std::string::npos) values[field.substr(0, equals)] = std::stod(field.substr(equals + 1));
  }
  assert(values.at("sequence") == sequence);
  assert(values.at("plan_reused") == (sequence == 2));
  double accounted = 0;
  for (const auto* name : {"notify_read_us", "header_read_us", "address_list_read_us", "address_validate_us",
                           "output_desc_read_us", "output_import_us", "plan_us", "target_update_us",
                           "source_sge_update_us", "data_begin_us", "payload_post_us", "payload_cq_wait_us",
                           "output_unimport_us", "done_post_us", "done_cq_wait_us"}) {
    assert(values.at(name) >= 0 && std::isfinite(values.at(name)));
    accounted += values.at(name);
  }
  assert(accounted <= values.at("cpu_total_us") + 0.02);  // Allow printed rounding.
  assert(std::abs(values.at("payload_post_us") + values.at("payload_cq_wait_us") -
                  sample.elapsed_ns / 1000.0) <= 0.002);
  if (delay_output_import) {
    assert(values.at("output_import_us") >= 2000);
    assert(values.at("output_unimport_us") >= 2000);
    assert(values.at("cpu_total_us") >= sample.elapsed_ns / 1000.0 + 4000);
  }
}
void Run(std::uint32_t requested, std::uint32_t expected, std::uint32_t count = 4096,
         std::uint32_t token_bytes = 656, bool payload_only = false, bool profiling = true,
         bool pipeline = false, kvaf::cpu_push::Result suffix_result = kvaf::cpu_push::Result::kOk,
         bool bad_suffix_address = false) {
  using namespace kvaf::cpu_push;
  Layout layout;
  assert(MakeLayout(token_bytes, 4, &layout, true));
  auto* hbm = static_cast<std::uint8_t*>(std::aligned_alloc(64, layout.bytes));
  assert(hbm);
  std::memset(hbm, 0, layout.bytes);
  std::vector<std::uint8_t> source(count * token_bytes), output(source.size()), second(output.size());
  for (std::size_t i = 0; i < source.size(); ++i) source[i] = (i * 17 + i / token_bytes) % 251;
  urma_device_t device{};
  urma_context_t context{};
  context.dev = &device;
  urma_target_seg_t source_segment{};
  source_segment.seg.ubva.va = reinterpret_cast<std::uint64_t>(source.data());
  source_segment.seg.len = source.size();
  urma_seg_t hbm_segment{};
  hbm_segment.ubva.va = reinterpret_cast<std::uint64_t>(hbm);
  hbm_segment.len = layout.bytes;
  kvaf::peer::OpenCpuPushGetRequest setup{};
  setup.source_bytes = source.size(); setup.hbm_base = hbm_segment.ubva.va; setup.hbm_bytes = layout.bytes;
  setup.npu_tp = 1; setup.npu_psn = 1; setup.token_value = 7;
  setup.segment_bytes = sizeof(hbm_segment); setup.token_bytes = token_bytes; setup.workers = 4;
  setup.output_mode = pipeline ? kPipelinedOutput : payload_only ? kPayloadOnlyOutput : kDirectOutput; setup.sge_per_wr = requested;
  std::memcpy(setup.segment, &hbm_segment, sizeof(hbm_segment));
  for (auto& channel : setup.channels) channel.bytes = 28;
  kvaf::peer::OpenCpuPushGetResponse response{};
  Transport transport;
  assert(transport.Initialize(&context, &source_segment, setup, QueryTp, ImportJetty, 1, &response).ok());
  assert(queue_sges == expected);
  done_address = setup.hbm_base + layout.done;
  for (auto sequence : {1U, 2U}) {
    auto& result = sequence == 1 ? output : second;
    std::fill(result.begin(), result.end(), 0xA5);
    output_base = reinterpret_cast<std::uint64_t>(result.data()); output_bytes = result.size();
    auto* request = reinterpret_cast<Request*>(hbm);
    *request = {kMagic, sequence, count, token_bytes, 4, kDirectOutput, {}};
    auto* target = reinterpret_cast<OutputTarget*>(hbm + layout.output_target);
    *target = {};
    target->base = output_base; target->bytes = result.size(); target->token_value = 7;
    target->segment_bytes = sizeof(urma_seg_t);
    urma_seg_t output_segment{};
    output_segment.ubva.va = target->base; output_segment.len = target->bytes;
    std::memcpy(target->segment, &output_segment, sizeof(output_segment));
    auto* addresses = reinterpret_cast<std::uint64_t*>(hbm + layout.descriptor);
    for (std::uint32_t i = 0; i < count; ++i)
      addresses[i] = source_segment.seg.ubva.va + ((count - i + sequence - 1) % count) * token_bytes;
    const auto head = std::min(count, kPipelineHeadRows);
    if (pipeline) {
      request->reserved_words[0] = head;
      head_bytes = std::uint64_t{head} * token_bytes;
      head_written = 0;
      saw_async_suffix_read = false;
      suffix_descriptor = head < count && suffix_result == Result::kOk
          ? setup.hbm_base + layout.descriptor + std::uint64_t{head} * 8 : 0;
      if (head < count) {
        for (auto i = head; i < count; ++i) addresses[i] = UINT64_MAX;
        publish_suffix = [&, sequence] {
          for (auto i = head; i < count; ++i)
            addresses[i] = source_segment.seg.ubva.va + ((count - i + sequence - 1) % count) * token_bytes;
          if (bad_suffix_address) addresses[head] = source_segment.seg.ubva.va - 1;
          *reinterpret_cast<std::uint64_t*>(hbm + kSuffixNotifyOffset) = Progress(sequence, std::uint32_t(suffix_result));
        };
      } else {
        *reinterpret_cast<std::uint64_t*>(hbm + kSuffixNotifyOffset) = Progress(sequence, std::uint32_t(suffix_result));
      }
    }
    data_wrs = data_sges = ready_wrs = data_tails = 0;
    reinterpret_cast<Word*>(hbm + layout.notify)->value = sequence | (profiling ? kProfileNotify : 0);
    std::uint64_t received = 0;
    assert(transport.ReadRequest(&received).ok() && received == sequence);
    assert(transport.ReadDescriptor(sequence).ok());
    assert(transport.WriteBatch().ok());
    const auto expected_result = bad_suffix_address ? Result::kInvalidRequest : suffix_result;
    assert(transport.batch_result() == expected_result);
    assert(transport.Complete(sequence, transport.batch_result()).ok());
    const auto& sample = reinterpret_cast<Completion*>(hbm + layout.done)->cpu_sgl_write;
    if (payload_only) {
      assert(ready_wrs == 0 && sample.data_wrs == data_wrs && sample.tail_cqes == data_tails);
      assert(sample.waves == (count == 65536 ? 2 : 1));
      assert((sample.elapsed_ns > 0) == (sample.waves == 1 && profiling));
      assert(data_tails == (count == 1 ? 1 : 4 * sample.waves));
    } else if (pipeline) {
      assert(!publish_suffix && ready_wrs == 0 && sample.waves == 0 && sample.elapsed_ns == 0);
      if (head < count && suffix_result == Result::kOk) assert(saw_async_suffix_read);
      const auto token = reinterpret_cast<Completion*>(hbm + layout.done)->token;
      assert(token == Progress(sequence, std::uint32_t(expected_result)));
    } else {
      assert(ready_wrs > 0 && sample.waves == 0 && sample.elapsed_ns == 0);
    }
    CheckProfile(transport, sample, sequence, payload_only, profiling, pipeline);
    if (count == 4096 && token_bytes == 656 && expected_result == Result::kOk) {
      assert(data_wrs == (expected == 4 ? 1024 : expected == 8 ? 512 : 320));
      assert(data_sges == count);
    }
    const auto written = expected_result == Result::kOk ? count : head;
    for (std::uint32_t i = 0; i < written; ++i)
      assert(std::memcmp(result.data() + i * token_bytes, reinterpret_cast<void*>(addresses[i]), token_bytes) == 0);
    for (std::size_t i = std::size_t(written) * token_bytes; i < result.size(); ++i) assert(result[i] == 0xA5);
  }
  suffix_descriptor = 0;
  assert(transport.Shutdown().ok());
  assert(segments == 0 && completions.empty());
  std::free(hbm);
}
void Reject(std::uint32_t requested, std::uint32_t device_limit, std::uint32_t mode) {
  device_sges = device_limit;
  urma_context_t context{};
  urma_device_t device{};
  context.dev = &device;
  urma_target_seg_t source{};
  kvaf::cpu_push::Layout layout;
  assert(kvaf::cpu_push::MakeLayout(656, 4, &layout, mode == kvaf::cpu_push::kDirectOutput));
  kvaf::peer::OpenCpuPushGetRequest setup{};
  setup.npu_tp = 1; setup.npu_psn = 1; setup.token_value = 7; setup.source_bytes = 656;
  setup.segment_bytes = sizeof(urma_seg_t); setup.token_bytes = 656; setup.workers = 4;
  setup.hbm_base = 4096; setup.hbm_bytes = layout.bytes;
  setup.output_mode = mode; setup.sge_per_wr = requested;
  kvaf::peer::OpenCpuPushGetResponse response{};
  kvaf::cpu_push::Transport transport;
  assert(!transport.Initialize(&context, &source, setup, QueryTp, ImportJetty, 1, &response).ok());
  assert(transport.Shutdown().ok());
  assert(segments == 0 && completions.empty());
}
}  // namespace

int main() {
  Run(4, 4);
  Run(8, 8);
  Run(0, 13);
  Run(8, 8, 65, 10000);  // Byte-limited WRs and split-token source offsets.
  Run(4, 4, 4096, 656, true);
  Run(8, 8, 4096, 656, true);
  Run(8, 8, 65, 10000, true);
  Run(4, 4, 1, 656, true);  // Only one Jetty has payload.
  Run(4, 4, 65536, 16, true);  // Complete both waves, leave elapsed_ns unset.
  Run(4, 4, 4096, 656, true, false);  // No diagnostic output without profiling.
  delay_output_import = true;
  Run(4, 4, 4096, 656, true);  // Import/retirement delay stays outside the payload window.
  delay_output_import = false;
  Run(4, 4, 4096, 656, false, true, true);
  Run(8, 8, 8192, 656, false, true, true);
  Run(4, 4, 7, 656, false, true, true);
  Run(4, 4, 1, 656, false, false, true);
  Run(4, 4, 65536, 16, false, true, true);
  Run(4, 4, 4096, 656, false, true, true, kvaf::cpu_push::Result::kInvalidRequest);
  Run(4, 4, 4096, 656, false, true, true, kvaf::cpu_push::Result::kOk, true);
  Reject(8, 4, kvaf::cpu_push::kDirectOutput);
  Reject(17, 32, kvaf::cpu_push::kDirectOutput);
  Reject(0, 0, kvaf::cpu_push::kDirectOutput);
  Reject(4, 13, kvaf::cpu_push::kScatterOutput);
}
