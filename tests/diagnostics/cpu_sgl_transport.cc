// Real CPU SGL WRITE transport with a synchronous fake URMA provider.
// Requires the public UMDK headers, no URMA library or hardware.
#include "cpu_service/push/cpu_push_get_transport.h"
#include "urma_device_abi.h"
#include "cpu_sgl_push_layout.h"

#include <cassert>
#include <cerrno>
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
std::uint32_t device_sges = 13, device_message = 65536, queue_sges{}, data_wrs{}, data_sges{}, segments{};
std::uint64_t output_base{}, output_bytes{}, done_address{};
std::uint32_t ready_wrs{}, data_tails{};
bool delay_output_import{};
std::function<void()> publish_suffix, delayed_progress;
std::function<void(std::uint64_t)> data_post;
std::uint32_t read_posts{};
std::uint64_t head_bytes{}, head_written{}, suffix_descriptor{};
bool saw_async_suffix_read{};
std::function<void(std::uint64_t, std::uint32_t)> chunk_read, chunk_write;
bool retain_chunk_storage{};
urma_jfc_t* last_read_cq{};
using Checks = std::vector<std::function<void()>>;
std::map<urma_jfc_t*, Checks> unfenced;
std::map<std::pair<urma_jfc_t*, std::uint64_t>, Checks> retained;
std::map<urma_jfc_t*, std::size_t> pending_sges, unfenced_sges;
std::map<std::pair<urma_jfc_t*, std::uint64_t>, std::size_t> fenced_sges;
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
  value.dev_cap.max_msg_size = device_message;
  return URMA_SUCCESS;
}
urma_token_id_t* urma_alloc_token_id(urma_context_t*) { return new urma_token_id_t{}; }
urma_status_t urma_free_token_id(urma_token_id_t* token) { delete token; return URMA_SUCCESS; }
urma_target_seg_t* urma_register_seg(urma_context_t*, urma_seg_cfg_t* config) {
  // Public UMDK rejects LOCAL_ONLY combined with remote access, and WRITE
  // without READ. Enforce provider validation before simulating registration.
  const auto access = config->flag.bs.access;
  if (((access & 1U) && (access & 14U)) || ((access & 4U) && !(access & 2U)) ||
      ((access & 8U) && (access & 6U) != 6U)) {
    errno = EINVAL;
    return nullptr;
  }
  auto* segment = new urma_target_seg_t{};
  segment->seg.ubva.va = config->va;
  segment->seg.len = config->len;
  segment->seg.attr.bs.access = access;
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
    if (wr->opcode == URMA_OPC_READ) ++read_posts;
    if (wr->opcode == URMA_OPC_READ) last_read_cq = queue->jetty_cfg.jfs_cfg.jfc;
    if (chunk_read && wr->opcode == URMA_OPC_READ) chunk_read(src[0].addr, src[0].len);
    const bool data = wr->opcode == URMA_OPC_WRITE && dst->addr >= output_base && dst->addr < output_base + output_bytes;
    if (data && data_post) data_post(dst->addr);
    if (retain_chunk_storage && data) {
      auto* cq = queue->jetty_cfg.jfs_cfg.jfc;
      pending_sges[cq] += wr->rw.src.num_sge;
      unfenced_sges[cq] += wr->rw.src.num_sge;
      assert(pending_sges[cq] <= 8192);
      const auto saved = *wr;
      const std::vector<Sge> sources(src, src + wr->rw.src.num_sge);
      const auto destination = *dst;
      unfenced[queue->jetty_cfg.jfs_cfg.jfc].push_back([wr, saved, src, dst, sources, destination] {
        assert(wr->next == saved.next && wr->user_ctx == saved.user_ctx);
        assert(wr->rw.src.sge == saved.rw.src.sge && wr->rw.dst.sge == saved.rw.dst.sge);
        for (std::size_t i = 0; i < sources.size(); ++i)
          assert(src[i].addr == sources[i].addr && src[i].len == sources[i].len && src[i].tseg == sources[i].tseg);
        assert(dst->addr == destination.addr && dst->len == destination.len && dst->tseg == destination.tseg);
      });
      if (wr->flag.bs.complete_enable) {
        fenced_sges[{cq, wr->user_ctx}] = unfenced_sges[cq];
        unfenced_sges.erase(cq);
        retained[{queue->jetty_cfg.jfs_cfg.jfc, wr->user_ctx}] = std::move(unfenced[queue->jetty_cfg.jfs_cfg.jfc]);
        unfenced.erase(queue->jetty_cfg.jfs_cfg.jfc);
      }
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
      if (chunk_write) chunk_write(dst->addr, bytes);
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
  if (delayed_progress) { auto callback = std::move(delayed_progress); delayed_progress = {}; callback(); }
  auto& pending = completions[cq];
  int count = 0;
  while (count < capacity && !pending.empty()) {
    const auto key = std::make_pair(cq, pending.front());
    auto storage = retained.find(key);
    if (storage != retained.end()) {
      for (const auto& check : storage->second) check();
      retained.erase(storage);
      pending_sges[cq] -= fenced_sges.at(key);
      if (!pending_sges[cq]) pending_sges.erase(cq);
      fenced_sges.erase(key);
    }
    result[count] = {};
    result[count].status = URMA_CR_SUCCESS;
    result[count++].user_ctx = pending.front();
    pending.pop_front();
  }
  return count;
}
}  // extern "C"

namespace {
std::string CaptureProfile(const kvaf::cpu_push::Transport& transport) {
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
  return log;
}
void CheckProfile(const kvaf::cpu_push::Transport& transport,
                  const kvaf::CpuSglWriteSample& sample, std::uint32_t sequence,
                  bool payload_only, bool profiling, bool pipeline = false, bool chunked = false, bool pushed = false) {
  const auto log = CaptureProfile(transport);
  if (!profiling) { assert(log.empty()); return; }
  if (chunked) {
    assert(log.find(std::string(pushed ? "CPU_SGL_PUSH_CHUNK_PROFILE sequence=" : "CPU_SGL_CHUNK_PROFILE sequence=") + std::to_string(sequence)) == 0);
    std::istringstream fields(log);
    std::map<std::string, double> values;
    std::string field;
    while (fields >> field) {
      const auto equals = field.find('=');
      if (equals != std::string::npos && field[equals + 1] >= '0' && field[equals + 1] <= '9')
        values[field.substr(0, equals)] = std::stod(field.substr(equals + 1));
    }
    assert(values.at("chunk_rows") == kvaf::cpu_push::kChunkRows);
    assert(values.at("max_inflight") <= 4 && values.at("ready_wrs") == 0);
    assert(values.at("data_wrs") == data_wrs && values.at("tail_cqes") == data_tails);
    assert(values.at("plan_reused") == (sequence == 2));
    if (values.at("result") == double(kvaf::cpu_push::Result::kOk))
      assert(values.at("chunks") == kvaf::cpu_push::PushChunkCount(uint32_t(values.at("rows")), pushed ? uint32_t(values.at("head_rows")) : kvaf::cpu_push::kChunkRows));
    if (pushed) {
      assert(values.at("descriptor_reads") == 0 && values.at("address_validate_us") == 0);
      assert(values.at("no_ready_us") >= 0 && values.at("no_ready_us") <= values.at("cpu_total_us"));
      assert(std::abs(values.at("no_ready_us") - values.at("no_ready_inflight_us") - values.at("no_ready_idle_us")) < 0.002);
    }
    else assert(values.at("chunk_read_us") >= 0);
    for (const auto* name : {"chunk_ready_us", "chunk_submit_us", "chunk_cq_wait_us", "cpu_total_us"})
      assert(std::isfinite(values.at(name)) && values.at(name) >= 0);
    return;
  }
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
// Release chunks only after a prior chunk has actually been posted. Chunk 2
// precedes chunk 1, proving the consumer has no head-of-line builder barrier.
void RunChunks(std::uint32_t count, std::uint32_t token_bytes = 656, int fault = 0, bool profiling = true, bool all_ready = false, std::uint32_t workers = 4, bool pushed = false, std::uint32_t head_rows = kvaf::cpu_push::kChunkRows, bool delay_ready = false) {
  using namespace kvaf::cpu_push;
  Layout layout;
  assert(MakeLayout(token_bytes, workers, &layout, true));
  const auto hbm_bytes = layout.bytes + (pushed ? PushWorkspaceBytes(head_rows) : 0);
  auto* hbm = static_cast<std::uint8_t*>(std::aligned_alloc(64, hbm_bytes));
  assert(hbm); std::memset(hbm, pushed ? 0xee : 0, hbm_bytes);
  std::vector<std::uint8_t> source(std::size_t(count) * token_bytes), output(source.size()), second(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) source[i] = uint8_t(i * 37 + i / token_bytes);
  urma_device_t device{}; urma_context_t context{}; context.dev = &device;
  urma_target_seg_t source_segment{};
  source_segment.seg.ubva.va = reinterpret_cast<std::uint64_t>(source.data()); source_segment.seg.len = source.size();
  urma_seg_t hbm_segment{};
  hbm_segment.ubva.va = reinterpret_cast<std::uint64_t>(hbm); hbm_segment.len = hbm_bytes;
  kvaf::peer::OpenCpuPushGetRequest setup{};
  setup.source_bytes = source.size(); setup.hbm_base = hbm_segment.ubva.va; setup.hbm_bytes = hbm_bytes;
  setup.npu_tp = 1; setup.npu_psn = 1; setup.token_value = 7;
  setup.segment_bytes = sizeof(hbm_segment); setup.token_bytes = token_bytes; setup.workers = workers;
  setup.output_mode = pushed ? PushOutputMode(head_rows) : kChunkedOutput; setup.sge_per_wr = 4;
  std::memcpy(setup.segment, &hbm_segment, sizeof(hbm_segment));
  for (auto& channel : setup.channels) channel.bytes = 28;
  kvaf::peer::OpenCpuPushGetResponse response{};
  Transport transport;
  kvaf::peer::OpenCpuSglPushRequest push{};
  push.base = setup; push.control_tp = push.control_psn = 1; push.control.bytes = 28;
  kvaf::peer::OpenCpuSglPushResponse mailbox{};
  if (pushed && head_rows < kChunkRows) {
    auto invalid = setup;
    invalid.output_mode = kPushHeadOutput; // Missing Head rows.
    assert(!transport.Initialize(&context, &source_segment, invalid, QueryTp, ImportJetty, 1, &response, 0, &push, &mailbox).ok());
    invalid.output_mode = kPushHeadOutput | (1024U << 8); // Canonical default must use mode 5.
    assert(!transport.Initialize(&context, &source_segment, invalid, QueryTp, ImportJetty, 1, &response, 0, &push, &mailbox).ok());
    invalid = setup; invalid.hbm_bytes -= sizeof(Word); // Missing the 65th immutable commit.
    assert(!transport.Initialize(&context, &source_segment, invalid, QueryTp, ImportJetty, 1, &response, 0, &push, &mailbox).ok());
    assert(!transport.Initialize(&context, &source_segment, setup, QueryTp, ImportJetty, 1, &response).ok());
  }
  assert(transport.Initialize(&context, &source_segment, setup, QueryTp, ImportJetty, 1, &response, 0,
      pushed ? &push : nullptr, pushed ? &mailbox : nullptr).ok());
  auto* control = pushed ? reinterpret_cast<std::uint8_t*>(mailbox.mailbox_base) : hbm;
  done_address = setup.hbm_base + layout.done;
  const auto chunks = PushChunkCount(count, head_rows);
  retain_chunk_storage = true;
  for (auto sequence : {1U, 2U}) {
    auto& result = sequence == 1 ? output : second;
    std::fill(result.begin(), result.end(), 0xa5);
    output_base = reinterpret_cast<std::uint64_t>(result.data()); output_bytes = result.size();
    *reinterpret_cast<Request*>(control) = {kMagic, sequence, count, token_bytes, workers, kChunkedOutput, {kChunkRows, head_rows < kChunkRows ? head_rows : 0}};
    auto* target = reinterpret_cast<OutputTarget*>(control + layout.output_target);
    *target = {}; target->base = output_base; target->bytes = output_bytes;
    target->token_value = 7; target->segment_bytes = sizeof(urma_seg_t);
    urma_seg_t segment{}; segment.ubva.va = output_base; segment.len = output_bytes;
    std::memcpy(target->segment, &segment, sizeof(segment));
    auto* addresses = reinterpret_cast<std::uint64_t*>(control + layout.descriptor);
    std::fill(addresses, addresses + count, UINT64_MAX);
    std::vector<bool> published(chunks), written(chunks);
    std::vector<std::uint64_t> written_bytes(chunks);
    std::atomic<bool> stopped{false};
    const auto expected_address = [&](std::uint32_t row) {
      return source_segment.seg.ubva.va + std::uint64_t((count - row + sequence - 1) % count) * token_bytes;
    };
    auto publish = [&](std::uint32_t c) {
      assert(c < chunks && (c < workers || published[c - workers]));
      for (auto i = PushChunkBegin(c, head_rows); i < PushChunkEnd(count, c, head_rows); ++i) addresses[i] = expected_address(i);
      if (fault == 2 && c == 1) addresses[kChunkRows] = source_segment.seg.ubva.va - 1;
      published[c] = true;
      __atomic_store_n(&reinterpret_cast<Word*>(control + layout.ready)[c % workers].value,
                       Progress(sequence, c / workers + 1), __ATOMIC_RELEASE);
    };
    if (all_ready) for (uint32_t c = 0; c < chunks; ++c) publish(c);
    else publish(0);
    if (fault == 3) reinterpret_cast<Word*>(control + layout.ready)[1].value = Progress(sequence + 1, 1);
    if (fault == 4) reinterpret_cast<Word*>(control + layout.ready)[1].value = Progress(sequence, chunks / workers + 2);
    bool read_with_pending_data = false, multiple_pending_chunks = false, post_with_pending_read = false;
    chunk_read = [&](std::uint64_t address, std::uint32_t bytes) {
      const auto descriptors = setup.hbm_base + layout.descriptor;
      if (address < descriptors || address >= descriptors + uint64_t(count) * 8) return;
      const auto row = uint32_t((address - descriptors) / 8), c = PushChunkIndex(row, head_rows);
      assert(published[c] && row + bytes / 8 <= PushChunkEnd(count, c, head_rows));
      if (c) for (const auto& cq : completions) read_with_pending_data |= !cq.second.empty();
    };
    chunk_write = [&](std::uint64_t address, std::uint32_t bytes) {
      const auto c = PushChunkIndex(uint32_t((address - output_base) / token_bytes), head_rows);
      assert(published[c]); written[c] = true; written_bytes[c] += bytes;
      post_with_pending_read |= last_read_cq && !completions[last_read_cq].empty();
      std::size_t pending = 0;
      for (const auto& cq : completions) pending += cq.second.size();
      multiple_pending_chunks |= pending > kJettyCount;
      if (written_bytes[c] != uint64_t(PushChunkEnd(count, c, head_rows) - PushChunkBegin(c, head_rows)) * token_bytes) return;
      if (fault == 5) { stopped.store(true, std::memory_order_release); return; }
      if (delay_ready && c == 0) {
        assert(chunks == 2);
        delayed_progress = [&] { std::this_thread::sleep_for(std::chrono::milliseconds(2)); publish(1); };
        return;
      }
      if (all_ready) return;
      if (chunks > 2 && c == 0) { publish(2); return; }
      if (c == 2) {
        if (fault == 1) reinterpret_cast<Word*>(control + layout.ready)[1].value = Progress(sequence, kChunkFailed | uint32_t(Result::kInvalidRequest));
        else publish(1);
        return;
      }
      const auto next = c == 1 && chunks > 2 ? 3U : c + 1;
      if (next < chunks) publish(next);
    };
    data_wrs = data_sges = ready_wrs = data_tails = 0;
    read_posts = 0;
    reinterpret_cast<Word*>(control + layout.notify)->value = sequence | (profiling ? kProfileNotify : 0);
    std::uint64_t received = 0;
    assert(transport.ReadRequest(&received).ok() && received == sequence);
    if (pushed && head_rows < kChunkRows && sequence == 1) {
      auto& advertised = reinterpret_cast<Request*>(control)->reserved_words[1];
      advertised = 0; assert(!transport.ReadDescriptor(sequence).ok());
      advertised = head_rows == 256 ? 128 : 256; assert(!transport.ReadDescriptor(sequence).ok());
      advertised = head_rows;
    }
    assert(transport.ReadDescriptor(sequence).ok());
    assert(transport.WriteBatch(pushed ? &stopped : nullptr).ok());
    assert(transport.batch_result() == (fault == 5 ? Result::kTimeout : fault == 3 ? Result::kSequenceError : fault ? Result::kInvalidRequest : Result::kOk));
    assert(retained.empty() && unfenced.empty() && pending_sges.empty() && unfenced_sges.empty() && fenced_sges.empty());
    assert(transport.Complete(sequence, transport.batch_result()).ok());
    assert(reinterpret_cast<Completion*>(hbm + layout.done)->token == Progress(sequence, uint32_t(transport.batch_result())));
    assert(ready_wrs == 0);
    if (!fault && count == 8192 && token_bytes == 656) assert(data_wrs == 2048 && data_tails == chunks * 4 && data_sges == count);
    CheckProfile(transport, {}, sequence, false, profiling, false, true, pushed);
    if (pushed) assert(read_posts == 0);
    if (delay_ready) {
      const auto log = CaptureProfile(transport);
      assert(std::stod(log.substr(log.find("no_ready_inflight_us=") + 21)) >= 1000);
    }
    if (!pushed && all_ready && chunks > 1) assert(post_with_pending_read);
    if (!pushed && chunks > 1 && fault < 3) assert(read_with_pending_data);
    if (chunks > 2 && fault < 3 && device_message == 65536) assert(multiple_pending_chunks);
    for (uint32_t c = 0; c < chunks; ++c) {
      if (fault) assert(written[c] == (fault == 5 ? c == 0 : fault < 3 && (c == 0 || c == 2)));
      else assert(written[c]);
      for (auto i = PushChunkBegin(c, head_rows); i < PushChunkEnd(count, c, head_rows); ++i)
        if (written[c]) assert(!std::memcmp(result.data() + uint64_t(i) * token_bytes, reinterpret_cast<void*>(expected_address(i)), token_bytes));
        else for (uint32_t j = 0; j < token_bytes; ++j) assert(result[uint64_t(i) * token_bytes + j] == 0xa5);
    }
  }
  chunk_read = {}; chunk_write = {}; retain_chunk_storage = false;
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

#ifndef KVAF_CPU_SGL_REUSE_FIXTURE
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
  RunChunks(8192);
  RunChunks(8192, 656, 0, true, true); // Descriptor prefetch stays outstanding during the prior post.
  RunChunks(8192, 656, 0, true, false, 5); // User AICPU worker count.
  RunChunks(8192, 656, 0, true, true, 1);
  RunChunks(1, 656, 0, true, false, 8);
  RunChunks(65536, 16); // More than the in-flight window and tracker slots over time.
  RunChunks(7); RunChunks(1, 656, 0, false);
  RunChunks(4097, 20000); // Byte-limited SGEs and non-aligned final chunk.
  RunChunks(8192, 656, 1); // Producer error after earlier writes.
  RunChunks(8192, 656, 2); // Invalid descriptor must never become an SGE.
  RunChunks(8192, 656, 3); // Future sequence is not a current publication.
  RunChunks(8192, 656, 4); // Progress cannot exceed the producer's assigned chunks.
  device_message = 1024;
  RunChunks(4097, 10000); // Concurrent plans must reserve total SGEs as well as WR credits.
  device_message = 256;
  RunChunks(3073, 10000); // One chunk spans multiple SQ waves; credits require draining.
  device_message = 65536;
  Reject(8, 4, kvaf::cpu_push::kDirectOutput);
  Reject(17, 32, kvaf::cpu_push::kDirectOutput);
  Reject(0, 0, kvaf::cpu_push::kDirectOutput);
  Reject(4, 13, kvaf::cpu_push::kScatterOutput);
}

#endif
