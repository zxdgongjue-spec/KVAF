#include "aicpu_gather_read_report.h"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <string>

int main() {
  kvaf::GatherReadTiming first{};
  first.configure_ns = 1000U;
  first.cpu_copy_ns = 2000U;
  first.scatter_cache_ns = 1000U;
  first.scatter_plan_ns = 2000U;
  first.scatter_post_ns = 3000U;
  first.scatter_cq_ns = 4000U;
  first.read_wqes = 6U;
  kvaf::GatherReadTiming second{};
  second.configure_ns = 3000U;
  second.cpu_copy_ns = 4000U;
  second.scatter_cache_ns = 3000U;
  second.scatter_plan_ns = 4000U;
  second.scatter_post_ns = 5000U;
  second.scatter_cq_ns = 6000U;
  second.read_wqes = 8U;

  std::FILE* output = std::tmpfile();
  assert(output != nullptr);
  kvaf::aicpu_gather_read_client::PrintGatherReadTimeline(output, 1.5, {first, second});
  kvaf::aicpu_gather_read_client::PrintGatherReadTechnicalProfile(output, {first, second});
  std::rewind(output);
  std::string report;
  char buffer[512];
  std::size_t bytes = 0U;
  while ((bytes = std::fread(buffer, 1U, sizeof(buffer), output)) != 0U)
    report.append(buffer, bytes);
  std::fclose(output);

  assert(report.find("GatherRead call: 1500.000 us") != std::string::npos);
  assert(report.find("Send DDR source list to CPU: 2.000 us") != std::string::npos);
  assert(report.find("CPU gather DDR -> staging: 3.000 us") != std::string::npos);
  assert(report.find("HBM cache maintenance:") == std::string::npos);
  assert(report.find("AICPU READ staging -> HBM: 12.000 us (7.00 READ WQEs on average)") !=
         std::string::npos);
  assert(report.find("AICPU wait for CPU gather ack:") != std::string::npos);
  assert(report.find("overlaps CPU gather") != std::string::npos);
  assert(report.find("Host wait includes CPU/AICPU") != std::string::npos);
  assert(report.find("CPU worker before notification:") <
         report.find("2 Host waits for completion:"));
  assert(report.find("GATHER_READ_TECHNICAL avg_ns") != std::string::npos);
  assert(report.find("configure_rpc=2000") != std::string::npos);
  assert(report.find("read_wqes=7.00") != std::string::npos);
  assert(report.find("Per-SQ-group timing: disabled") != std::string::npos);
  assert(report.find("Within submit: Encode WQEs:") == std::string::npos);

  std::FILE* direct_output = std::tmpfile();
  assert(direct_output != nullptr);
  first.source_write_ns = 8000U;
  first.source_upload_ns = 5000U;
  first.cpu_source_cache_ns = 3000U;
  first.ack_started_ns = 4000U;
  kvaf::aicpu_gather_read_client::PrintGatherReadTimeline(direct_output, 1.5, {first});
  std::rewind(direct_output);
  std::string direct_report;
  while ((bytes = std::fread(buffer, 1U, sizeof(buffer), direct_output)) != 0U)
    direct_report.append(buffer, bytes);
  std::fclose(direct_output);
  assert(direct_report.find("Upload DDR source list to HBM: 5.000 us") != std::string::npos);
  assert(direct_report.find("AICPU WRITE source list to CPU DDR: 8.000 us") != std::string::npos);
  assert(direct_report.find("CPU read and prepare source list: 3.000 us") != std::string::npos);
  assert(direct_report.find("CPU start observed after ACK polling began: 4.000 us (1/1 rounds)") !=
         std::string::npos);
  assert(direct_report.find("Assign sequence / enable CPU worker on first call: 1.000 us") !=
         std::string::npos);
  assert(direct_report.find("Send DDR source list to CPU:") == std::string::npos);
  std::FILE* technical_output = std::tmpfile();
  assert(technical_output != nullptr);
  kvaf::aicpu_gather_read_client::PrintGatherReadTechnicalProfile(technical_output, {first});
  std::rewind(technical_output);
  std::string technical_report;
  while ((bytes = std::fread(buffer, 1U, sizeof(buffer), technical_output)) != 0U)
    technical_report.append(buffer, bytes);
  std::fclose(technical_output);
  assert(technical_report.find("direct_setup=1000") != std::string::npos);
  assert(technical_report.find("ack_started=4000") != std::string::npos);
  assert(technical_report.find("idle_since_previous_ack=") != std::string::npos);

  first.source_upload_ns = 0U;
  first.descriptor_upload_ns = 0U;
  first.metadata_upload_ns = 6500U;
  std::FILE* packed_output = std::tmpfile();
  assert(packed_output != nullptr);
  kvaf::aicpu_gather_read_client::PrintGatherReadTimeline(packed_output, 1.5, {first});
  kvaf::aicpu_gather_read_client::PrintGatherReadTechnicalProfile(packed_output, {first});
  std::rewind(packed_output);
  std::string packed_report;
  while ((bytes = std::fread(buffer, 1U, sizeof(buffer), packed_output)) != 0U)
    packed_report.append(buffer, bytes);
  std::fclose(packed_output);
  assert(packed_report.find("Pack source list + HBM targets and upload once: 6.500 us") !=
         std::string::npos);
  assert(packed_report.find("Upload DDR source list to HBM:") == std::string::npos);
  assert(packed_report.find("Upload HBM targets:") == std::string::npos);
  assert(packed_report.find("metadata_upload=6500") != std::string::npos);

  first.aicpu_cores = 4U;
  first.profile_detail = true;
  first.scatter_total_ns = 6500U;
  first.output_layout_ns = 2300U;
  first.scatter_init_ns = 700U;
  first.scatter_encode_ns = 1000U;
  first.scatter_sq_write_ns = 1500U;
  first.scatter_doorbell_ns = 500U;
  first.scatter_start_skew_ns = 1500U;
  first.scatter_worker_ns = 5000U;
  first.read_task_spans = 192U;
  first.read_tasks = 22U;
  first.scatter_schedule_ns = 500U;
  std::FILE* parallel_output = std::tmpfile();
  assert(parallel_output != nullptr);
  kvaf::aicpu_gather_read_client::PrintGatherReadTimeline(parallel_output, 1.5, {first});
  kvaf::aicpu_gather_read_client::PrintGatherReadTechnicalProfile(parallel_output, {first});
  std::rewind(parallel_output);
  std::string parallel_report;
  while ((bytes = std::fread(buffer, 1U, sizeof(buffer), parallel_output)) != 0U)
    parallel_report.append(buffer, bytes);
  std::fclose(parallel_output);
  assert(parallel_report.find("AICPU READ staging -> HBM: 6.500 us") != std::string::npos);
  assert(parallel_report.find("4 AICPU workers, independent SQ/CQ") != std::string::npos);
  assert(parallel_report.find("must not be added") != std::string::npos);
  assert(parallel_report.find("prepare worker starts: 2.300 us") != std::string::npos);
  assert(parallel_report.find("Worker/task initialization (includes WQE templates): 0.700 us") !=
         std::string::npos);
  assert(parallel_report.find("Dynamic READ: up to 192 spans/task, 22.00 tasks/round; claim tasks: 0.500 us") !=
         std::string::npos);
  assert(parallel_report.find("Within submit: Encode WQEs: 1.000 us | Write SQ: 1.500 us") !=
         std::string::npos);
  assert(parallel_report.find("Doorbell and queue update: 0.500 us") != std::string::npos);
  assert(parallel_report.find("Worker start spread: 1.500 us | Longest worker duration: 5.000 us") !=
         std::string::npos);
  assert(parallel_report.find("encode=1000 sq_write=1500 doorbell_queue_update=500") != std::string::npos);
  assert(parallel_report.find("READ schedule: task_spans=192 tasks=22.00 claim_ns=500") !=
         std::string::npos);

  first.pipeline_chunk_bytes = 131072U;
  first.ready_wait_ns = 1200U;
  first.ack_poll_ns = 4500U;
  std::FILE* pipeline_output = std::tmpfile();
  assert(pipeline_output != nullptr);
  kvaf::aicpu_gather_read_client::PrintGatherReadTimeline(pipeline_output, 1.5, {first});
  kvaf::aicpu_gather_read_client::PrintGatherReadTechnicalProfile(pipeline_output, {first});
  std::rewind(pipeline_output);
  std::string pipeline_report;
  while ((bytes = std::fread(buffer, 1U, sizeof(buffer), pipeline_output)) != 0U)
    pipeline_report.append(buffer, bytes);
  std::fclose(pipeline_output);
  assert(pipeline_report.find("CPU publishes ready prefixes in 131072-byte chunks") != std::string::npos);
  assert(pipeline_report.find("AICPU READ staging -> HBM (includes waiting for ready prefixes): 6.500 us") !=
         std::string::npos);
  assert(pipeline_report.find("Wait for ready prefixes: 1.200 us") != std::string::npos);
  assert(pipeline_report.find("Final CPU ACK confirmation after worker 0 READs: 4.500 us") != std::string::npos);
  assert(pipeline_report.find("AICPU wait for CPU gather ack:") == std::string::npos);
  assert(pipeline_report.find("CPU start observed after ACK polling began:") == std::string::npos);
  assert(pipeline_report.find("Pipeline: chunk_bytes=131072 ready_wait_ns=1200") != std::string::npos);

  first.workers[1] = {1000U, 5000U, 4000U, 2U, 1U, 6U, 1U};
  second = first;
  second.workers[1] = {3000U, 7000U, 4000U, 2U, 1U, 6U, 3U};
  std::FILE* worker_output = std::tmpfile();
  assert(worker_output != nullptr);
  kvaf::aicpu_gather_read_client::PrintGatherReadWorkers(worker_output, {first, second});
  std::rewind(worker_output);
  std::string worker_report;
  while ((bytes = std::fread(buffer, 1U, sizeof(buffer), worker_output)) != 0U)
    worker_report.append(buffer, bytes);
  std::fclose(worker_output);
  assert(worker_report.find("worker=1 start_avg=2.000 start_p50=2.000 start_p95=2.900 end_avg=6.000") !=
         std::string::npos);
  assert(worker_report.find("duration_avg=4.000 read_wqes=2.00 sq_write_ranges=1.00 sq_wqebbs=6.00 read_tasks=2.00") !=
         std::string::npos);
  assert(worker_report.find("not physical CPU IDs") != std::string::npos);
  bool rejected = false;
  second.aicpu_cores = 2U;
  try {
    kvaf::aicpu_gather_read_client::PrintGatherReadWorkers(stdout, {first, second});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}
