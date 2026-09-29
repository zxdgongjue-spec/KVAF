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
  assert(direct_report.find("Check DDR mailbox (enable worker on first call): 1.000 us") !=
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
}
