#pragma once
// Benchmark-only Linux observations. None of these run in ordinary BatchGet.
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include "status.h"

namespace kvaf::cpu_push::diagnostic {
class Affinity {
 public:
  Status Bind(int cpu) {
    if (cpu == -1) return Status::OK();
    if (cpu < 0 || cpu >= CPU_SETSIZE)
      return Status::InvalidArgument("diagnostic CPU is outside CPU_SETSIZE");
    int rc = pthread_getaffinity_np(pthread_self(), sizeof(original_), &original_);
    if (rc || !CPU_ISSET(cpu, &original_))
      return Status::InvalidArgument("diagnostic CPU is outside this CPU service thread's allowed mask");
    cpu_set_t selected;
    CPU_ZERO(&selected); CPU_SET(cpu, &selected);
    rc = pthread_setaffinity_np(pthread_self(), sizeof(selected), &selected);
    if (rc) return Status::Internal("cannot set diagnostic CPU affinity, errno=" + std::to_string(rc));
    changed_ = true;
    return Status::OK();
  }
  ~Affinity() {
    if (changed_ && pthread_setaffinity_np(pthread_self(), sizeof(original_), &original_))
      std::fprintf(stderr, "[KVAF_CPU] WRITE_DIAGNOSTIC_AFFINITY_RESTORE_FAILED\n");
  }
 private:
  cpu_set_t original_{};
  bool changed_{};
};

inline std::uint64_t ThreadCpuNs() {
  timespec t{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t)) return UINT64_MAX;
  return std::uint64_t(t.tv_sec) * 1000000000 + t.tv_nsec;
}

inline int CpuNode(int cpu) {
  const auto path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
  DIR* directory = opendir(path.c_str());
  if (!directory) return -1;
  int node = -1;
  while (auto* entry = readdir(directory))
    if (std::sscanf(entry->d_name, "node%d", &node) == 1) break;
  closedir(directory);
  return node;
}

struct Memory {
  std::uint64_t kernel_page{}, mmu_page{}, anon_huge{}, vma_bytes{};
  int node_min{-1}, node_max{-1}, query_errno{}, sampled_pages{};
  std::map<int, int> node_counts;
};
inline Memory ReadMemory(void* memory, std::uint64_t bytes) {
  Memory result;
  std::ifstream input("/proc/self/smaps");
  std::string line;
  bool found = false;
  while (std::getline(input, line)) {
    unsigned long long first, last, value;
    if (std::sscanf(line.c_str(), "%llx-%llx", &first, &last) == 2) {
      if (found) break;
      const auto address = reinterpret_cast<std::uintptr_t>(memory);
      found = first <= address && address < last;
      if (found) result.vma_bytes = last - first;
    } else if (found) {
      if (std::sscanf(line.c_str(), "KernelPageSize: %llu kB", &value) == 1) result.kernel_page = value * 1024;
      if (std::sscanf(line.c_str(), "MMUPageSize: %llu kB", &value) == 1) result.mmu_page = value * 1024;
      if (std::sscanf(line.c_str(), "AnonHugePages: %llu kB", &value) == 1) result.anon_huge = value * 1024;
    }
  }
#ifdef SYS_move_pages
  std::array<void*, 64> pages{};
  std::array<int, 64> nodes{};
  const auto page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) { result.query_errno = EINVAL; return result; }
  const auto page = static_cast<std::uint64_t>(page_size);
  const auto count = std::min<std::uint64_t>(pages.size(), (bytes + page - 1) / page);
  for (std::size_t i = 0; i < count; ++i)
    pages[i] = reinterpret_cast<void*>((reinterpret_cast<std::uintptr_t>(memory) + bytes * i / count) & ~(page - 1));
  if (syscall(SYS_move_pages, 0, count, pages.data(), nullptr, nodes.data(), 0) < 0) {
    result.query_errno = errno;  // Seccomp/permissions may forbid this query.
  } else {
    for (std::size_t i = 0; i < count; ++i) {
      if (nodes[i] < 0) { result.query_errno = -nodes[i]; continue; }
      result.node_min = result.node_min < 0 ? nodes[i] : std::min(result.node_min, nodes[i]);
      result.node_max = std::max(result.node_max, nodes[i]);
      ++result.sampled_pages;
      ++result.node_counts[nodes[i]];
    }
  }
#else
  result.query_errno = ENOSYS;
#endif
  return result;
}

struct Cgroup {
  std::string path;
  std::int64_t quota_us{-2}, period_us{-1};
  std::uint64_t throttled_events{}, throttled_us{};
  bool available{};
};
inline Cgroup ReadCgroup() {
  Cgroup result;
  std::ifstream groups("/proc/self/cgroup");
  std::string line;
  while (std::getline(groups, line)) if (line.rfind("0::", 0) == 0) {
    result.path = "/sys/fs/cgroup" + line.substr(3);
    break;
  }
  if (result.path.empty()) return result;
  std::ifstream limit(result.path + "/cpu.max");
  std::string quota;
  if (limit >> quota >> result.period_us) {
    if (quota == "max") result.quota_us = -1;
    else std::istringstream(quota) >> result.quota_us;
  }
  std::ifstream stat(result.path + "/cpu.stat");
  result.available = stat.good();
  std::string key;
  std::uint64_t value;
  while (stat >> key >> value) {
    if (key == "nr_throttled") result.throttled_events = value;
    if (key == "throttled_usec") result.throttled_us = value;
  }
  return result;
}

inline std::string Environment(const Memory& memory, int cpu, const Cgroup& before, const Cgroup& after) {
  const bool same = before.available && after.available && before.path == after.path;
  const auto events = same && after.throttled_events >= before.throttled_events ?
      after.throttled_events - before.throttled_events : 0;
  const auto us = same && after.throttled_us >= before.throttled_us ? after.throttled_us - before.throttled_us : 0;
  std::ostringstream out;
  out << "{\"cpu_before\":" << cpu << ",\"cpu_numa_node\":" << CpuNode(cpu)
      << ",\"source_numa_min\":" << memory.node_min << ",\"source_numa_max\":" << memory.node_max
      << ",\"numa_query_errno\":" << memory.query_errno << ",\"source_pages_sampled\":" << memory.sampled_pages
      << ",\"kernel_page_bytes\":" << memory.kernel_page << ",\"mmu_page_bytes\":" << memory.mmu_page
      << ",\"vma_anon_huge_bytes\":" << memory.anon_huge << ",\"source_vma_bytes\":" << memory.vma_bytes
      << ",\"cgroup_v2_available\":" << (same ? 1 : 0) << ",\"cpu_quota_us\":" << before.quota_us
      << ",\"cpu_period_us\":" << before.period_us << ",\"case_throttled_events\":" << events
      << ",\"case_throttled_us\":" << us << ",\"source_numa_counts\":{";
  bool comma = false;
  for (const auto& item : memory.node_counts) {
    if (comma) out << ',';
    out << '"' << item.first << "\":" << item.second;
    comma = true;
  }
  out << "}}";
  return out.str();
}
}  // namespace kvaf::cpu_push::diagnostic
