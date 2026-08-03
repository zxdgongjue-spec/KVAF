#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kvaf/access_operator/access_operator.h"
#include "kvaf/meta_manager/meta_manager.h"
#include "kvaf/runtime/runtime.h"

namespace {

using kvaf::AccessOperator;
using kvaf::Address;
using kvaf::BatchGetExecRequest;
using kvaf::BatchPutExecRequest;
using kvaf::ConstBufferView;
using kvaf::DeviceKind;
using kvaf::MutableBufferView;
using kvaf::RemoteRegionDesc;
using kvaf::Status;
using kvaf::StatusCode;

Status InvalidArgument(const std::string& message) {
  return Status::Error(StatusCode::kInvalidArgument, message);
}

Status Internal(const std::string& message) {
  return Status::Error(StatusCode::kInternal, message);
}

class MockAccessOperator final : public AccessOperator {
 public:
  Status Initialize(const RemoteRegionDesc& remote_region) override {
    if (initialized_) return Status::Error(StatusCode::kInvalidState, "already initialized");
    if (remote_region.base == 0 || remote_region.length == 0) {
      return InvalidArgument("invalid remote region");
    }
    initialized_ = true;
    return Status::OK();
  }

  Status Shutdown() override {
    if (!initialized_) return Status::Error(StatusCode::kNotInitialized, "not initialized");
    initialized_ = false;
    return Status::OK();
  }

  Status ExecuteBatchGet(const BatchGetExecRequest& exec) override {
    if (fail_next_get_) {
      fail_next_get_ = false;
      return Internal("configured mock BatchGet failure");
    }
    if (!initialized_ || exec.request == nullptr || exec.host_access_table.addresses == nullptr ||
        exec.host_access_table.lengths == nullptr || exec.host_access_table.valid == nullptr) {
      return InvalidArgument("invalid mock BatchGet request");
    }
    const kvaf::BatchGetRequest& request = *exec.request;
    if (request.indices.device != DeviceKind::kCpu ||
        request.output_indices.device != DeviceKind::kCpu ||
        request.output.device != DeviceKind::kCpu) {
      return InvalidArgument("mock operator supports CPU buffers only");
    }

    const auto count = request.indices.bytes / sizeof(std::int32_t);
    const auto* indices = static_cast<const std::int32_t*>(request.indices.data);
    const auto* output_indices =
        static_cast<const std::int32_t*>(request.output_indices.data);
    const auto* mask = request.mask.data == nullptr
                           ? nullptr
                           : static_cast<const std::uint8_t*>(request.mask.data);
    auto* output = static_cast<std::uint8_t*>(request.output.data);
    const std::size_t output_rows =
        request.output.bytes / request.output_block_bytes;

    for (std::size_t i = 0; i < count; ++i) {
      if (mask != nullptr && mask[i] == 0) continue;
      if (indices[i] < 0 ||
          static_cast<std::size_t>(indices[i]) >= exec.host_access_table.size ||
          exec.host_access_table.valid[indices[i]] == 0) {
        return InvalidArgument("BatchGet index is outside the valid AccessTable");
      }
      if (output_indices[i] < 0 ||
          static_cast<std::size_t>(output_indices[i]) >= output_rows) {
        return InvalidArgument("BatchGet output index is outside the output buffer");
      }
      const std::size_t table_index = static_cast<std::size_t>(indices[i]);
      const std::uint64_t length = exec.host_access_table.lengths[table_index];
      if (length > request.output_block_bytes) {
        return InvalidArgument("BatchGet output block is smaller than source data");
      }
      std::uint8_t* destination =
          output + static_cast<std::size_t>(output_indices[i]) * request.output_block_bytes;
      std::memcpy(destination,
                  reinterpret_cast<const void*>(exec.host_access_table.addresses[table_index]),
                  static_cast<std::size_t>(length));
    }
    return Status::OK();
  }

  Status ExecuteBatchPut(const BatchPutExecRequest& exec) override {
    if (fail_next_put_) {
      fail_next_put_ = false;
      return Internal("configured mock BatchPut failure");
    }
    if (!initialized_ || exec.request == nullptr || exec.put_desc == nullptr ||
        exec.request->keys.size() != exec.request->blocks.size() ||
        exec.request->keys.size() != exec.put_desc->entries.size()) {
      return InvalidArgument("invalid mock BatchPut request");
    }
    for (std::size_t i = 0; i < exec.request->blocks.size(); ++i) {
      const ConstBufferView& source = exec.request->blocks[i];
      const kvaf::PutAllocation& target = exec.put_desc->entries[i];
      if (source.device != DeviceKind::kCpu || source.bytes != target.reserved_length ||
          source.data == nullptr || target.key != exec.request->keys[i]) {
        return InvalidArgument("invalid mock BatchPut block");
      }
      std::memcpy(reinterpret_cast<void*>(target.address), source.data, source.bytes);
    }
    return Status::OK();
  }

  void FailNextPut() { fail_next_put_ = true; }
  void FailNextGet() { fail_next_get_ = true; }

 private:
  bool initialized_{false};
  bool fail_next_put_{false};
  bool fail_next_get_{false};
};

std::vector<std::string> Split(const std::string& line) {
  std::istringstream input(line);
  std::vector<std::string> words;
  std::string word;
  while (input >> word) words.push_back(word);
  return words;
}

bool ParseU64(const std::string& text, std::uint64_t* value) {
  if (value == nullptr || text.empty()) return false;
  std::size_t consumed = 0;
  try {
    *value = std::stoull(text, &consumed, 0);
  } catch (const std::exception&) {
    return false;
  }
  return consumed == text.size();
}

bool ParseI32(const std::string& text, std::int32_t* value) {
  if (value == nullptr || text.empty()) return false;
  std::size_t consumed = 0;
  try {
    const long long parsed = std::stoll(text, &consumed, 0);
    if (parsed < INT32_MIN || parsed > INT32_MAX) return false;
    *value = static_cast<std::int32_t>(parsed);
  } catch (const std::exception&) {
    return false;
  }
  return consumed == text.size();
}

bool ParseHex(const std::string& text, std::vector<std::uint8_t>* bytes) {
  if (bytes == nullptr || text.size() % 2 != 0) return false;
  bytes->clear();
  bytes->reserve(text.size() / 2);
  for (std::size_t i = 0; i < text.size(); i += 2) {
    const char high = text[i];
    const char low = text[i + 1];
    if (!std::isxdigit(static_cast<unsigned char>(high)) ||
        !std::isxdigit(static_cast<unsigned char>(low))) {
      return false;
    }
    const unsigned int value = static_cast<unsigned int>(std::stoul(text.substr(i, 2), nullptr, 16));
    bytes->push_back(static_cast<std::uint8_t>(value));
  }
  return true;
}

std::string Hex(const std::vector<std::uint8_t>& bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2);
  for (std::uint8_t byte : bytes) {
    result.push_back(kHex[byte >> 4]);
    result.push_back(kHex[byte & 0x0f]);
  }
  return result;
}

void PrintStatus(const Status& status, const std::string& ok_message = "") {
  if (status.ok()) {
    std::cout << "OK";
    if (!ok_message.empty()) std::cout << ' ' << ok_message;
    std::cout << '\n';
  } else {
    std::cout << "ERR code=" << static_cast<int>(status.code()) << '\n';
  }
  std::cout.flush();
}

bool ParseGetItem(const std::string& text, std::int32_t* index,
                  std::int32_t* output_index, std::uint8_t* mask) {
  const std::size_t first = text.find(':');
  const std::size_t second = text.find(':', first == std::string::npos ? first : first + 1);
  if (first == std::string::npos || second == std::string::npos ||
      text.find(':', second + 1) != std::string::npos) {
    return false;
  }
  std::uint64_t mask_value = 0;
  return ParseI32(text.substr(0, first), index) &&
         ParseI32(text.substr(first + 1, second - first - 1), output_index) &&
         ParseU64(text.substr(second + 1), &mask_value) && mask_value <= 1 &&
         ((*mask = static_cast<std::uint8_t>(mask_value)), true);
}

}  // namespace

int main() {
  auto meta_manager = kvaf::CreateInMemoryMetaManager();
  auto allocator = kvaf::CreateHostDdrMemoryAllocator();
  MockAccessOperator access_operator;
  kvaf::Runtime runtime(*meta_manager, access_operator, *allocator);
  std::unordered_map<std::string, kvaf::AccessTableHandle> handles;

  std::string line;
  while (std::getline(std::cin, line)) {
    const std::vector<std::string> words = Split(line);
    if (words.empty() || words[0][0] == '#') continue;

    if (words[0] == "QUIT") {
      PrintStatus(Status::OK(), "QUIT");
      return 0;
    }
    if (words[0] == "INIT" && words.size() == 3) {
      std::uint64_t bytes = 0;
      std::uint64_t slot_size = 0;
      if (!ParseU64(words[1], &bytes) || !ParseU64(words[2], &slot_size)) {
        PrintStatus(InvalidArgument("invalid INIT arguments"));
        continue;
      }
      PrintStatus(runtime.Initialize(kvaf::RuntimeInitRequest{bytes, slot_size, DeviceKind::kCpu}),
                  "INIT");
      continue;
    }
    if (words[0] == "SHUTDOWN" && words.size() == 1) {
      PrintStatus(runtime.Shutdown(), "SHUTDOWN");
      continue;
    }
    if (words[0] == "FAIL_NEXT_PUT" && words.size() == 1) {
      access_operator.FailNextPut();
      PrintStatus(Status::OK(), "FAIL_NEXT_PUT");
      continue;
    }
    if (words[0] == "FAIL_NEXT_GET" && words.size() == 1) {
      access_operator.FailNextGet();
      PrintStatus(Status::OK(), "FAIL_NEXT_GET");
      continue;
    }
    if (words[0] == "PUT" && words.size() >= 3 && words.size() % 2 == 1) {
      kvaf::BatchPutRequest request;
      std::vector<std::vector<std::uint8_t>> payloads;
      bool valid = true;
      for (std::size_t i = 1; i < words.size(); i += 2) {
        std::uint64_t key = 0;
        std::vector<std::uint8_t> payload;
        if (!ParseU64(words[i], &key) || !ParseHex(words[i + 1], &payload) ||
            payload.empty()) {
          valid = false;
          break;
        }
        request.keys.push_back(key);
        payloads.push_back(std::move(payload));
      }
      if (!valid) {
        PrintStatus(InvalidArgument("invalid PUT arguments"));
        continue;
      }
      request.blocks.reserve(payloads.size());
      for (const auto& payload : payloads) {
        request.blocks.push_back(
            ConstBufferView{payload.data(), payload.size(), DeviceKind::kCpu});
      }
      PrintStatus(runtime.BatchPut(request), "PUT");
      continue;
    }
    if (words[0] == "PREPARE" && words.size() >= 3) {
      kvaf::AccessTablePrepareRequest request;
      bool valid = true;
      for (std::size_t i = 2; i < words.size(); ++i) {
        std::uint64_t key = 0;
        if (!ParseU64(words[i], &key)) {
          valid = false;
          break;
        }
        request.flat_keys.push_back(key);
      }
      if (!valid) {
        PrintStatus(InvalidArgument("invalid PREPARE arguments"));
        continue;
      }
      kvaf::AccessTableHandle handle;
      const Status status = runtime.PrepareAccessTable(request, &handle);
      if (status.ok()) handles[words[1]] = handle;
      PrintStatus(status, status.ok() ? "PREPARE" : "");
      continue;
    }
    if (words[0] == "VIEW" && words.size() == 2) {
      const auto handle = handles.find(words[1]);
      if (handle == handles.end()) {
        PrintStatus(Status::Error(StatusCode::kNotFound, "handle not found"));
        continue;
      }
      kvaf::AccessTableView view;
      const Status status = runtime.GetAccessTableView(handle->second, &view);
      if (!status.ok()) {
        PrintStatus(status);
        continue;
      }
      std::ostringstream output;
      output << "VIEW";
      for (std::size_t i = 0; i < view.size; ++i) {
        output << (i == 0 ? " " : ",") << view.addresses[i] << ':'
               << view.lengths[i] << ':' << static_cast<unsigned int>(view.valid[i]);
      }
      PrintStatus(Status::OK(), output.str());
      continue;
    }
    if (words[0] == "RELEASE" && words.size() == 2) {
      const auto handle = handles.find(words[1]);
      if (handle == handles.end()) {
        PrintStatus(Status::Error(StatusCode::kNotFound, "handle not found"));
        continue;
      }
      const Status status = runtime.ReleaseAccessTable(handle->second);
      if (status.ok()) handles.erase(handle);
      PrintStatus(status, status.ok() ? "RELEASE" : "");
      continue;
    }
    if (words[0] == "FREE" && words.size() >= 2) {
      kvaf::BatchFreeRequest request;
      bool valid = true;
      for (std::size_t i = 1; i < words.size(); ++i) {
        std::uint64_t key = 0;
        if (!ParseU64(words[i], &key)) {
          valid = false;
          break;
        }
        request.keys.push_back(key);
      }
      PrintStatus(valid ? runtime.BatchFree(request)
                        : InvalidArgument("invalid FREE arguments"),
                  valid ? "FREE" : "");
      continue;
    }
    if (words[0] == "GET" && words.size() >= 4) {
      const auto handle = handles.find(words[1]);
      std::uint64_t block_bytes = 0;
      if (handle == handles.end() || !ParseU64(words[2], &block_bytes) || block_bytes == 0) {
        PrintStatus(InvalidArgument("invalid GET header"));
        continue;
      }
      std::vector<std::int32_t> indices;
      std::vector<std::int32_t> output_indices;
      std::vector<std::uint8_t> masks;
      std::int32_t largest_output_index = -1;
      bool valid = true;
      for (std::size_t i = 3; i < words.size(); ++i) {
        std::int32_t index = 0;
        std::int32_t output_index = 0;
        std::uint8_t mask = 0;
        if (!ParseGetItem(words[i], &index, &output_index, &mask)) {
          valid = false;
          break;
        }
        indices.push_back(index);
        output_indices.push_back(output_index);
        masks.push_back(mask);
        largest_output_index = std::max(largest_output_index, output_index);
      }
      if (!valid || largest_output_index < 0) {
        PrintStatus(InvalidArgument("invalid GET items"));
        continue;
      }
      std::vector<std::uint8_t> output(
          static_cast<std::size_t>(largest_output_index + 1) * block_bytes, 0);
      kvaf::BatchGetRequest request;
      request.access_table = handle->second;
      request.indices = ConstBufferView{indices.data(), indices.size() * sizeof(indices[0]),
                                        DeviceKind::kCpu};
      request.mask = ConstBufferView{masks.data(), masks.size(), DeviceKind::kCpu};
      request.output_indices = ConstBufferView{output_indices.data(),
                                               output_indices.size() * sizeof(output_indices[0]),
                                               DeviceKind::kCpu};
      request.output = MutableBufferView{output.data(), output.size(), DeviceKind::kCpu};
      request.output_block_bytes = block_bytes;
      const Status status = runtime.BatchGet(request);
      PrintStatus(status, status.ok() ? "GET " + Hex(output) : "");
      continue;
    }
    PrintStatus(InvalidArgument("unknown command"));
  }
  return 0;
}
