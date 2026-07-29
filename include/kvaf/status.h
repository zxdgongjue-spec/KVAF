#pragma once

#include <string>
#include <utility>

namespace kvaf {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kOutOfRange,
  kNotInitialized,
  kNotImplemented,
  kOutOfMemory,
  kUnavailable,
  kTimeout,
  kTransportError,
  kInternal,
};

class Status {
 public:
  Status() = default;

  static Status OK() { return Status(); }

  static Status Error(StatusCode code, std::string message) {
    return Status(code, std::move(message));
  }

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

 private:
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  StatusCode code_{StatusCode::kOk};
  std::string message_;
};

}  // namespace kvaf
