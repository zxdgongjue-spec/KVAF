#pragma once

#include <string>
#include <utility>

namespace kvaf {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kNotInitialized,
  kNotImplemented,
  kNotFound,
  kAlreadyExists,
  kBusy,
  kInvalidState,
  kOutOfMemory,
  kInternal,
  kCompletionUnknown,
};

class Status {
 public:
  Status() = default;

  static Status OK() { return Status(); }

  static Status Error(StatusCode code, std::string message) {
    return Status(code, std::move(message));
  }

  static Status InvalidArgument(std::string message) {
    return Error(StatusCode::kInvalidArgument, std::move(message));
  }

  static Status NotInitialized(std::string message) {
    return Error(StatusCode::kNotInitialized, std::move(message));
  }

  static Status NotImplemented(std::string message) {
    return Error(StatusCode::kNotImplemented, std::move(message));
  }

  static Status NotFound(std::string message) {
    return Error(StatusCode::kNotFound, std::move(message));
  }

  static Status AlreadyExists(std::string message) {
    return Error(StatusCode::kAlreadyExists, std::move(message));
  }

  static Status Busy(std::string message) {
    return Error(StatusCode::kBusy, std::move(message));
  }

  static Status InvalidState(std::string message) {
    return Error(StatusCode::kInvalidState, std::move(message));
  }

  static Status OutOfMemory(std::string message) {
    return Error(StatusCode::kOutOfMemory, std::move(message));
  }

  static Status Internal(std::string message) {
    return Error(StatusCode::kInternal, std::move(message));
  }

  static Status CompletionUnknown(std::string message) {
    return Error(StatusCode::kCompletionUnknown, std::move(message));
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
