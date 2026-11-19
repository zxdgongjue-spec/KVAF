#pragma once

#include <chrono>
#include "requests.h"

namespace kvaf {
inline std::uint64_t PutHostTimingStart(const PutProviderTiming* timing) {
  return timing ? std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count() : 0;
}
inline void PutHostTimingRecord(PutProviderTiming* timing,
                                std::uint64_t PutProviderTiming::*field,
                                std::uint64_t started) {
  if (timing) timing->*field += PutHostTimingStart(timing) - started;
}
// Stack-only timer: includes failure exits and can stop before the next phase.
class PutHostTimingScope {
 public:
  PutHostTimingScope(PutProviderTiming* timing, std::uint64_t PutProviderTiming::*field)
      : timing_(timing), field_(field), started_(PutHostTimingStart(timing)) {}
  ~PutHostTimingScope() { Finish(); }
  PutHostTimingScope(const PutHostTimingScope&) = delete;
  PutHostTimingScope& operator=(const PutHostTimingScope&) = delete;
  void Finish() {
    PutHostTimingRecord(timing_, field_, started_);
    timing_ = nullptr;
  }
 private:
  PutProviderTiming* timing_;
  std::uint64_t PutProviderTiming::*field_;
  std::uint64_t started_;
};
}  // namespace kvaf
