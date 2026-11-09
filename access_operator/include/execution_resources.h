#pragma once

#include <memory>
#include "status.h"

namespace kvaf {

// Owns a provider's communication sessions, mappings, workspaces and reusable
// execution slots. The resource manager, not a borrowed operator, closes it.
class ExecutionResourceOwner {
 public:
  virtual ~ExecutionResourceOwner() = default;
  virtual Status Close() = 0;
};

// Provider-specific state for one invocation/graph. Physical workspace belongs
// to its exclusively leased resource slot; pending completion belongs here.
class ExecutionState {
 public:
  virtual ~ExecutionState() = default;
};

enum class PutOutcome { Unknown, Succeeded, Failed };
struct PutCompletion {
  PutOutcome outcome{PutOutcome::Unknown};
  Status result{Status::CompletionUnknown("Put completion unresolved")};
};

}  // namespace kvaf
