#pragma once
#include <urma_api.h>
#include "offload_service.h"
namespace kvaf::offload {
std::unique_ptr<Backend> CreateUrmaBackend(urma_context_t* context);
}
