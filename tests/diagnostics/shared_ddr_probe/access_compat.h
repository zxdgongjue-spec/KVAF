#pragma once
#include <urma_api.h>

// This repository pins br_UMDK_24.12.0_LTS. Newer LOCAL_ONLY headers have
// different semantics; accepting them silently invalidates the experiment.
#if !defined(URMA_ACCESS_LOCAL_WRITE) || defined(URMA_ACCESS_LOCAL_ONLY)
#error "Use br_UMDK_24.12.0_LTS headers (LOCAL_WRITE); LOCAL_ONLY is a different access model"
#endif
constexpr const char* kHeaderAccessModel = "br_UMDK_24.12.0_LTS: LOCAL_WRITE";
constexpr unsigned kLocalWrite = URMA_ACCESS_LOCAL_WRITE;
constexpr unsigned kRemoteAccess = URMA_ACCESS_REMOTE_READ | URMA_ACCESS_REMOTE_WRITE |
                                   URMA_ACCESS_REMOTE_ATOMIC;
static_assert(kLocalWrite == 1 && kRemoteAccess == 14, "Unexpected UMDK access bits");
