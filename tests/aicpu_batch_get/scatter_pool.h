#pragma once

#include "abi.h"
#include <cerrno>
#include <pthread.h>

namespace kvaf::aicpu_test {
// The caller is worker 0; N-1 persistent pthreads execute the other ranges.
// Initialize/Shutdown are outside round timing. Dispatch and the completion
// barrier are inside it. Work must not throw and must complete its cache ops.
class ScatterPool {
 public:
  using Work = void (*)(void*, std::uint32_t);
  ~ScatterPool() { Shutdown(); }
  ScatterPool() = default;
  ScatterPool(const ScatterPool&) = delete;
  ScatterPool& operator=(const ScatterPool&) = delete;

  int Initialize(std::uint32_t count, Work work, void* context) {
    if (initialized_ || count == 0 || count > kMaxScatterThreads || !work) return EINVAL;
    count_ = count; work_ = work; context_ = context;
    if (count == 1) { initialized_ = true; return 0; }
    int rc = pthread_mutex_init(&mutex_, nullptr);
    if (rc) { Shutdown(); return rc; }
    mutex_ready_ = true;
    rc = pthread_cond_init(&work_ready_, nullptr);
    if (rc) { Shutdown(); return rc; }
    work_ready_initialized_ = true;
    rc = pthread_cond_init(&done_, nullptr);
    if (rc) { Shutdown(); return rc; }
    done_initialized_ = true;
    for (std::uint32_t i = 1; i < count; ++i) {
      workers_[i - 1] = {this, i};
      rc = pthread_create(&threads_[i - 1], nullptr, &Entry, &workers_[i - 1]);
      if (rc) { Shutdown(); return rc; }
      ++created_;
    }
    initialized_ = true;
    return 0;
  }

  void Run() {
    if (count_ == 1) { work_(context_, 0); return; }
    pthread_mutex_lock(&mutex_);
    pending_ = created_;
    ++generation_;
    pthread_cond_broadcast(&work_ready_);
    pthread_mutex_unlock(&mutex_);
    work_(context_, 0);
    pthread_mutex_lock(&mutex_);
    while (pending_ != 0) pthread_cond_wait(&done_, &mutex_);
    pthread_mutex_unlock(&mutex_);
  }

  void Shutdown() {
    if (created_) {
      pthread_mutex_lock(&mutex_);
      stop_ = true;
      pthread_cond_broadcast(&work_ready_);
      pthread_mutex_unlock(&mutex_);
      for (std::uint32_t i = 0; i < created_; ++i) pthread_join(threads_[i], nullptr);
    }
    if (done_initialized_) pthread_cond_destroy(&done_);
    if (work_ready_initialized_) pthread_cond_destroy(&work_ready_);
    if (mutex_ready_) pthread_mutex_destroy(&mutex_);
    initialized_ = mutex_ready_ = work_ready_initialized_ = done_initialized_ = false;
    stop_ = false; created_ = pending_ = 0; generation_ = 0;
  }

 private:
  struct Worker { ScatterPool* pool; std::uint32_t index; };
  static void* Entry(void* argument) {
    auto* worker = static_cast<Worker*>(argument);
    worker->pool->Loop(worker->index);
    return nullptr;
  }
  void Loop(std::uint32_t index) {
    std::uint64_t observed = 0;
    pthread_mutex_lock(&mutex_);
    for (;;) {
      while (!stop_ && generation_ == observed) pthread_cond_wait(&work_ready_, &mutex_);
      if (stop_) break;
      observed = generation_;
      pthread_mutex_unlock(&mutex_);
      work_(context_, index);
      pthread_mutex_lock(&mutex_);
      if (--pending_ == 0) pthread_cond_signal(&done_);
    }
    pthread_mutex_unlock(&mutex_);
  }
  Work work_{};
  void* context_{};
  std::uint32_t count_{1}, created_{}, pending_{};
  std::uint64_t generation_{};
  bool initialized_{}, mutex_ready_{}, work_ready_initialized_{}, done_initialized_{}, stop_{};
  pthread_mutex_t mutex_{};
  pthread_cond_t work_ready_{}, done_{};
  pthread_t threads_[kMaxScatterThreads - 1]{};
  Worker workers_[kMaxScatterThreads - 1]{};
};
}  // namespace kvaf::aicpu_test
