#ifndef KVAF_UB_CLOSE_POLICY_H
#define KVAF_UB_CLOSE_POLICY_H

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kvaf_ub_send_tracker {
  uint32_t inflight;
  bool accounting_unknown;
  bool flush_barrier_required;
  bool flush_barrier_seen;
};

enum kvaf_ub_close_retry_class {
  KVAF_UB_CLOSE_RETRY_NONE = 0,
  KVAF_UB_CLOSE_RETRY_CONTROL,
  KVAF_UB_CLOSE_RETRY_CHECKED_DESTROY,
};

static inline void kvaf_ub_send_tracker_reset(
    struct kvaf_ub_send_tracker *tracker)
{
  tracker->inflight = 0;
  tracker->accounting_unknown = false;
  tracker->flush_barrier_required = false;
  tracker->flush_barrier_seen = false;
}

static inline void kvaf_ub_send_tracker_require_flush_barrier(
    struct kvaf_ub_send_tracker *tracker)
{
  /* A successfully returned management Jetty is the first point at which a
   * send queue can own a WQE and ERROR can produce WR_FLUSH_ERR_DONE.  Earlier
   * partial-create failures (for example JFC/JFR only) must not wait for an
   * impossible send-side completion. */
  tracker->flush_barrier_required = true;
}

static inline void kvaf_ub_send_tracker_record_post(
    struct kvaf_ub_send_tracker *tracker, bool completion_requested)
{
  if (!completion_requested || tracker->inflight == UINT32_MAX) {
    tracker->accounting_unknown = true;
    return;
  }
  ++tracker->inflight;
}

static inline void kvaf_ub_send_tracker_mark_post_unknown(
    struct kvaf_ub_send_tracker *tracker)
{
  tracker->accounting_unknown = true;
}

static inline void kvaf_ub_send_tracker_record_completion(
    struct kvaf_ub_send_tracker *tracker, bool flush_barrier)
{
  if (flush_barrier) {
    /* A flush completion is a barrier for every older SQ WQE. */
    tracker->inflight = 0;
    tracker->accounting_unknown = false;
    tracker->flush_barrier_seen = true;
    return;
  }
  if (tracker->inflight != 0U) {
    --tracker->inflight;
  } else if (!tracker->accounting_unknown) {
    /* An unaccounted send CQE means the exact-count proof was incomplete. */
    tracker->accounting_unknown = true;
  }
}

static inline bool kvaf_ub_send_tracker_is_quiesced(
    const struct kvaf_ub_send_tracker *tracker)
{
  return tracker->inflight == 0U && !tracker->accounting_unknown;
}

static inline bool kvaf_ub_send_tracker_close_is_quiesced(
    const struct kvaf_ub_send_tracker *tracker)
{
  /* The management JFC is shared by the send Jetty and receive JFR.  An
   * empty poll is only a transient observation; after both queues enter
   * ERROR, the send-side WR_FLUSH_ERR_DONE marker is the durable barrier
   * required before receive buffers can be unregistered as well.  Before a
   * Jetty was ever created there can be no management WQE, and therefore no
   * possible send-side barrier to wait for. */
  return kvaf_ub_send_tracker_is_quiesced(tracker) &&
      (!tracker->flush_barrier_required || tracker->flush_barrier_seen);
}

static inline bool kvaf_ub_close_retry_is_proven(
    int operation_status, int operation_errno,
    enum kvaf_ub_close_retry_class retry_class,
    bool explicit_retry_proven, bool state_ambiguous, bool timed_out)
{
  const bool try_again = operation_status == EAGAIN ||
      operation_status == EBUSY || operation_errno == EAGAIN ||
      operation_errno == EBUSY;

  if (state_ambiguous || timed_out ||
      retry_class == KVAF_UB_CLOSE_RETRY_NONE) {
    return false;
  }
  if (explicit_retry_proven) {
    return true;
  }
  if (retry_class == KVAF_UB_CLOSE_RETRY_CHECKED_DESTROY) {
    /* EINTR is not failure-atomic for a destructive ioctl. */
    return try_again;
  }
  return try_again || operation_status == EINTR ||
      operation_errno == EINTR;
}

#ifdef __cplusplus
}
#endif

#endif
