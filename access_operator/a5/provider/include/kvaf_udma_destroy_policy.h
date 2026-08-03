#ifndef KVAF_UDMA_DESTROY_POLICY_H
#define KVAF_UDMA_DESTROY_POLICY_H

#include <errno.h>
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>

enum kvaf_udma_destroy_decision {
    KVAF_UDMA_DESTROY_ISSUE_IOCTL = 0,
    KVAF_UDMA_DESTROY_DRAIN_ASYNC,
    KVAF_UDMA_DESTROY_RETRY_LATER,
    KVAF_UDMA_DESTROY_RELEASE_LOCAL,
    KVAF_UDMA_DESTROY_FAIL_UNSAFE,
};

/* Direct KVAF JFCs reject JFCEs.  Completion-event debt therefore has no
 * legal consumer and cannot be treated as a retryable wait. */
static inline enum kvaf_udma_destroy_decision kvaf_udma_destroy_decide(
    bool kernel_created, bool delete_pending,
    uint32_t comp_events_reported, uint32_t async_events_reported,
    uint32_t comp_events_acked, uint32_t async_events_acked,
    bool drain_attempted)
{
    if (kernel_created && delete_pending) {
        return KVAF_UDMA_DESTROY_FAIL_UNSAFE;
    }
    if (kernel_created) {
        return KVAF_UDMA_DESTROY_ISSUE_IOCTL;
    }
    if (!delete_pending) {
        return KVAF_UDMA_DESTROY_RELEASE_LOCAL;
    }
    if (comp_events_acked > comp_events_reported ||
        async_events_acked > async_events_reported) {
        return KVAF_UDMA_DESTROY_FAIL_UNSAFE;
    }
    if (comp_events_acked != comp_events_reported) {
        return KVAF_UDMA_DESTROY_FAIL_UNSAFE;
    }
    if (async_events_acked != async_events_reported) {
        return drain_attempted ? KVAF_UDMA_DESTROY_RETRY_LATER
                               : KVAF_UDMA_DESTROY_DRAIN_ASYNC;
    }
    return KVAF_UDMA_DESTROY_RELEASE_LOCAL;
}

enum kvaf_udma_async_poll_decision {
    KVAF_UDMA_ASYNC_POLL_IDLE = 0,
    KVAF_UDMA_ASYNC_POLL_READ_ONE,
    KVAF_UDMA_ASYNC_POLL_RETRY_LATER,
    KVAF_UDMA_ASYNC_POLL_FAIL_UNSAFE,
};

static inline enum kvaf_udma_async_poll_decision
kvaf_udma_async_poll_decide(
    int poll_status, short revents, int poll_errno)
{
    const short unsafe_events = POLLERR | POLLHUP | POLLNVAL;

    if (poll_status < 0) {
        return poll_errno == EINTR ? KVAF_UDMA_ASYNC_POLL_RETRY_LATER
                                   : KVAF_UDMA_ASYNC_POLL_FAIL_UNSAFE;
    }
    if (poll_status == 0) {
        return revents == 0 ? KVAF_UDMA_ASYNC_POLL_IDLE
                            : KVAF_UDMA_ASYNC_POLL_FAIL_UNSAFE;
    }
    if (poll_status != 1 || (revents & unsafe_events) != 0 ||
        revents != POLLIN) {
        return KVAF_UDMA_ASYNC_POLL_FAIL_UNSAFE;
    }
    return KVAF_UDMA_ASYNC_POLL_READ_ONE;
}

#endif
