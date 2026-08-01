#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <stdint.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static struct timespec g_kvaf_control_deadline;
static int g_kvaf_control_deadline_valid;

int kvaf_perftest_set_control_timeout(unsigned int timeout_seconds)
{
    if (timeout_seconds == 0 ||
        clock_gettime(CLOCK_MONOTONIC, &g_kvaf_control_deadline) != 0) {
        return -1;
    }
    g_kvaf_control_deadline.tv_sec += (time_t)timeout_seconds;
    g_kvaf_control_deadline_valid = 1;
    return 0;
}

static int kvaf_deadline_remaining_ms(void)
{
    struct timespec now;
    int64_t seconds;
    int64_t nanoseconds;
    int64_t milliseconds;

    if (!g_kvaf_control_deadline_valid ||
        clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        errno = ETIMEDOUT;
        return -1;
    }
    seconds = (int64_t)g_kvaf_control_deadline.tv_sec -
        (int64_t)now.tv_sec;
    nanoseconds = (int64_t)g_kvaf_control_deadline.tv_nsec -
        (int64_t)now.tv_nsec;
    milliseconds = seconds * 1000 + nanoseconds / 1000000;
    if (nanoseconds > 0 && nanoseconds % 1000000 != 0) {
        ++milliseconds;
    }
    if (milliseconds <= 0) {
        errno = ETIMEDOUT;
        return -1;
    }
    return milliseconds > INT_MAX ? INT_MAX : (int)milliseconds;
}

static int kvaf_wait_socket(int fd, short events)
{
    struct pollfd descriptor = {
        .fd = fd,
        .events = events,
        .revents = 0,
    };

    for (;;) {
        int timeout_ms = kvaf_deadline_remaining_ms();
        int result;

        if (timeout_ms < 0) {
            return -1;
        }
        result = poll(&descriptor, 1, timeout_ms);
        if (result > 0) {
            return 0;
        }
        if (result == 0) {
            errno = ETIMEDOUT;
            return -1;
        }
        if (errno != EINTR) {
            return -1;
        }
    }
}

static int kvaf_deadline_connect(
    int fd, const struct sockaddr *address, socklen_t address_length)
{
    int original_flags = fcntl(fd, F_GETFL, 0);
    int result;
    int socket_error = 0;
    socklen_t socket_error_bytes = sizeof(socket_error);

    /* Upstream retries ECONNREFUSED on the same socket.  Reject every retry
     * after the shared monotonic deadline, including failures that do not
     * enter poll(), so its fixed retry loop cannot extend the contract. */
    if (kvaf_deadline_remaining_ms() < 0 || original_flags < 0 ||
        fcntl(fd, F_SETFL, original_flags | O_NONBLOCK) != 0) {
        return -1;
    }
    result = connect(fd, address, address_length);
    if (result != 0 && errno == EINPROGRESS &&
        kvaf_wait_socket(fd, POLLOUT) == 0 &&
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error,
            &socket_error_bytes) == 0) {
        if (socket_error == 0) {
            result = 0;
        } else {
            errno = socket_error;
            result = -1;
        }
    }
    if (fcntl(fd, F_SETFL, original_flags) != 0) {
        return -1;
    }
    return result;
}

static int kvaf_deadline_usleep(useconds_t requested_microseconds)
{
    const int remaining_milliseconds = kvaf_deadline_remaining_ms();
    uint64_t remaining_microseconds;
    useconds_t bounded_microseconds = requested_microseconds;

    if (remaining_milliseconds < 0) {
        return -1;
    }
    remaining_microseconds = (uint64_t)remaining_milliseconds * 1000U;
    if ((uint64_t)bounded_microseconds > remaining_microseconds) {
        bounded_microseconds = (useconds_t)remaining_microseconds;
    }
    return usleep(bounded_microseconds);
}

static int kvaf_deadline_accept(
    int fd, struct sockaddr *address, socklen_t *address_length)
{
    int original_flags = fcntl(fd, F_GETFL, 0);
    int accepted = -1;

    if (original_flags < 0 ||
        fcntl(fd, F_SETFL, original_flags | O_NONBLOCK) != 0) {
        return -1;
    }
    for (;;) {
        accepted = accept(fd, address, address_length);
        if (accepted >= 0) {
            break;
        }
        if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
            break;
        }
        if (kvaf_wait_socket(fd, POLLIN) != 0) {
            break;
        }
    }
    if (fcntl(fd, F_SETFL, original_flags) != 0) {
        if (accepted >= 0) {
            (void)close(accepted);
        }
        return -1;
    }
    return accepted;
}

#include "kvaf_source_matched_perftest.h"
#include "perftest_mgmt_tcp.h"

#ifndef KVAF_PERFTEST_MGMT_TCP_SOURCE
#error "KVAF_PERFTEST_MGMT_TCP_SOURCE must name perftest_mgmt_tcp.c"
#endif
/*
 * UMDK 26.06 already uses a complete MSG_NOSIGNAL send loop.  Only replace
 * connect/accept so its process-global TCP context observes KVAF's one-shot
 * monotonic deadline as well.
 */
#define connect kvaf_deadline_connect
#define accept kvaf_deadline_accept
#define usleep kvaf_deadline_usleep
#include KVAF_PERFTEST_MGMT_TCP_SOURCE
#undef usleep
#undef accept
#undef connect

int kvaf_perftest_establish_connection(const perftest_config_t *cfg)
{
    comm_tcp_cfg_t tcp_cfg;
    int result;
    int saved_errno;

    if (cfg == NULL || cfg->mgmt_type != PERFTEST_MGMT_TCP) {
        errno = EINVAL;
        return -1;
    }
    tcp_cfg.server_ip = cfg->server_ip;
    tcp_cfg.bind_ip = cfg->bind_ip;
    tcp_cfg.enable_ipv6 = cfg->enable_ipv6;
    tcp_cfg.port = cfg->port;
    tcp_cfg.sock_num = cfg->pair_num;
    errno = 0;
    result = tcp_establish_connection(&tcp_cfg);
    if (result == 0) {
        return 0;
    }
    /*
     * Upstream collapses DNS/bind/connect/accept failures to -1 and its
     * cleanup can overwrite errno. Preserve the best concrete socket errno;
     * use EIO only when the upstream path supplied no errno at all.
     */
    saved_errno = errno == 0 ? EIO : errno;
    errno = saved_errno;
    return -1;
}

void kvaf_perftest_close_connection(perftest_config_t *cfg)
{
    (void)cfg;
    tcp_close_connection();
}

int kvaf_perftest_sync_data(const perftest_config_t *cfg, uint32_t index,
    int size, char *local_data, char *remote_data)
{
    if (cfg == NULL || cfg->mgmt_type != PERFTEST_MGMT_TCP) {
        errno = EINVAL;
        return -1;
    }
    return tcp_sync_data(index, size, local_data, remote_data);
}

/* Used only for SO_RCVTIMEO/SO_SNDTIMEO and emergency shutdown. */
int kvaf_perftest_control_fd(uint32_t index)
{
    if (comm_ctx.sock_fd == NULL || index >= comm_ctx.sock_num) {
        errno = EINVAL;
        return -1;
    }
    return comm_ctx.sock_fd[index];
}

/* Official perftest_resources.c consumes the generic management symbols. */
int sync_data(const perftest_config_t *cfg, uint32_t index, int size,
    char *local_data, char *remote_data)
{
    return kvaf_perftest_sync_data(
        cfg, index, size, local_data, remote_data);
}

int sync_time(const perftest_config_t *cfg, uint32_t index, const char *value)
{
    if (cfg == NULL || cfg->mgmt_type != PERFTEST_MGMT_TCP) {
        errno = EINVAL;
        return -1;
    }
    return tcp_sync_time(index, value);
}

ssize_t comm_send(const perftest_config_t *cfg, uint32_t index,
    const void *buffer, size_t bytes)
{
    if (cfg == NULL || cfg->mgmt_type != PERFTEST_MGMT_TCP) {
        errno = EINVAL;
        return -1;
    }
    return tcp_comm_send(index, buffer, bytes);
}

ssize_t comm_recv(const perftest_config_t *cfg, uint32_t index,
    void *buffer, size_t bytes)
{
    if (cfg == NULL || cfg->mgmt_type != PERFTEST_MGMT_TCP) {
        errno = EINVAL;
        return -1;
    }
    return tcp_comm_recv(index, buffer, bytes);
}

int comm_poll(const perftest_config_t *cfg, uint32_t index, int timeout_ms)
{
    if (cfg == NULL || cfg->mgmt_type != PERFTEST_MGMT_TCP) {
        errno = EINVAL;
        return -1;
    }
    return tcp_comm_poll(index, timeout_ms);
}
