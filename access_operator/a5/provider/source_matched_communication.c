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

    if (original_flags < 0 ||
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

/*
 * The upstream helper performs one write(2) for an entire protocol frame.
 * Besides permitting a short write, that can terminate the resource-owning
 * process with SIGPIPE after the peer disappears.  Compile the selected
 * communication source through this wrapper so every one of its writes is a
 * complete, no-SIGPIPE socket send.
 */
static ssize_t kvaf_full_socket_write(
    int fd, const void *buffer, size_t bytes)
{
    const unsigned char *cursor = buffer;
    size_t written = 0;

    while (written < bytes) {
        ssize_t result = send(
            fd, cursor + written, bytes - written, MSG_NOSIGNAL);
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            return -1;
        }
        written += (size_t)result;
    }
    return (ssize_t)written;
}

#ifndef KVAF_PERFTEST_COMMUNICATION_SOURCE
#error "KVAF_PERFTEST_COMMUNICATION_SOURCE must name perftest_communication.c"
#endif
#define write kvaf_full_socket_write
#define connect kvaf_deadline_connect
#define accept kvaf_deadline_accept
#include KVAF_PERFTEST_COMMUNICATION_SOURCE
#undef accept
#undef connect
#undef write
