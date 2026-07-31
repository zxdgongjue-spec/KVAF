#include <errno.h>
#include <sys/socket.h>
#include <unistd.h>

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
#include KVAF_PERFTEST_COMMUNICATION_SOURCE
#undef write
