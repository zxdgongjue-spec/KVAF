#define _GNU_SOURCE

#include "kvaf_source_matched_ummu_fd.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef ENOTUNIQ
#define ENOTUNIQ 76
#endif

#define KVAF_UMMU_DUP_MIN_FD 3

static bool kvaf_parse_fd_name(const char *name, int *fd_out)
{
  char *end = NULL;
  long value;

  if (name == NULL || fd_out == NULL || name[0] == '\0') {
    return false;
  }
  errno = 0;
  value = strtol(name, &end, 10);
  if (errno != 0 || end == name || *end != '\0' ||
      value < 0 || value > INT_MAX) {
    return false;
  }
  *fd_out = (int)value;
  return true;
}

static bool kvaf_fd_matches_device(
    int fd, const struct stat *device_status)
{
  struct stat candidate_status;
  int descriptor_flags;
  int file_flags;

  if (fd < 0 || device_status == NULL ||
      fstat(fd, &candidate_status) != 0 ||
      !S_ISCHR(candidate_status.st_mode) ||
      candidate_status.st_dev != device_status->st_dev ||
      candidate_status.st_ino != device_status->st_ino ||
      candidate_status.st_rdev != device_status->st_rdev) {
    return false;
  }
  file_flags = fcntl(fd, F_GETFL);
  descriptor_flags = fcntl(fd, F_GETFD);
  if (file_flags < 0 || descriptor_flags < 0 ||
      (file_flags & O_ACCMODE) != O_RDWR ||
      (descriptor_flags & FD_CLOEXEC) == 0) {
    return false;
  }
#ifdef O_PATH
  if ((file_flags & O_PATH) != 0) {
    return false;
  }
#endif
  return true;
}

int kvaf_source_ummu_duplicate_unique_process_fd(
    const char *device_path, const char *fd_dir)
{
  struct stat device_status;
  struct dirent *entry;
  DIR *directory;
  int adopted_fd = -1;
  int directory_fd;
  int result = -ENOENT;
  int scan_error;

  if (device_path == NULL || fd_dir == NULL) {
    return -EINVAL;
  }
  if (stat(device_path, &device_status) != 0) {
    return -errno;
  }
  if (!S_ISCHR(device_status.st_mode)) {
    return -ENODEV;
  }
  directory = opendir(fd_dir);
  if (directory == NULL) {
    return -errno;
  }
  directory_fd = dirfd(directory);

  errno = 0;
  while ((entry = readdir(directory)) != NULL) {
    int candidate_fd;
    int duplicate_fd;

    if (!kvaf_parse_fd_name(entry->d_name, &candidate_fd) ||
        candidate_fd == directory_fd || candidate_fd == adopted_fd) {
      errno = 0;
      continue;
    }
    /* Own the file description first; validate only the stable duplicate. */
    duplicate_fd = fcntl(
        candidate_fd, F_DUPFD_CLOEXEC, KVAF_UMMU_DUP_MIN_FD);
    if (duplicate_fd < 0 ||
        !kvaf_fd_matches_device(duplicate_fd, &device_status)) {
      if (duplicate_fd >= 0) {
        (void)close(duplicate_fd);
      }
      errno = 0;
      continue;
    }
    if (adopted_fd >= 0) {
      /* Multiple file descriptions are ambiguous without a driver owner ID. */
      (void)close(duplicate_fd);
      (void)close(adopted_fd);
      adopted_fd = -1;
      result = -ENOTUNIQ;
      break;
    }
    adopted_fd = duplicate_fd;
    errno = 0;
  }
  scan_error = entry == NULL ? errno : 0;
  if (scan_error != 0) {
    if (adopted_fd >= 0) {
      (void)close(adopted_fd);
      adopted_fd = -1;
    }
    result = -scan_error;
  } else if (adopted_fd >= 0) {
    result = adopted_fd;
  }
  (void)closedir(directory);
  if (result >= 0) {
    errno = 0;
  }
  return result;
}
