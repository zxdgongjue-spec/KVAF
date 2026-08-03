#ifndef KVAF_SOURCE_MATCHED_UMMU_FD_H
#define KVAF_SOURCE_MATCHED_UMMU_FD_H

#ifdef __cplusplus
extern "C" {
#endif

enum kvaf_source_ummu_fd_mode {
  KVAF_SOURCE_UMMU_FD_NONE = 0,
  KVAF_SOURCE_UMMU_FD_PRIVATE_OPEN = 1,
  KVAF_SOURCE_UMMU_FD_PROCESS_DUP = 2
};

/*
 * Duplicate the unique O_RDWR descriptor for device_path found in fd_dir.
 * The returned descriptor is a new CLOEXEC reference owned by the caller.
 * A negative errno value is returned on failure.
 *
 * This helper deliberately duplicates a candidate before validating it.  A
 * concurrent close/reuse of the numeric descriptor therefore cannot make
 * validation apply to one file description and ownership to another.
 */
int kvaf_source_ummu_duplicate_unique_process_fd(
    const char *device_path, const char *fd_dir);

#ifdef __cplusplus
}
#endif

#endif
