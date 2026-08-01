#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef KVAF_URMA_MAIN_SOURCE
#error "KVAF_URMA_MAIN_SOURCE must name the pinned upstream urma_main.c"
#endif

/*
 * liburma's upstream urma_init() scans a directory next to liburma.so and
 * dlopens every liburma*.so provider it finds.  That is incompatible with
 * KVAF's checked source-owned UDMA/UMMU teardown: merely registering our
 * provider first would still run the system UDMA provider's init/uninit.
 *
 * Compile the pinned core into each KVAF provider and make every directory
 * except the public /dev/uburma enumeration path unavailable in this
 * translation unit.  This blocks the private provider-SO directory without
 * breaking urma_get_jfce_cnt_info().  SO_LINKED tells the pinned core that
 * having no dynamically loaded provider is valid; source_matched_udma_main.c
 * registers the sole provider before urma_init().
 */
#include <dirent.h>
#include <errno.h>
#include <stddef.h>
#include <string.h>

static DIR *kvaf_urma_restricted_opendir(const char *path)
{
    if (path != NULL && strcmp(path, "/dev/uburma") == 0) {
        /* This call is compiled before the opendir macro below is defined. */
        return opendir(path);
    }

    errno = ENOENT;
    return NULL;
}

int kvaf_urma_core_has_dynamic_provider_scan_disabled(void)
{
    return 1;
}

#define SO_LINKED 1
#define opendir(path) kvaf_urma_restricted_opendir(path)
#include KVAF_URMA_MAIN_SOURCE
#undef opendir
#undef SO_LINKED
