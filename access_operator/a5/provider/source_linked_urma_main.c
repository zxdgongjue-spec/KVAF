#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef KVAF_URMA_MAIN_SOURCE
#error "KVAF_URMA_MAIN_SOURCE must name the pinned upstream urma_main.c"
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * liburma's upstream urma_init() scans a directory next to liburma.so and
 * dlopens every liburma*.so provider it finds.  That is incompatible with
 * KVAF's checked source-owned UDMA/UMMU teardown: merely registering our
 * provider first would still run the system UDMA provider's init/uninit.
 *
 * Compile the pinned core into each KVAF provider and make its provider
 * directory permanently unavailable in this translation unit.  Device
 * discovery lives in urma_device.c and is therefore unaffected.  SO_LINKED
 * tells the pinned core that having no dynamically loaded provider is valid;
 * source_matched_udma_main.c registers the sole provider before urma_init().
 */
#include <dirent.h>
#include <errno.h>

static DIR *kvaf_urma_dynamic_provider_scan_forbidden(const char *path)
{
    (void)path;
    errno = ENOENT;
    return NULL;
}

int kvaf_urma_core_has_dynamic_provider_scan_disabled(void)
{
    return 1;
}

#define SO_LINKED 1
#define opendir(path) kvaf_urma_dynamic_provider_scan_forbidden(path)
#include KVAF_URMA_MAIN_SOURCE
#undef opendir
#undef SO_LINKED
