#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint32_t (*kvaf_private_abi_query_t)(void);

int main(int argc, char **argv)
{
    kvaf_private_abi_query_t query;
    unsigned long expected;
    char *end = NULL;
    void *library;
    uint32_t actual;

    if (argc != 3) {
        (void)fprintf(stderr,
            "usage: %s PROVIDER_SO EXPECTED_PRIVATE_ABI\n", argv[0]);
        return 2;
    }
    errno = 0;
    expected = strtoul(argv[2], &end, 10);
    if (errno != 0 || end == argv[2] || *end != '\0' ||
        expected == 0 || expected > UINT32_MAX) {
        (void)fprintf(stderr, "invalid expected private ABI: %s\n",
            argv[2]);
        return 2;
    }

    library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        (void)fprintf(stderr, "dlopen(%s) failed: %s\n",
            argv[1], dlerror());
        return 1;
    }
    dlerror();
    *(void **)(&query) =
        dlsym(library, "kvaf_urma_provider_private_abi_v2");
    if (query == NULL || dlerror() != NULL) {
        (void)fprintf(stderr,
            "provider private-ABI query is unavailable\n");
        (void)dlclose(library);
        return 1;
    }
    actual = query();
    if (actual != (uint32_t)expected) {
        (void)fprintf(stderr,
            "provider private ABI mismatch: expected=%lu actual=%u\n",
            expected, actual);
        (void)dlclose(library);
        return 1;
    }
    if (dlclose(library) != 0) {
        (void)fprintf(stderr, "dlclose(%s) failed: %s\n",
            argv[1], dlerror());
        return 1;
    }
    (void)printf("KVAF_URMA_PROVIDER_PRIVATE_ABI=%u\n", actual);
    return 0;
}
