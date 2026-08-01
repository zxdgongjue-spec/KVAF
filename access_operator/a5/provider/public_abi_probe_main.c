#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

uint64_t kvaf_public_abi_consumer_signature(void);
uint64_t kvaf_public_abi_udma_signature(void);

int main(void)
{
    const uint64_t consumer = kvaf_public_abi_consumer_signature();
    const uint64_t udma = kvaf_public_abi_udma_signature();

    if (consumer != udma || consumer == 0) {
        (void)fprintf(stderr,
            "KVAF_URMA_PUBLIC_ABI=MISMATCH:consumer=%016" PRIx64
            ":udma=%016" PRIx64 "\n",
            consumer, udma);
        return 1;
    }
    (void)printf("KVAF_URMA_PUBLIC_ABI=MATCH:%016" PRIx64 "\n",
        consumer);
    return 0;
}
