#pragma once
#include "kvaf_aiv_batch_get.h"
#include "batch_get_raw_addr_abi.h"
extern "C" {
int32_t KvafExecuteBatchGetRawAddrLoadKernel(const char* path);
int32_t KvafExecuteBatchGetRawAddrLaunch(const KvafExecuteBatchGetRawAddrKernelArgs*, aclrtStream);
// Checked retirement for direct reads. Call only after all DMA has completed.
// Failed calls retain handles for retry; Discard also unregisters local HBM.
int32_t KvafBatchGetLocalBufferDiscard(KvafBatchGetLocalBuffer**);
int32_t KvafBatchGetTransportClose(KvafBatchGetTransport**);
// Startup-probe recovery only: the AIV kernel must have finished. Destroy
// every initiator Jetty before allowing its outstanding READ buffers to retire.
int32_t KvafBatchGetTransportQuiesce(KvafBatchGetTransport*);
}
