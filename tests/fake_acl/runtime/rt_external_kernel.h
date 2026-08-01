#ifndef KVAF_TEST_FAKE_RT_EXTERNAL_KERNEL_H
#define KVAF_TEST_FAKE_RT_EXTERNAL_KERNEL_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef char char_t;
typedef int rtError_t;
typedef void* rtStream_t;
typedef struct rtSmDesc rtSmDesc_t;
typedef struct rtHostInputInfo rtHostInputInfo_t;

typedef struct tagRtAicpuArgsEx {
  void* args;
  rtHostInputInfo_t* hostInputInfoPtr;
  rtHostInputInfo_t* kernelOffsetInfoPtr;
  uint32_t argsSize;
  uint16_t hostInputInfoNum;
  uint16_t kernelOffsetInfoNum;
  uint32_t soNameAddrOffset;
  uint32_t kernelNameAddrOffset;
  bool isNoNeedH2DCopy;
  uint16_t timeout;
  uint8_t reserved;
} rtAicpuArgsEx_t;

rtError_t rtAicpuKernelLaunchExWithArgs(
    uint32_t kernel_type, const char_t* op_name, uint32_t num_blocks,
    const rtAicpuArgsEx_t* args_info, rtSmDesc_t* sm_desc,
    rtStream_t stream, uint32_t flags);

#ifdef __cplusplus
}
#endif

#endif
