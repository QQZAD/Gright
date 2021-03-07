#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdint.h>

// #define LAN_TO_WAN
#define WAN_TO_LAN
#define SET_ENTRY 133

/*NAT表项*/
struct natEntry
{
    uint32_t lanIpaddr;
    uint16_t lanPort;
    uint8_t valid;
};

#ifdef GPU_SCHEDULE

/*在调用gpuNat前必须运行一次此函数*/
extern void initNat();

extern __device__ void gpuNat(uint32_t packetsPerTh, uint8_t *in, uint32_t *out);

/*在结束时运行一次此函数*/
extern void freeNat();

#endif