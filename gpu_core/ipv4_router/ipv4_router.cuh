#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <unordered_map>
#include <stdint.h>

#define HASH_TABLES_LENGTH 33
#define TBL24_SIZE ((1 << 24) + 1) //2^24
#define TBLLONG_SIZE ((1 << 24) + 1)

typedef std::unordered_map<uint32_t, uint16_t> route_hash_t;

#ifdef GPU_SCHEDULE

/*在调用gpuIpv4Router前必须运行一次此函数*/
extern void initIpv4Router(const char *routingInfo);

extern __device__ void gpuIpv4Router(uint32_t packetsPerTh, uint8_t *in, uint32_t *out);

/*在结束时运行一次此函数*/
extern void freeIpv4Router();

#endif