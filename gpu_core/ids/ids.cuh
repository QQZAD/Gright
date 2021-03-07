#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#define RULE_STRING_MAXSIZE 512
#define DEV_ACM_PATTERN(pacm, pid) ((pacm)->dim1_patterns + (pid)*RULE_STRING_MAXSIZE)

#define DEV_RULE_NB 121

#ifdef GPU_SCHEDULE

/*在调用gpuIds前必须运行一次此函数*/
extern void initIds(const char *acRule);

extern __device__ void gpuIds(uint32_t packetsPerTh, uint8_t *in, uint32_t *out);

/*在结束时运行一次此函数*/
extern void freeIds();

#endif