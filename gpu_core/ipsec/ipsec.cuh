#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#define GPU_IPSEC

/*ipsec的通道的最大数量*/
extern int tunnelsNb;

#ifdef GPU_SCHEDULE

/*在调用espEncap和gpuIpsec前必须运行一次此函数*/
extern void initIpsec(uint32_t srcAddr = SRC_ADDR);

extern __device__ void gpuIpsec(uint32_t packetsPerTh, uint8_t *in, uint32_t *out);

/*在结束时运行一次此函数*/
extern void freeIpsec();

#endif

/*
ipsec espencap
Input packet: (pkt_in)
+----------+---------------+---------+
| Ethernet | IP(proto=UDP) | payload |
+----------+---------------+---------+
^ethh      ^iph
Output packet: (pkt_out)
+----------+---------------+--------+----+------------+---------+-------+---------------------+
| Ethernet | IP(proto=ESP) |  ESP   | IP |  payload   | padding | extra | HMAC-SHA1 signature |
+----------+---------------+--------+----+------------+---------+-------+---------------------+
     14            20          24     20                pad_len     2    SHA_DIGEST_LENGTH = 20
^ethh      ^iph            ^esph    ^encapped_iph     ^esp_trail
*/