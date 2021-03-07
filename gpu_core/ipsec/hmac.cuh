#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef GPU_IPSEC

/*在调用gpuAes前必须运行一次此函数*/
extern void initHmac();

extern __device__ void gpuHmac(uint8_t *in, uint32_t *out);

/*在结束时运行一次此函数*/
extern void freeHmac();

#endif
/*
ipsec HMAC

Input packet: assumes encaped
+----------+---------------+--------+----+------------+---------+-------+---------------------+
| Ethernet | IP(proto=ESP) |  ESP   | IP |  payload   | padding | extra | HMAC-SHA1 signature |
+----------+---------------+--------+----+------------+---------+-------+---------------------+
     14            20          24     20                pad_len     2    SHA_DIGEST_LENGTH = 20
^ethh      ^iph            ^esph    ^encaped_iph
                           ^payload_out
                           ^encapsulated
                           <===== authenticated part (payload_len) =====>
*/