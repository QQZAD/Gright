#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef GPU_IPSEC

/*在调用gpuAes前必须运行一次此函数*/
extern void initAes();

extern __device__ void gpuAes(uint8_t *in, uint32_t *out);

/*在结束时运行一次此函数*/
extern void freeAes();

#endif
/*
ipsec AES

Input packet: assumes ESP encaped, but payload not encrypted yet.
+----------+---------------+--------+----+------------+---------+-------+---------------------+
| Ethernet | IP(proto=ESP) |  ESP   | IP |  payload   | padding | extra | HMAC-SHA1 signature |
+----------+---------------+--------+----+------------+---------+-------+---------------------+
     14            20          24     20                pad_len     2    SHA_DIGEST_LENGTH = 20
^ethh      ^iph            ^esph    ^encrypt_ptr
                                    <===== to be encrypted with AES ====>
*/