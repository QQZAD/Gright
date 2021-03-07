#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <time.h>

#include "nat.cuh"
#include "../header/net_struct.hpp"
#include "../header/gpu_config.hpp"

static uint16_t port = 10000;
static __device__ uint16_t devPort;
static struct natEntry natTable[MAX_NAT_ENTRIES];
static struct natEntry *_natTable;
static __device__ struct natEntry *devNatTable;

void initNat()
{
    printf("[nat] 正在初始化NAT表项\n");
    for (int i = 0; i < MAX_NAT_ENTRIES; i++)
    {
        natTable[i].valid = SET_ENTRY;
        natTable[i].lanIpaddr = LAN_NET_IP + i;
        natTable[i].lanPort = rand() % MAX_NAT_ENTRIES;
    }
    cudaMalloc((void **)&_natTable, sizeof(struct natEntry) * MAX_NAT_ENTRIES);
    cudaMemcpy(_natTable, natTable, sizeof(struct natEntry) * MAX_NAT_ENTRIES, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(devNatTable, &_natTable, sizeof(_natTable));
    cudaMemcpyToSymbol(devPort, &port, sizeof(port));
    printf("[nat] 已经成功初始化%d条NAT表项\n", MAX_NAT_ENTRIES);
}

void freeNat()
{
    cudaFree(_natTable);
}

// static __device__ uint16_t ntohsGpu(uint16_t data)
// {
//     uint16_t result;
//     uint16_t dataH = data / pow(16, 2);
//     uint16_t dataL = data - dataH * pow(16, 2);
//     result = dataL * pow(16, 2) + dataH;
//     return result;
// }

static __device__ uint16_t htonsGpu(uint16_t data)
{
    uint16_t result;
    uint16_t dataH = data / pow(16, 2);
    uint16_t dataL = data - dataH * pow(16, 2);
    result = dataL * pow(16, 2) + dataH;
    return result;
}

// static __device__ uint32_t ntohlGpu(uint32_t data)
// {
//     uint32_t result = 0;
//     uint32_t _data[4] = {0};
//     uint32_t old_data, new_data;
//     old_data = data;
//     for (int i = 0; i < 4; i++)
//     {
//         if (i == 0)
//         {
//             new_data = old_data;
//         }
//         else
//         {
//             new_data = old_data - _data[i - 1] * pow(16, 8 - 2 * i);
//         }
//         _data[i] = new_data / pow(16, 6 - 2 * i);
//         result += _data[i] * pow(16, 2 * i);
//         old_data = new_data;
//     }
//     return result;
// }

// static __device__ uint32_t htonlGpu(uint32_t data)
// {
//     uint32_t result = 0;
//     uint32_t _data[4] = {0};
//     uint32_t old_data, new_data;
//     old_data = data;
//     for (int i = 0; i < 4; i++)
//     {
//         if (i == 0)
//         {
//             new_data = old_data;
//         }
//         else
//         {
//             new_data = old_data - _data[i - 1] * pow(16, 8 - 2 * i);
//         }
//         _data[i] = new_data / pow(16, 6 - 2 * i);
//         result += _data[i] * pow(16, 2 * i);
//         old_data = new_data;
//     }
//     return result;
// }

static __device__ __inline__ uint16_t u8ToU16(uint8_t *data, uint8_t *temp = NULL)
{
    /*网络字节顺序为__BIG_ENDIAN 地址的低位存储值的高位*/
    uint16_t result = 0;
    for (int i = 0; i < 2; i++)
    {
        if (temp != NULL)
        {
            temp[i] = data[i];
            result += temp[i] * pow(16, 2 - 2 * i);
        }
        else
        {
            result += data[i] * pow(16, 2 - 2 * i);
        }
    }
    return result;
}

static __device__ __inline__ void u16ToU8(uint16_t data, uint8_t *result)
{
    /*网络字节顺序为__BIG_ENDIAN 地址的低位存储值的高位*/
    result[0] = data / pow(16, 2);
    result[1] = data - result[0] * pow(16, 2);
}

static __device__ __inline__ uint32_t u8ToU32(uint8_t *data, uint8_t *temp = NULL)
{
    /*网络字节顺序为__BIG_ENDIAN 地址的低位存储值的高位*/
    uint32_t result = 0;
    for (int i = 0; i < 4; i++)
    {
        if (temp != NULL)
        {
            temp[i] = data[i];
            result += temp[i] * pow(16, 6 - 2 * i);
        }
        else
        {
            result += data[i] * pow(16, 6 - 2 * i);
        }
    }
    return result;
}

static __device__ __inline__ void u32ToU8(uint32_t data, uint8_t *result)
{
    /*网络字节顺序为__BIG_ENDIAN 地址的低位存储值的高位*/
    result[0] = data / pow(16, 6);
    uint32_t temp1 = data - result[0] * pow(16, 6);
    result[1] = temp1 / pow(16, 4);
    uint32_t temp2 = temp1 - result[1] * pow(16, 4);
    result[2] = temp2 / pow(16, 2);
    result[3] = temp2 - result[2] * pow(16, 2);
}

/*寻找NAT表项，返回对应的序号*/
__device__ uint16_t findNatEntry(uint32_t saddr, uint16_t sport)
{
    for (int i = 0; i < MAX_NAT_ENTRIES; i++)
    {
        if ((devNatTable[i].lanIpaddr == saddr) && (devNatTable[i].lanPort == sport) && devNatTable[i].valid)
        {
            devNatTable[i].valid = 0;
            return i;
        }
    }
    return 0;
}

__device__ void gpuNat(uint32_t packetsPerTh, uint8_t *in, uint32_t *out)
{
    /*
    Cuda API error detected: cudaLaunchKernel returned (0x7)
    这表明没有启动，因为它没有适当的资源，这说明线程或寄存器使用过多
    解决方案：
    cd /home/adzhu/source/Gright/scripts;sudo chmod +x *.sh;./device_query.sh
    一个线程块中可用寄存器的最多数量为65536
    一个线程块中的线程数量为THREADS_PER_BLOCK=928
    nvcc --maxrregcount 60
    */
    int incremental = 0;
    uint32_t checkSum = 0;

    for (uint32_t i = 0; i < packetsPerTh; i++)
    {
        struct iphdr *iph = (struct iphdr *)((struct ether_header *)in + 1);
        uint8_t *pIpCheck = (uint8_t *)(iph + 1) - 2 * sizeof(uint32_t) - sizeof(uint16_t);
        uint16_t ipCheck = u8ToU16(pIpCheck);

        uint16_t packetLen = sizeof(struct ether_header) + u8ToU16((uint8_t *)iph + 2 * sizeof(uint8_t));
        assert(packetLen != sizeof(struct ether_header));

        uint8_t protocol = ((uint8_t *)(iph + 1) - 2 * sizeof(uint32_t) - sizeof(uint16_t) - sizeof(uint8_t))[0];
        assert(protocol == 0x06);

        struct tcphdr *tcph = (struct tcphdr *)(iph + 1);
        uint8_t *pTcpCheck = (uint8_t *)(tcph + 1) - 2 * sizeof(uint16_t);
        uint16_t tcpCheck = u8ToU16(pTcpCheck);

#ifdef WAN_TO_LAN
        /*WAN->LAN 源IP为主机IP 目的IP和目的端口:公有->私有*/
        uint8_t *pDstAddr = (uint8_t *)(iph + 1) - sizeof(uint32_t);
        uint8_t pDstAddrOld[4];
        uint32_t dstAddr = u8ToU32(pDstAddr, pDstAddrOld);
        uint8_t *pDstPort = (uint8_t *)tcph + sizeof(uint16_t);
        uint8_t pDstPortOld[2];
        uint16_t dstPort = u8ToU16(pDstPort, pDstPortOld);
        assert(dstPort < MAX_NAT_ENTRIES);

        if (dstAddr == HOST_IP_ADDR && devNatTable[dstPort].valid == SET_ENTRY)
        {
            /*从值低位到值高位*/
            uint8_t _pDstAddr[4];
            u32ToU8(devNatTable[dstPort].lanIpaddr, _pDstAddr);
            incremental = 0;
            incremental += (_pDstAddr[3] - pDstAddrOld[3]) * 0x100;
            incremental += (_pDstAddr[2] - pDstAddrOld[2]);
            incremental += (_pDstAddr[1] - pDstAddrOld[1]) * 0x100;
            incremental += (_pDstAddr[0] - pDstAddrOld[0]);
            /*目的IP作为返回值*/
            u32ToU8(devNatTable[dstPort].lanIpaddr, pDstAddr);

            /*重新计算IP头的校验值*/
            if (iph->ttl > 1)
            {
                iph->ttl--;
                incremental -= 0x100;
            }
            checkSum = (~ipCheck & 0xFFFF) + 0xFFFF + incremental;
            /*修改IP头的校验值*/
            u16ToU8(~(checkSum + (checkSum >> 16)), pIpCheck);

            /*从值低位到值高位*/
            incremental = 0;
            incremental += ((_pDstAddr[3] << 8) & 0xFF00) + (_pDstAddr[2] & 0xFF);
            incremental -= ((pDstAddrOld[3] << 8) & 0xFF00) + (pDstAddrOld[2] & 0xFF);
            incremental += ((_pDstAddr[1] << 8) & 0xFF00) + (_pDstAddr[0] & 0xFF);
            incremental -= ((pDstAddrOld[1] << 8) & 0xFF00) + (pDstAddrOld[0] & 0xFF);
            uint8_t _pDstPort[2];
            u16ToU8(devNatTable[dstPort].lanPort, _pDstPort);
            incremental += ((_pDstPort[1] << 8) & 0xFF00) + (_pDstPort[0] & 0xFF);
            incremental -= ((pDstPortOld[1] << 8) & 0xFF00) + (pDstPortOld[0] & 0xFF);
            /*目的端口作为返回值*/
            u16ToU8(devNatTable[dstPort].lanPort, pDstPort);

            checkSum = (~tcpCheck & 0xFFFF) + 0xFFFF + incremental;
            /*修改TCP头的校验值*/
            u16ToU8(~(checkSum + (checkSum >> 16)), pTcpCheck);

            devNatTable[dstPort].valid = 0;
            out = (uint32_t *)in;
        }
#endif

#ifdef LAN_TO_WAN
        /*LAN->WAN 源IP和源端口:私有->公有 目的IP位于LAN网段内*/
        uint8_t *pSrcAddr = (uint8_t *)(iph + 1) - 2 * sizeof(uint32_t);
        uint8_t pSrcAddrOld[4];
        uint32_t srcAddr = u8ToU32(pSrcAddr, pSrcAddrOld);

        if ((srcAddr & LAN_NET_MASK) == LAN_NET_IP)
        {
            uint8_t *pSrcPort = (uint8_t *)tcph;
            uint8_t pSrcPortOld[2];
            uint32_t srcPort = u8ToU16(pSrcPort, pSrcPortOld);
            assert(srcPort < MAX_NAT_ENTRIES);

            uint32_t newPort = findNatEntry(srcAddr, srcPort);
            // uint32_t newPort = 0;
            if (!newPort)
            {
                /*添加一个端口大于10000的NAT表项*/
                newPort = htonsGpu(devPort++);
                if (devPort == 0)
                    devPort = 10000;
                devNatTable[newPort].valid = SET_ENTRY;
                devNatTable[newPort].lanIpaddr = srcAddr;
                devNatTable[newPort].lanPort = srcPort;
            }

            /*从值低位到值高位*/
            uint8_t _pSrcAddr[4];
            u32ToU8(HOST_IP_ADDR, _pSrcAddr);
            incremental = 0;
            incremental += (_pSrcAddr[3] - pSrcAddrOld[3]) * 0x100;
            incremental += (_pSrcAddr[2] - pSrcAddrOld[2]);
            incremental += (_pSrcAddr[1] - pSrcAddrOld[1]) * 0x100;
            incremental += (_pSrcAddr[0] - pSrcAddrOld[0]);
            /*源IP作为返回值*/
            u32ToU8(HOST_IP_ADDR, pSrcAddr);

            if (iph->ttl > 1)
            {
                iph->ttl--;
                incremental -= 0x100;
            }
            checkSum = (~ipCheck & 0xFFFF) + 0xFFFF + incremental;
            /*修改IP头的校验值*/
            u16ToU8(~(checkSum + (checkSum >> 16)), pIpCheck);

            /*从值低位到值高位*/
            incremental = 0;
            incremental += ((_pSrcAddr[3] << 8) & 0xFF00) + (_pSrcAddr[2] & 0xFF);
            incremental -= ((pSrcAddrOld[3] << 8) & 0xFF00) + (pSrcAddrOld[2] & 0xFF);
            incremental += ((_pSrcAddr[1] << 8) & 0xFF00) + (_pSrcAddr[0] & 0xFF);
            incremental -= ((pSrcAddrOld[1] << 8) & 0xFF00) + (pSrcAddrOld[0] & 0xFF);
            uint8_t _pSrcPort[2];
            u16ToU8(newPort, _pSrcPort);
            incremental += ((_pSrcPort[1] << 8) & 0xFF00) + (_pSrcPort[0] & 0xFF);
            incremental -= ((pSrcPortOld[1] << 8) & 0xFF00) + (pSrcPortOld[0] & 0xFF);
            /*源端口作为返回值*/
            u16ToU8(newPort, pSrcPort);

            checkSum = (~tcpCheck & 0xFFFF) + 0xFFFF + incremental;
            /*修改TCP头的校验值*/
            u16ToU8(~(checkSum + (checkSum >> 16)), pTcpCheck);
        }
#endif

        in += packetLen;
    }
}