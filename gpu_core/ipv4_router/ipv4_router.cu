#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>

#include "ipv4_router.cuh"
#include "../header/net_struct.hpp"
#include "../header/gpu_config.hpp"

route_hash_t hashTables[HASH_TABLES_LENGTH];

static uint16_t *u16Tbl24;
static uint16_t *u16TblLong;

static uint16_t *_u16Tbl24;
static uint16_t *_u16TblLong;

static __device__ uint16_t *devU16Tbl24;
static __device__ uint16_t *devU16TblLong;

/*此函数用于向tables中添加下一跳地址*/
static int addRoute(route_hash_t *tables, uint32_t addr, uint16_t len, uint16_t nexthop)
{
    tables[len][addr] = nexthop;
    return 0;
}

/*此函数用于从routing_info.txt生成tables，其中下一跳的地址带有随机性*/
static int loadRibFromFile(route_hash_t *tables, const char *routingInfo)
{
    printf("[ipv4 router] 正在打开文件%s\n", routingInfo);
    FILE *fp = fopen(routingInfo, "r");
    char buf[256];

    if (fp == NULL)
    {
        printf("[ipv4 router] 打开文件%s出错\n", routingInfo);
    }
    assert(fp != NULL);

    while (fgets(buf, 256, fp))
    {
        char *strAddr = strtok(buf, "/");
        char *strLen = strtok(NULL, "\n");
        assert(strLen != NULL);

        uint32_t addr = ntohl(inet_addr(strAddr));
        uint16_t len = atoi(strLen);

        addRoute(tables, addr, len, rand() % 65532 + 1);
    }
    fclose(fp);
    return 0;
}

/*DIR-24-8-BASIC将IPv4地址空间分为长度分别为24 和8的两部分(TBL24和TBLlong)
此函数用于从tables生成TBL24表和TBLlong表*/
static int buildDirectFib(const route_hash_t *tables, uint16_t *TBL24, uint16_t *TBLlong)
{
    memset(TBL24, 0, TBL24_SIZE * sizeof(uint16_t));
    memset(TBLlong, 0, TBLLONG_SIZE * sizeof(uint16_t));
    unsigned int current_TBLlong = 0;

    for (unsigned i = 0; i <= 24; i++)
    {
        for (auto it = tables[i].begin(); it != tables[i].end(); it++)
        {
            uint32_t addr = (*it).first;
            uint16_t dest = (uint16_t)(0xffffu & (uint64_t)(*it).second);
            uint32_t start = addr >> 8;
            uint32_t end = start + (0x1u << (24 - i));
            for (unsigned k = start; k < end; k++)
            {
                TBL24[k] = dest;
            }
        }
    }

    for (unsigned i = 25; i <= 32; i++)
    {
        for (auto it = tables[i].begin(); it != tables[i].end(); it++)
        {
            uint32_t addr = (*it).first;
            uint16_t dest = (uint16_t)(0x0000ffff & (uint64_t)(*it).second);
            uint16_t dest24 = TBL24[addr >> 8];
            if (((uint16_t)dest24 & 0x8000u) == 0)
            {
                uint32_t start = current_TBLlong + (addr & 0x000000ff);
                uint32_t end = start + (0x00000001u << (32 - i));

                for (unsigned j = current_TBLlong; j <= current_TBLlong + 256; j++)
                {
                    if (j < start || j >= end)
                    {
                        TBLlong[j] = dest24;
                    }
                    else
                    {
                        TBLlong[j] = dest;
                    }
                }
                TBL24[addr >> 8] = (uint16_t)(current_TBLlong >> 8) | 0x8000u;
                current_TBLlong += 256;
                assert(current_TBLlong <= TBLLONG_SIZE);
            }
            else
            {
                uint32_t start = ((uint32_t)dest24 & 0x7fffu) * 256 + (addr & 0x000000ff);
                uint32_t end = start + (0x00000001u << (32 - i));

                for (unsigned j = start; j < end; j++)
                {
                    TBLlong[j] = dest;
                }
            }
        }
    }
    return 0;
}

void saveTableToFile()
{
    FILE *fp = fopen("ipv4_router/TBL24.txt", "w");
    printf("正在生成TBL24.txt，这可能需要很长的时间...\n");
    for (int i = 0; i < TBL24_SIZE; i++)
    {
        fprintf(fp, "%d\n", u16Tbl24[i]);
    }
    fclose(fp);
    fp = fopen("ipv4_router/TBLlong.txt", "w");
    printf("正在生成TBLlong.txt，这可能需要很长的时间...\n");
    for (int i = 0; i < TBLLONG_SIZE; i++)
    {
        fprintf(fp, "%d\n", u16TblLong[i]);
    }
    fclose(fp);
}

void initIpv4Router(const char *routingInfo)
{
    loadRibFromFile(hashTables, routingInfo);
    int sizeTbl24 = sizeof(uint16_t) * TBL24_SIZE;
    int sizeTblLong = sizeof(uint16_t) * TBLLONG_SIZE;
    void *ptr1 = malloc(sizeof(char) * sizeTbl24);
    void *ptr2 = malloc(sizeof(char) * sizeTblLong);
    assert(ptr1 != NULL);
    assert(ptr2 != NULL);
    memset(ptr1, 0xcd, sizeTbl24);
    memset(ptr2, 0xcd, sizeTblLong);
    u16Tbl24 = (uint16_t *)ptr1;
    u16TblLong = (uint16_t *)ptr2;
    buildDirectFib(hashTables, u16Tbl24, u16TblLong);
    printf("[ipv4 router] 已经成功生成TBL24和TBLLong\n");
    // saveTableToFile();
    cudaMalloc((void **)&_u16Tbl24, sizeof(uint16_t) * TBL24_SIZE);
    cudaMalloc((void **)&_u16TblLong, sizeof(uint16_t) * TBLLONG_SIZE);
    cudaMemcpy(_u16Tbl24, u16Tbl24, sizeof(uint16_t) * TBL24_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(_u16TblLong, u16TblLong, sizeof(uint16_t) * TBLLONG_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(devU16Tbl24, &_u16Tbl24, sizeof(_u16Tbl24));
    cudaMemcpyToSymbol(devU16TblLong, &_u16TblLong, sizeof(_u16TblLong));
    free(u16Tbl24);
    free(u16TblLong);
}

void freeIpv4Router()
{
    cudaFree(_u16Tbl24);
    cudaFree(_u16TblLong);
}

// static __device__ uint16_t ntohsGpu(uint16_t data)
// {
//     uint16_t result;
//     uint16_t dataH = data / pow(16, 2);
//     uint16_t dataL = data - dataH * pow(16, 2);
//     result = dataL * pow(16, 2) + dataH;
//     return result;
// }

// static __device__ uint16_t htonsGpu(uint16_t data)
// {
//     uint16_t result;
//     uint16_t dataH = data / pow(16, 2);
//     uint16_t dataL = data - dataH * pow(16, 2);
//     result = dataL * pow(16, 2) + dataH;
//     return result;
// }

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

static __device__ __inline__ uint16_t u8ToU16(uint8_t *data)
{
    /*网络字节顺序为__BIG_ENDIAN 地址的低位存储值的高位*/
    uint16_t result = 0;
    for (int i = 0; i < 2; i++)
    {
        result += data[i] * pow(16, 2 - 2 * i);
    }
    return result;
}

static __device__ __inline__ void u16ToU8(uint16_t data, uint8_t *result)
{
    result[0] = data / pow(16, 2);
    result[1] = data - result[0] * pow(16, 2);
}

static __device__ __inline__ uint32_t u8ToU32(uint8_t *data)
{
    /*网络字节顺序为__BIG_ENDIAN 地址的低位存储值的高位*/
    uint32_t result = 0;
    for (int i = 0; i < 4; i++)
    {
        result += data[i] * pow(16, 6 - 2 * i);
    }
    return result;
}

__device__ void gpuIpv4Router(uint32_t packetsPerTh, uint8_t *in, uint32_t *out)
{
    for (uint32_t i = 0; i < packetsPerTh; i++)
    {
        struct iphdr *iph = (struct iphdr *)((struct ether_header *)in + 1);

        /*对于uint16_t在这里不要使用iph->tot_len方式，这样会出现CUDA_EXCEPTION_6 Warp Misaligned Address
        另外，如果使用u8ToU16则不需要转换字节顺序*/
        uint16_t packetLen = sizeof(struct ether_header) + u8ToU16((uint8_t *)iph + 2 * sizeof(uint8_t));
        assert(packetLen != sizeof(struct ether_header));
        // uint16_t packetLen = sizeof(struct ether_header) + ntohsGpu(iph->tot_len);
        // printf("%u\n",packetLen);

        /*对于uint32_t在这里不要使用iph->daddr方式，这样会出现CUDA_EXCEPTION_6 Warp Misaligned Address
        另外，如果使用u8ToU32则不需要转换字节顺序*/
        uint32_t dstAddr = u8ToU32((uint8_t *)(iph + 1) - sizeof(uint32_t));
        // uint32_t dstAddr = ntohlGpu(iph->daddr);
        // printf("%u\n", dstAddr);

        if (out != NULL)
        {
            out[i] = 0xffff;
        }
        /*不正常，16进制全为1忽略该IP地址*/
        if (dstAddr == IGNORED_IP)
        {
            if (out != NULL)
            {
                out[i] = 0;
            }
        }
        /*正常，开始查找*/
        else
        {
            /*返回在TBL24和TBLlong中的查找结果*/
            uint16_t tempDest = devU16Tbl24[dstAddr >> 8];
            if (tempDest & 0x8000u)
            {
                uint32_t index = (((uint32_t)(tempDest & 0x7fff)) << 8) + (dstAddr & 0xff);
                tempDest = devU16TblLong[index];
            }
            if (out != NULL)
            {
                out[i] = tempDest;
            }
        }

        uint32_t checkSum;
        if (iph->ttl > 1)
        {
            iph->ttl--;
            uint8_t *pCheck = (uint8_t *)(iph + 1) - 2 * sizeof(uint32_t) - sizeof(uint16_t);
            uint16_t check = u8ToU16(pCheck);
            // printf("%u\n", check);
            /*
            首先需要判断修改的字节是奇数字节还是偶数字节
            若为奇数字节
            则变化值为修改绝对值
            若为偶数字节
            则变化值为修改绝对值乘以2^8
            0xFFFF-2^8=0xFEFF
            */
            checkSum = (~check & 0xFFFF) + 0xFEFF;
            uint16_t newCheck = ~(checkSum + (checkSum >> 16));
            u16ToU8(newCheck, pCheck);
            // printf("%u\n", u8ToU16(pCheck));
        }

        in += packetLen;
    }
}