#include <stdio.h>
#include <assert.h>
#include <unordered_map>
#include <random>
#include <functional>
#include <emmintrin.h>

#include "../packet/packet.hpp"
#include "ipsec.cuh"
#include "aes.cuh"
#include "hmac.cuh"
#include "../header/gpu_config.hpp"

int tunnelsNb = 0;

static std::function<uint64_t()> frand;

static std::unordered_map<struct ipaddr_pair, struct espencap_sa_entry *> saTable;

void initIpsec(uint32_t srcAddr = SRC_ADDR)
{
    printf("[ipsec] 正在根据源IP地址%s生成<源-目的IP地址对，ESP封装结构>\n", u32ToIp(srcAddr).c_str());
    tunnelsNb = DEST_ADDR_NB;
    frand = std::bind(std::uniform_int_distribution<uint64_t>{}, std::mt19937_64());
    assert(tunnelsNb != 0);
    for (int i = 0; i < tunnelsNb; i++)
    {
        /*Hash表中源IP地址是相同的，而目的IP地址是随机生成的*/
        struct ipaddr_pair pair;
        pair.src_addr = srcAddr;
        /*(frand() % 0xffffffu)*/
        pair.dest_addr = 0xc0a80000u | (i + 1);
        /*生成ESP封装结构*/
        struct espencap_sa_entry *entry = new struct espencap_sa_entry;
        entry->spi = frand() % 0xffffffffu;
        entry->rpl = frand() % 0xffffffffu;
        /*指定IP网关地址*/
        entry->gwaddr = SRC_ADDR;
        entry->entry_idx = i;
        /*插入<源-目的IP地址对，ESP封装结构>并构建Hash表*/
        auto result = saTable.insert(std::make_pair<ipaddr_pair &, espencap_sa_entry *&>(pair, entry));
        assert(result.second == true);
    }
    printf("[ipsec] 已经成功构建ipsec的Hash表\n");
    initAes();
    initHmac();
}

uint64_t espEncap(uint8_t *in, uint32_t nbPackets, uint8_t *&out)
{
    uint64_t totlLen = 0;
    uint8_t **pacBytes = (uint8_t **)malloc(sizeof(uint8_t *) * nbPackets);
    uint32_t *pacLen = (uint32_t *)malloc(sizeof(uint32_t) * nbPackets);
    uint8_t *pin = in;

    for (int i = 0; i < nbPackets; i++)
    {
        struct ether_header *ethh = (struct ether_header *)pin;
        if (ntohs(ethh->ether_type) != ETHERTYPE_IP)
        {
            /*当前数据包的3层协议不是ipv4*/
            printf("[error] i-%d-ethh->ether_type-0x%04x\n", i, ntohs(ethh->ether_type));
            return 1;
        }
        struct iphdr *iph = (struct iphdr *)(ethh + 1);
        uint16_t ipLen = ntohs(iph->tot_len);
        uint16_t packetLen = sizeof(struct ether_header) + ipLen;

        struct ipaddr_pair pair;
        pair.src_addr = ntohl(iph->saddr);
        pair.dest_addr = ntohl(iph->daddr);

        pair.src_addr = SRC_ADDR;
        pair.dest_addr = 0xc0a80000u | rand() % (DEST_ADDR_NB - 1 + 1) + 1;

        /*在<源-目的IP地址对，ESP封装结构>Hash表中查找输入的源-目的IP地址对并得到对应的ESP封装结构*/
        auto saItem = saTable.find(pair);
        struct espencap_sa_entry *saEntry = NULL;

        if (saItem != saTable.end())
        {
            saEntry = saItem->second;
            assert(saEntry->entry_idx < 1024u);
        }
        else
        {
            /*在Hash表中找不到当前数据包对应的ESP封装结构*/
            return 2;
        }

        int padLen = AES_BLOCK_SIZE - (ipLen + 2) % AES_BLOCK_SIZE;
        int encSize = ipLen + padLen + 2; // "extra"部分附加的两个字节
        int extendedIpLen = sizeof(struct iphdr) + encSize + sizeof(struct esphdr) + SHA_DIGEST_LENGTH;
        int lengthToExtend = extendedIpLen - ipLen;

        pacLen[i] = packetLen + lengthToExtend;
        pacBytes[i] = (uint8_t *)malloc(sizeof(uint8_t) * pacLen[i]);
        memcpy(pacBytes[i], pin, packetLen);

        ethh = (struct ether_header *)(pacBytes[i]);
        iph = (struct iphdr *)(ethh + 1);

        assert(0 == (encSize % AES_BLOCK_SIZE));
        struct esphdr *esph = (struct esphdr *)(iph + 1);
        uint8_t *encappedIph = (uint8_t *)esph + sizeof(struct esphdr);
        uint8_t *espTrail = encappedIph + ipLen;

        /*将IP头和负载后移*/
        memmove(encappedIph, iph, ipLen);
        /*清空padding*/
        memset(espTrail, 0, padLen);
        /*将padLen储存在倒数第二个*/
        espTrail[padLen] = (uint8_t)padLen;
        /*将IP-in-IP协议ID储存在最后一个*/
        espTrail[padLen + 1] = 0x04;

        uint8_t *shaDigestHead = espTrail + padLen + 2;
        memset(shaDigestHead, 0, SHA_DIGEST_LENGTH);

        /*填充ESP header*/
        esph->esp_spi = saEntry->spi;
        esph->esp_rpl = saEntry->rpl;
        /*随机生成iv*/
        uint64_t ivFirstHalf = rand();
        uint64_t ivSecondHalf = rand();
        __m128i newIv = _mm_set_epi64((__m64)ivFirstHalf, (__m64)ivSecondHalf);
        _mm_storeu_si128((__m128i *)esph->esp_iv, newIv);
        /*设置标准IP头大小*/
        iph->ihl = (20 >> 2);
        iph->tot_len = htons(extendedIpLen);
        /*使用ESP安全协议*/
        iph->protocol = 0x32;
        iph->check = htons(0);
        iph->check = htons(getIpCheckSum((uint8_t *)iph, iph->ihl));

        pin += packetLen;
        totlLen += pacLen[i];
    }
    // free(in);
    out = (uint8_t *)malloc(sizeof(uint8_t) * totlLen);
    // printf("totlLen-%d\n", totlLen);

    uint8_t *pout = out;
    for (int i = 0; i < nbPackets; i++)
    {
        // printf("pacLen-%d\n", pacLen[i]);
        memcpy(pout, pacBytes[i], pacLen[i]);
        pout += pacLen[i];
        free(pacBytes[i]);
    }
    free(pacLen);
    return totlLen;
}

void freeIpsec()
{
    freeAes();
    freeHmac();
}

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

__device__ void gpuIpsec(uint32_t packetsPerTh, uint8_t *in, uint32_t *out)
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
        // #ifdef ASSERT_PACKETS
        //         uint16_t etherType = u8ToU16((uint8_t *)iph - sizeof(uint16_t));
        //         if (etherType != ETHERTYPE_IP)
        //         {
        //             printf("[ipsec error] etherType-0x%x\n", etherType);
        //         }
        //         assert(etherType == ETHERTYPE_IP);
        // #endif

        if (out != NULL)
        {
            gpuAes(in, out + i);
            // gpuHmac(in, out + i);
        }
        else
        {
            gpuAes(in, NULL);
            gpuHmac(in, NULL);
        }

        in += packetLen;
    }
}