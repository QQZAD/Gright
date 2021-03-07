#include <stdio.h>
#define PACKET_TRANSFER_MODE
#include "../../gpu_core/gpu_schedule.hpp"

extern __device__ void gpuIpv4Router(uint32_t packetsPerTh, uint8_t *in, uint32_t *out);
extern __device__ void gpuIpsec(uint32_t packetsPerTh, uint8_t *in, uint32_t *out);
extern __device__ void gpuIds(uint32_t packetsPerTh, uint8_t *in, uint32_t *out);
extern __device__ void gpuNat(uint32_t packetsPerTh, uint8_t *in, uint32_t *out);

extern void initIpv4Router(const char *routingInfo);
extern void initIpsec(uint32_t srcAddr = SRC_ADDR);
extern void initIds(const char *acRule);
extern void initNat();

extern void freeIpv4Router();
extern void freeIpsec();
extern void freeIds();
extern void freeNat();

/*PTM_BATCH是8的倍数*/
#define PTM_BATCH 208
#define INIT_PTM_BATCH_SIZE 32
#define PTM_BATCH_SIZE_CHANGE 64

static char ipv4Router[256] = "ipv4_router_experiment";
static char ipsec[256] = "ipsec_experiment";
static char ids[256] = "ids_experiment";
static char nat[256] = "nat_experiment";

static void packetTransferMode(int batch, uint32_t &nfId, uint32_t &batchSize, int &ptm)
{
    int nf01 = PTM_BATCH / 4 - 1;
    int nf12 = 2 * PTM_BATCH / 4 - 1;
    int nf23 = 3 * PTM_BATCH / 4 - 1;
    if (batch <= nf01)
    {
        nfId = IPV4_ROUTER_GPU;
        batchSize = INIT_PTM_BATCH_SIZE + PTM_BATCH_SIZE_CHANGE * (batch / 2);
    }
    else if (batch <= nf12)
    {
        nfId = IPSEC_GPU;
        batchSize = INIT_PTM_BATCH_SIZE + PTM_BATCH_SIZE_CHANGE * ((batch - (nf01 + 1)) / 2);
    }
    else if (batch <= nf23)
    {
        nfId = IDS_GPU;
        batchSize = INIT_PTM_BATCH_SIZE + PTM_BATCH_SIZE_CHANGE * ((batch - (nf12 + 1)) / 2);
    }
    else
    {
        nfId = NAT_GPU;
        batchSize = INIT_PTM_BATCH_SIZE + PTM_BATCH_SIZE_CHANGE * ((batch - (nf23 + 1)) / 2);
    }
    if (batch % 2 == 0)
    {
        ptm = DATA_COPY;
    }
    else
    {
        ptm = ADDR_MAP;
    }
    printf("batch-%d-batchSize-%u-nfId-%u-ptm-%d\n", batch, batchSize, nfId, ptm);
}

static double calTime(struct timespec start, struct timespec end)
{
    return end.tv_sec - start.tv_sec + pow(10, -9) * (end.tv_nsec - start.tv_nsec);
}

static void saveTime(int batchSize, int nfId, int currPtm, double nsec)
{
    char filename[512] = "../../data/";
    if (currPtm == DATA_COPY)
    {
        memcpy(filename + strlen(filename), "packet_data_copy/dc_", strlen("packet_data_copy/dc_"));
    }
    else if (currPtm == ADDR_MAP)
    {
        memcpy(filename + strlen(filename), "packet_addr_map/am_", strlen("packet_addr_map/am_"));
    }
    if (nfId == IPV4_ROUTER_GPU)
    {
        memcpy(filename + strlen(filename), ipv4Router, strlen(ipv4Router));
    }
    else if (nfId == IPSEC_GPU)
    {
        memcpy(filename + strlen(filename), ipsec, strlen(ipsec));
    }
    else if (nfId == IDS_GPU)
    {
        memcpy(filename + strlen(filename), ids, strlen(ids));
    }
    else
    {
        memcpy(filename + strlen(filename), nat, strlen(nat));
    }
    // printf("%s\n", filename);
    FILE *fp = fopen(filename, "a+");
    fprintf(fp, "%d %lf\n", batchSize, nsec);
    fclose(fp);
}

static __global__ void gpuNF(int batchSize, int nfId, uint8_t *pInPackets, uint64_t *pPackets)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchSize)
    {
        pInPackets += pPackets[tid];
        if (nfId == IPV4_ROUTER_GPU)
        {
            gpuIpv4Router(1, pInPackets, NULL);
            // printf("gpuIpv4Router-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
        else if (nfId == IPSEC_GPU)
        {
            gpuIpsec(1, pInPackets, NULL);
            // printf("gpuIpsec-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
        else if (nfId == IDS_GPU)
        {
            gpuIds(1, pInPackets, NULL);
            // printf("gpuIds-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
        else if (nfId == NAT_GPU)
        {
            gpuNat(1, pInPackets, NULL);
            // printf("gpuIds-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
    }
}

static void initNf(char *argv[])
{
    initIpv4Router("../../gpu_core/ipv4_router/routing_info.txt");
    initIpsec();
    initIds("../../gpu_core/ids/ac_rule.txt");
    initNat();
    memcpy(ipv4Router + strlen(ipv4Router), argv[1], strlen(argv[1]));
    memcpy(ipv4Router + strlen(ipv4Router), ".txt", strlen(".txt"));
    // printf("%s\n", ipv4Router);
    memcpy(ipsec + strlen(ipsec), argv[1], strlen(argv[1]));
    memcpy(ipsec + strlen(ipsec), ".txt", strlen(".txt"));
    // printf("%s\n", ipsec);
    memcpy(ids + strlen(ids), argv[1], strlen(argv[1]));
    memcpy(ids + strlen(ids), ".txt", strlen(".txt"));
    // printf("%s\n", ids);
    memcpy(nat + strlen(nat), argv[1], strlen(argv[1]));
    memcpy(nat + strlen(nat), ".txt", strlen(".txt"));
    // printf("%s\n", nat);
}

int main(int argc, char *argv[])
{
    initNf(argv);
    uint64_t totalLen = 0;
    struct STaskEntry taskEntry;
    taskEntry.init();

    uint32_t batchSize = 0;
    uint32_t nfId = 0;
    int ptm = 0;
    int blockNb = 0;
    int blockSize = 0;

    uint32_t maxBatchSize = INIT_PTM_BATCH_SIZE + PTM_BATCH_SIZE_CHANGE * (PTM_BATCH / 4 - 1);
    CPacket *pac = new CPacket[maxBatchSize];
    for (uint32_t i = 0; i < maxBatchSize; i++)
    {
        totalLen += pac[i].bytesLen;
    }

    cudaMallocHost((void **)&taskEntry.pInPacketsHost, 2 * totalLen * sizeof(uint8_t), cudaHostAllocMapped);
    cudaMalloc((void **)&taskEntry.pInPackets, 2 * totalLen * sizeof(uint8_t));

    taskEntry.pInPacketsBack = taskEntry.pInPackets;

    // uint8_t *p = taskEntry.pInPacketsHost;
    // for (uint32_t i = 0; i < maxBatchSize; i++)
    // {
    //     uint16_t bytesLen = pac[i].bytesLen;
    //     memcpy(p, pac[i].bytes(), bytesLen * sizeof(uint8_t));
    //     p += bytesLen;
    // }

    uint64_t *_pPackets = (uint64_t *)malloc(maxBatchSize * sizeof(uint64_t));
    cudaMalloc((void **)&taskEntry.pPackets, maxBatchSize * sizeof(uint64_t));

    for (int batch = 0; batch < PTM_BATCH; batch++)
    {
        packetTransferMode(batch, nfId, batchSize, ptm);
        taskEntry.setValue(batch, 0, batchSize, nfId, ptm);

        blockNb = ((batchSize / 1024 == float(batchSize) / 1024) ? batchSize / 1024 : batchSize / 1024 + 1);
        blockSize = (batchSize >= 1024 ? 1024 : batchSize);
        printf("batchSize-%u-blockNb-%d-blockSize-%d\n", batchSize, blockNb, blockSize);

        // CPacket *pac = new CPacket[batchSize];
        totalLen = 0;
        for (uint32_t i = 0; i < batchSize; i++)
        {
            totalLen += pac[i].bytesLen;
        }

        // cudaMallocHost((void **)&taskEntry.pInPacketsHost, 2 * totalLen * sizeof(uint8_t), cudaHostAllocMapped);
        // cudaMalloc((void **)&taskEntry.pInPackets, 2 * totalLen * sizeof(uint8_t));

        uint8_t *p = taskEntry.pInPacketsHost;
        for (uint32_t i = 0; i < batchSize; i++)
        {
            uint16_t bytesLen = pac[i].bytesLen;
            memcpy(p, pac[i].bytes(), bytesLen * sizeof(uint8_t));
            p += bytesLen;
        }
        // delete[] pac;

        if (taskEntry.nfId == IPSEC_GPU)
        {
            uint8_t *out;
            uint64_t result = espEncap(taskEntry.pInPacketsHost, batchSize, out);
            memcpy(taskEntry.pInPacketsHost, out, result);
            free(out);
            totalLen = result;
        }

        // uint64_t *_pPackets = (uint64_t *)malloc(batchSize * sizeof(uint64_t));
        _pPackets[0] = 0;
        for (uint32_t i = 0; i < batchSize - 1; i++)
        {
            struct ether_header *ethh = (struct ether_header *)(taskEntry.pInPacketsHost + _pPackets[i]);
            struct iphdr *iph = (struct iphdr *)(ethh + 1);
            uint32_t packetLen = sizeof(struct ether_header) + ntohs(iph->tot_len);
            _pPackets[i + 1] = _pPackets[i] + packetLen;
        }

        // cudaMalloc((void **)&taskEntry.pPackets, batchSize * sizeof(uint64_t));
        cudaMemcpy(taskEntry.pPackets, _pPackets, batchSize * sizeof(uint64_t), cudaMemcpyHostToDevice);

        /*注意！在开始记录时间时应该避免CUDA的启用开销*/
        gpuNF<<<blockNb, blockSize>>>(blockSize, 0, taskEntry.pInPackets, taskEntry.pPackets);
        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &(taskEntry.startTransfer));

        if (taskEntry.currPtm == DATA_COPY)
        {
            taskEntry.pInPackets = taskEntry.pInPacketsBack;
            // if (taskEntry.nfId == IPV4_ROUTER_GPU)
            // {
            //     /*仅拷贝IP头*/
            //     uint8_t *pDes, *pSrc;
            //     for (int i = 0; i < batchSize; i++)
            //     {
            //         pDes = taskEntry.pInPackets + _pPackets[i] + sizeof(struct ether_header);
            //         pSrc = taskEntry.pInPacketsHost + _pPackets[i] + sizeof(struct ether_header);
            //         cudaMemcpy(pDes, pSrc, sizeof(struct iphdr), cudaMemcpyHostToDevice);
            //     }
            // }
            // else
            // {
            //     /*IDS IPsec 拷贝整个数据包*/
            //     cudaMemcpy(taskEntry.pInPackets, taskEntry.pInPacketsHost, totalLen * sizeof(uint8_t), cudaMemcpyHostToDevice);
            // }
            cudaMemcpy(taskEntry.pInPackets, taskEntry.pInPacketsHost, totalLen * sizeof(uint8_t), cudaMemcpyHostToDevice);
        }
        else if (taskEntry.currPtm == ADDR_MAP)
        {
            cudaHostGetDevicePointer<uint8_t>(&(taskEntry.pInPackets), (void *)taskEntry.pInPacketsHost, 0);
        }

        gpuNF<<<blockNb, blockSize>>>(batchSize, taskEntry.nfId, taskEntry.pInPackets, taskEntry.pPackets);
        cudaDeviceSynchronize();

        clock_gettime(CLOCK_MONOTONIC, &(taskEntry.endExec));
        double nsec = calTime(taskEntry.startTransfer, taskEntry.endExec);
        saveTime(taskEntry.nbPackets, taskEntry.nfId, taskEntry.currPtm, nsec);

        // cudaFreeHost(taskEntry.pInPacketsHost);
        // taskEntry.pInPackets = taskEntry.pInPacketsBack;
        // cudaFree(taskEntry.pInPackets);
        // cudaFree(taskEntry.pPackets);
        // free(_pPackets);

        sleep(1);
    }

    delete[] pac;

    cudaFreeHost(taskEntry.pInPacketsHost);
    cudaFree(taskEntry.pInPackets);
    cudaFree(taskEntry.pPackets);
    free(_pPackets);

    freeIpv4Router();
    freeIpsec();
    freeIds();
    freeNat();
    return 0;
}