#include <stdio.h>
#include <sys/stat.h>
#include "../../gpu_core/gpu_schedule.hpp"

#define NORMAL 0
#define CONCURRENT 1
#define MULTITHREAD 2

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

static int userNb;
static struct STaskEntry *taskEntry;
static cudaStream_t *stream;
static uint64_t *totalLen;
static uint64_t *resultLen;
static uint8_t **_pInPackets;
static uint64_t **_pPackets;
static uint64_t **_pResults;
static float *cpTime;
static float *exeTime;
static float *resultTime;
static float *flowTime;
static cudaEvent_t *time1;
static cudaEvent_t *time2;
static cudaEvent_t *time3;
static cudaEvent_t *time4;
static cudaEvent_t *time5;
static cudaEvent_t *time6;
static cudaEvent_t *startTime;
static cudaEvent_t *endTime;
static int *timeDev;
static int *timeHost;
static float *timeGpu;
static pthread_t *vnfUser_t;
static int *userId;
static int *threadNb;
static int *blockNb;
static int *blockSize;
static int *pacPerTh;

static cudaDeviceProp deviceProp;

static __device__ __inline__ uint32_t getSmid()
{
    uint32_t smId;
    asm volatile("mov.u32 %0, %%smid;"
                 : "=r"(smId));
    return smId;
}

static __global__ void gpuNF(float gpuClock, int *gpuTime, int kernelId, int batchSize, int nfId, uint8_t *pInPackets, uint64_t *pPackets, uint32_t *pOutDev, uint64_t *pResults, int pacPerTh = 1)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t smId = getSmid();

    clock_t start;

    if (tid == 0)
    {
        start = clock();
    }

    __syncthreads();

    if (tid * pacPerTh < batchSize)
    {
        pInPackets += pPackets[tid * pacPerTh];
        pOutDev += pResults[tid * pacPerTh];
        pacPerTh = (batchSize - tid * pacPerTh <= pacPerTh) ? (batchSize - tid * pacPerTh) : pacPerTh;
        if (nfId == IPV4_ROUTER_GPU)
        {
            gpuIpv4Router(pacPerTh, pInPackets, pOutDev);
            // printf("gpuIpv4Router-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
        else if (nfId == IPSEC_GPU)
        {
            gpuIpsec(pacPerTh, pInPackets, pOutDev);
            // printf("gpuIpsec-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
        else if (nfId == IDS_GPU)
        {
            gpuIds(pacPerTh, pInPackets, pOutDev);
            // printf("gpuIds-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
        else if (nfId == NAT_GPU)
        {
            gpuNat(pacPerTh, pInPackets, pOutDev);
            // printf("gpuIds-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
    }

    __syncthreads();

    if (tid == 0)
    {
        float blockTime = 1000 * (clock() - start) / gpuClock;
        gpuTime[kernelId] = int(blockTime * pow(10, 6));
        printf("kernel=%d,SM=%u,Block=%u,NF=%s\n", kernelId, smId, blockIdx.x, GET_NF_NAME(nfId));
    }
}

static __global__ void st_gpuNF(float gpuClock, int *gpuTime, int kernelId, int batchSize, int nfId, uint8_t *pInPackets, uint64_t *pPackets, uint32_t *pOutDev, uint64_t *pResults, int pacPerTh = 1)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t smId = getSmid();

    clock_t start;

    if (tid == 0)
    {
        start = clock();
    }

    __syncthreads();

    if (tid * pacPerTh < batchSize)
    {
        pInPackets += pPackets[tid * pacPerTh];
        pOutDev += pResults[tid * pacPerTh];
        pacPerTh = (batchSize - tid * pacPerTh <= pacPerTh) ? (batchSize - tid * pacPerTh) : pacPerTh;
        if (nfId == IPV4_ROUTER_GPU)
        {
            gpuIpv4Router(pacPerTh, pInPackets, pOutDev);
            // printf("gpuIpv4Router-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
        else if (nfId == IPSEC_GPU)
        {
            gpuIpsec(pacPerTh, pInPackets, pOutDev);
            // printf("gpuIpsec-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
        else if (nfId == IDS_GPU)
        {
            gpuIds(pacPerTh, pInPackets, pOutDev);
            // printf("gpuIds-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
        else if (nfId == NAT_GPU)
        {
            gpuNat(pacPerTh, pInPackets, pOutDev);
            // printf("gpuIds-tid-%u-batchSize-%d-pInPackets-%p\n", tid, batchSize, pInPackets);
        }
    }

    __syncthreads();

    if (tid == 0)
    {
        float blockTime = 1000 * (clock() - start) / gpuClock;
        gpuTime[kernelId] = int(blockTime * pow(10, 6));
        printf("kernel=%d,SM=%u,Block=%u,NF=%s\n", kernelId, smId, blockIdx.x, GET_NF_NAME(nfId));
    }
}

static void initNf()
{
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
    initIpv4Router("../../gpu_core/ipv4_router/routing_info.txt");
    initIpsec();
    initIds("../../gpu_core/ids/ac_rule.txt");
    initNat();
}

void *vnfUser(void *args)
{
    int i = *((int *)args);

    cudaEventRecord(time3[i], stream[i]);
    //注意一个TB所能容纳的最大线程的数量是1024
    gpuNF<<<blockNb[i], blockSize[i], 0, stream[i]>>>(1000 * deviceProp.clockRate, timeDev, i, taskEntry[i].nbPackets, taskEntry[i].nfId, taskEntry[i].pInPackets, taskEntry[i].pPackets, taskEntry[i].pOutDev, taskEntry[i].pResults);
    cudaEventRecord(time4[i], stream[i]);

    return NULL;
}

/*
注意区分CUDA调用开销和kernel固定开销
CUDA调用开销：运行CUDA代码前的固定时间开销
kernel固定开销：运行kernel代码前的固定时间开销
NORMAL模式下kernel0包含CUDA调用开销
CONCURRENT模式下所有kernel均包含CUDA调用开销

所有时间单位都是ms

对四种NF的分析
0 ipv4_router 计算需求小    显存占用大    带宽需求小
1 ipsec       计算需求大    显存占用小    带宽需求大
2 ids         计算需求中    显存占用中    带宽需求大
3 nat         计算需求小    显存占用大    带宽需求小

1.探讨批处理大小和NF类型对packet拷贝时间的影响
[服务器]
b   32          256         512         768
nf0 0.009696    0.023808    0.040224    0.053056
nf1 0.008704    0.024448    0.037984    0.052416
nf2 0.009760    0.020640    0.045216    0.051136
nf3 0.008992    0.022304    0.039936    0.048928
av  0.009288    0.022800    0.040840    0.051384
[笔记本]
b   32          256         512         768
nf0 0.009728    0.020512    0.032672    0.044256
nf1 0.011168    0.022976    0.033728    0.048064
nf2 0.010048    0.020832    0.030688    0.044544
nf3 0.010400    0.018592    0.036768    0.043872
av  0.010336    0.020728    0.033464    0.045184

2.探讨批处理大小和NF类型对kernel固定开销的影响
[服务器]
b   32          256         512         768         1024
nf0 0.092999    0.095635    0.185124    0.199999    0.182881
nf1 3.224085    3.315989    3.324406    3.371000    3.439100
nf2 0.186273    0.193723    0.201518    0.217000    0.268297
nf3 0.103483    0.100367    0.109562    0.117128    0.232419
[笔记本]
b   32          256         512         768         1024
nf0 0.098076    0.104385    0.105358    0.112052    0.129310
nf1 3.532480    3.562363    3.479073    3.613949    4.068150
nf2 0.188254    0.201775    0.206135    0.204336    0.231786
nf3 0.103520    0.106297    0.112826    0.118166    0.132190

3.探讨批处理大小和NF类型对GPU端执行时间的影响
[服务器]
b   32
nf0 0.181074
nf1 24.903656
nf2 0.898943
nf3 0.219461
[笔记本]
b   32
nf0 0.197412
nf1 29.696060
nf2 1.064258
nf3 0.239616

4.探讨批处理大小和NF类型对结果拷贝时间的影响
[服务器]
b   32          256         512         768
nf0 0.015904    0.016160    0.017760    0.018272
nf1 0.025888    0.127008    0.180544    0.262816
nf2 0.017952    0.018560    0.022816    0.056256
nf3 0.020736    0.056224    0.089920    0.141280
[笔记本]
b   32          256         512         768
nf0 0.016864    0.016896    0.017472    0.017376
nf1 0.024096    0.074912    0.107264    0.168704
nf2 0.015392    0.016512    0.022144    0.021888
nf3 0.019840    0.045088    0.083328    0.114080

5.[并发NF数量]对[NF平均执行时间]的影响（IPsec=1 IDS=2）CONCURRENT
批处理大小 1024
4   29.067743
8   29.081177
12  29.895155
16  32.897644
20  34.029507
24  34.856991
28  34.901348
32  35.202816

6.[GPU线程数量]对[Kernel固定开销与NF执行时间之比]的影响（IPv4 Router=0 NAT=3）NORMAL
批处理大小 512
32  12.508062 11.601877
64  12.718633 12.342840
96  13.951083 13.484018
128 14.543674 14.129029
160 15.474757 14.998028
192 17.822798 16.784811
224 18.414654 17.163008
256 20.145590 18.579493

7.kernel固定开销的拟合结果
b   384,        448,        512,        576,        640,        704,        768,        832,        896,        960,        1024
nf0 0.106489,   0.108856,   0.109404,   0.110823,   0.113827,   0.114155,   0.116921,   0.119526,   0.119823,   0.123157,   0.122592
nf1 3.090656,   3.105534,   3.203785,   3.379311,   3.433670,   3.498293,   3.541555,   3.508383,   3.475975,   3.539230,   3.580465
nf2 0.199170,   0.206576,   0.206679,   0.205573,   0.208591,   0.213596,   0.207549,   0.215795,   0.217216,   0.225121,   0.225726
nf3 0.110405,   0.113905,   0.111352,   0.113146,   0.114775,   0.116276,   0.117897,   0.124425,   0.121791,   0.127618,   0.129308

8.temporal_overhead
VNF     NF  批处理大小  空间线程数量    时间线程数量
(0)1    0   512*5     512           512*2
(1)2    3   512*5     512           512*2

b               256         384         512         640         768
0-spatial 1,2   0.978470    0.981200    1.045407    1.046526    1.046770
1-temporal 1->2 1.040317    1.044150    1.049140    1.051673    1.047339
*/

double calTime(struct timespec start, struct timespec end)
{
    return end.tv_sec - start.tv_sec + pow(10, -9) * (end.tv_nsec - start.tv_nsec);
}

void temporal_overhead_and_spatial_limitations(int exp, int spatial_temporal, int baseBatchsize = 512)
{
    /*
    exp=8 9
    spatial_temporal=0 spatial
    spatial_temporal=1 temporal
    */

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    if (exp == 8)
    {
        userNb = 2;
    }
    else if (exp == 9)
    {
        userNb = 3;
    }

    int flowThreadNb = 0;
    // int mode = NORMAL;

    CPacket *pac = NULL;

    taskEntry = (struct STaskEntry *)malloc(sizeof(struct STaskEntry) * userNb);
    stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * userNb);
    totalLen = (uint64_t *)malloc(sizeof(uint64_t) * userNb);
    resultLen = (uint64_t *)malloc(sizeof(uint64_t) * userNb);
    _pInPackets = (uint8_t **)malloc(sizeof(uint8_t *) * userNb);
    _pPackets = (uint64_t **)malloc(sizeof(uint64_t *) * userNb);
    _pResults = (uint64_t **)malloc(sizeof(uint64_t *) * userNb);

    time1 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);
    time2 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);
    time3 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);
    time4 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);
    time5 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);
    time6 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);

    struct timespec time_start;
    struct timespec time_end;

    cpTime = (float *)malloc(sizeof(float) * userNb);
    exeTime = (float *)malloc(sizeof(float) * userNb);
    resultTime = (float *)malloc(sizeof(float) * userNb);

    cudaMalloc((void **)&timeDev, sizeof(int) * userNb);
    timeHost = (int *)malloc(sizeof(int) * userNb);
    timeGpu = (float *)malloc(sizeof(float) * userNb);
    memset(timeHost, 0, sizeof(int) * userNb);
    cudaMemcpy(timeDev, timeHost, sizeof(int) * userNb, cudaMemcpyHostToDevice);

    vnfUser_t = (pthread_t *)malloc(sizeof(pthread_t) * userNb);
    userId = (int *)malloc(sizeof(int) * userNb);

    blockNb = (int *)malloc(sizeof(int) * userNb);
    blockSize = (int *)malloc(sizeof(int) * userNb);

    //采用ADDR_MAP方式的VNF的数量
    int amNb = 0;
    assert(amNb <= userNb);

    int pacPerTh = 0;

    for (int i = 0; i < userNb; i++)
    {
        taskEntry[i].init();
        if (exp == 8)
        {
            taskEntry[i].nbPackets = baseBatchsize * 5;
            taskEntry[0].nfId = 0;
            taskEntry[1].nfId = 3;
        }
        else if (exp == 9)
        {
            taskEntry[0].nbPackets = baseBatchsize * 2;
            taskEntry[1].nbPackets = baseBatchsize * 1;
            taskEntry[2].nbPackets = baseBatchsize * 6;
            taskEntry[0].nfId = 3;
            taskEntry[1].nfId = 2;
            taskEntry[2].nfId = 3;
        }
        if (spatial_temporal == 0)
        {
            flowThreadNb = baseBatchsize;
            if (exp == 9 && i == 2)
            {
                flowThreadNb = baseBatchsize * 2;
            }
        }
        else if (spatial_temporal == 1)
        {
            flowThreadNb = baseBatchsize * 2;
        }

        pacPerTh = taskEntry[i].nbPackets / flowThreadNb == float(taskEntry[i].nbPackets) / flowThreadNb ? (taskEntry[i].nbPackets / flowThreadNb) : (taskEntry[i].nbPackets / flowThreadNb + 1);

        if (amNb != 0)
        {
            taskEntry[i].currPtm = ADDR_MAP;
            amNb--;
        }
        else
        {
            taskEntry[i].currPtm = DATA_COPY;
        }

        cudaStreamCreate(stream + i);

        cudaEventCreate(time1 + i);
        cudaEventCreate(time2 + i);
        cudaEventCreate(time3 + i);
        cudaEventCreate(time4 + i);
        cudaEventCreate(time5 + i);
        cudaEventCreate(time6 + i);

        pac = new CPacket[taskEntry[i].nbPackets];
        totalLen[i] = 0;
        for (int j = 0; j < taskEntry[i].nbPackets; j++)
        {
            totalLen[i] += pac[j].bytesLen;
        }

        float scaleHd = 1.5;
        if (taskEntry[i].nfId == IPSEC_GPU)
        {
            // _pInPackets[i] = (uint8_t *)malloc(sizeof(uint8_t) * scaleHd * totalLen[i]);
            cudaMallocHost((void **)(_pInPackets + i), sizeof(uint8_t) * scaleHd * totalLen[i], cudaHostAllocMapped);
            cudaMalloc((void **)&(taskEntry[i].pInPackets), sizeof(uint8_t) * scaleHd * totalLen[i]);
        }
        else
        {
            // _pInPackets[i] = (uint8_t *)malloc(sizeof(uint8_t) * totalLen[i]);
            cudaMallocHost((void **)(_pInPackets + i), sizeof(uint8_t) * totalLen[i], cudaHostAllocMapped);
            cudaMalloc((void **)&(taskEntry[i].pInPackets), sizeof(uint8_t) * totalLen[i]);
        }
        _pPackets[i] = (uint64_t *)malloc(sizeof(uint64_t) * taskEntry[i].nbPackets);
        cudaMalloc((void **)&(taskEntry[i].pPackets), sizeof(uint64_t) * taskEntry[i].nbPackets);

        uint8_t *p = _pInPackets[i];
        for (int j = 0; j < taskEntry[i].nbPackets; j++)
        {
            uint16_t bytesLen = pac[j].bytesLen;
            memcpy(p, pac[j].bytes(), bytesLen * sizeof(uint8_t));
            p += bytesLen;
        }
        delete[] pac;

        if (taskEntry[i].nfId == IPSEC_GPU)
        {
            uint8_t *out;
            uint64_t result = espEncap(_pInPackets[i], taskEntry[i].nbPackets, out);
            memcpy(_pInPackets[i], out, result);
            free(out);
            totalLen[i] = result;
        }

        _pPackets[i][0] = 0;
        for (uint32_t j = 0; j < taskEntry[i].nbPackets - 1; j++)
        {
            struct ether_header *ethh = (struct ether_header *)(_pInPackets[i] + _pPackets[i][j]);
            struct iphdr *iph = (struct iphdr *)(ethh + 1);
            uint32_t packetLen = sizeof(struct ether_header) + ntohs(iph->tot_len);
            _pPackets[i][j + 1] = _pPackets[i][j] + packetLen;
        }

        userId[i] = i;

        // blockNb[i] = (taskEntry[i].nbPackets / 1024 == float(taskEntry[i].nbPackets) / 1024) ? taskEntry[i].nbPackets / 1024 : taskEntry[i].nbPackets / 1024 + 1;
        // blockSize[i] = (taskEntry[i].nbPackets >= 1024 ? 1024 : taskEntry[i].nbPackets);

        blockNb[i] = (flowThreadNb / 1024 == float(flowThreadNb) / 1024) ? flowThreadNb / 1024 : flowThreadNb / 1024 + 1;
        blockSize[i] = (flowThreadNb >= 1024 ? 1024 : flowThreadNb);

        if (exp == 8 || exp == 9)
        {
            blockSize[i] = deviceProp.multiProcessorCount;
        }

        /*分配处理结果的内存*/
        _pResults[i] = (uint64_t *)malloc(sizeof(uint64_t) * taskEntry[i].nbPackets);
        cudaMalloc((void **)&(taskEntry[i].pResults), sizeof(uint64_t) * taskEntry[i].nbPackets);

        float scaleDh = 1.8;
        if (taskEntry[i].nfId == IPV4_ROUTER_GPU)
        {
            resultLen[i] = taskEntry[i].nbPackets * sizeof(uint32_t);
        }
        else if (taskEntry[i].nfId == IPSEC_GPU)
        {
            resultLen[i] = scaleDh * totalLen[i];
        }
        else if (taskEntry[i].nfId == IDS_GPU)
        {
            resultLen[i] = taskEntry[i].nbPackets * sizeof(uint32_t);
        }
        else
        {
            resultLen[i] = totalLen[i];
        }

        for (uint32_t j = 0; j < taskEntry[i].nbPackets; j++)
        {
            if (taskEntry[i].nfId == IPV4_ROUTER_GPU)
            {
                _pResults[i][j] = j * sizeof(uint32_t);
            }
            else if (taskEntry[i].nfId == IPSEC_GPU)
            {
                _pResults[i][j] = _pPackets[i][j] * scaleDh;
            }
            else if (taskEntry[i].nfId == IDS_GPU)
            {
                _pResults[i][j] = j * sizeof(uint32_t);
            }
            else
            {
                _pResults[i][j] = _pPackets[i][j];
            }
        }

        taskEntry[i].pOutHost = (uint32_t *)malloc(sizeof(uint8_t) * resultLen[i]);
        cudaMalloc((void **)&(taskEntry[i].pOutDev), sizeof(uint8_t) * resultLen[i]);
    }

    for (int i = 0; i < userNb; i++)
    {
        if (taskEntry[i].currPtm == DATA_COPY)
        {
            cudaEventRecord(time1[i], stream[i]);
            cudaMemcpyAsync(taskEntry[i].pPackets, _pPackets[i], taskEntry[i].nbPackets * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(taskEntry[i].pInPackets, _pInPackets[i], totalLen[i] * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i]);
            cudaEventRecord(time2[i], stream[i]);
        }
        else if (taskEntry[i].currPtm == ADDR_MAP)
        {
            cudaEventRecord(time1[i], stream[i]);
            cudaMemcpyAsync(taskEntry[i].pPackets, _pPackets[i], taskEntry[i].nbPackets * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i]);
            cudaHostGetDevicePointer<uint8_t>(&(taskEntry[i].pInPackets), (void *)_pInPackets[i], 0);
            cudaEventRecord(time2[i], stream[i]);
        }
        cudaEventSynchronize(time1[i]);
        cudaEventSynchronize(time2[i]);
        cudaEventElapsedTime(cpTime + i, time1[i], time2[i]);
    }

    clock_gettime(CLOCK_MONOTONIC, &time_start);

    if (exp == 8)
    {
        if (spatial_temporal == 0)
        {
            for (int i = 0; i < userNb; i++)
            {
                cudaEventRecord(time3[i], stream[i]);
            }

            for (int i = 0; i < userNb; i++)
            {
                st_gpuNF<<<blockNb[i], blockSize[i], 0, stream[i]>>>(1000 * deviceProp.clockRate, timeDev, i, taskEntry[i].nbPackets, taskEntry[i].nfId, taskEntry[i].pInPackets, taskEntry[i].pPackets, taskEntry[i].pOutDev, taskEntry[i].pResults, taskEntry[i].nbPackets / flowThreadNb);
            }

            for (int i = 0; i < userNb; i++)
            {
                cudaEventRecord(time4[i], stream[i]);
            }
        }
        else if (spatial_temporal == 1)
        {
            for (int i = 0; i < userNb; i++)
            {
                cudaEventRecord(time3[i], stream[i]);
                st_gpuNF<<<blockNb[i], blockSize[i], 0, stream[i]>>>(1000 * deviceProp.clockRate, timeDev, i, taskEntry[i].nbPackets, taskEntry[i].nfId, taskEntry[i].pInPackets, taskEntry[i].pPackets, taskEntry[i].pOutDev, taskEntry[i].pResults, taskEntry[i].nbPackets / flowThreadNb);
                cudaEventRecord(time4[i], stream[i]);
            }
        }
    }
    else if (exp == 9)
    {
        if (spatial_temporal == 0)
        {
            for (int i = 0; i < userNb - 1; i++)
            {
                cudaEventRecord(time3[i], stream[i]);
            }

            for (int i = 0; i < userNb - 1; i++)
            {
                st_gpuNF<<<blockNb[i], blockSize[i], 0, stream[i]>>>(1000 * deviceProp.clockRate, timeDev, i, taskEntry[i].nbPackets, taskEntry[i].nfId, taskEntry[i].pInPackets, taskEntry[i].pPackets, taskEntry[i].pOutDev, taskEntry[i].pResults, taskEntry[i].nbPackets / flowThreadNb);
            }

            for (int i = 0; i < userNb - 1; i++)
            {
                cudaEventRecord(time4[i], stream[i]);
            }

            for (int i = userNb - 1; i < userNb; i++)
            {
                cudaEventRecord(time3[i], stream[i]);
                st_gpuNF<<<blockNb[i], blockSize[i], 0, stream[i]>>>(1000 * deviceProp.clockRate, timeDev, i, taskEntry[i].nbPackets, taskEntry[i].nfId, taskEntry[i].pInPackets, taskEntry[i].pPackets, taskEntry[i].pOutDev, taskEntry[i].pResults, taskEntry[i].nbPackets / flowThreadNb);
                cudaEventRecord(time4[i], stream[i]);
            }
        }
        else if (spatial_temporal == 1)
        {
            for (int i = 0; i < userNb; i++)
            {
                cudaEventRecord(time3[i], stream[i]);
                st_gpuNF<<<blockNb[i], blockSize[i], 0, stream[i]>>>(1000 * deviceProp.clockRate, timeDev, i, taskEntry[i].nbPackets, taskEntry[i].nfId, taskEntry[i].pInPackets, taskEntry[i].pPackets, taskEntry[i].pOutDev, taskEntry[i].pResults, taskEntry[i].nbPackets / flowThreadNb);
                cudaEventRecord(time4[i], stream[i]);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &time_end);

    for (int i = 0; i < userNb; i++)
    {
        cudaEventSynchronize(time3[i]);
        cudaEventSynchronize(time4[i]);
        cudaEventElapsedTime(exeTime + i, time3[i], time4[i]);

        cudaEventRecord(time5[i], stream[i]);
        cudaMemcpyAsync(_pResults[i], taskEntry[i].pResults, taskEntry[i].nbPackets * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream[i]);
        cudaMemcpyAsync(taskEntry[i].pOutHost, taskEntry[i].pOutDev, resultLen[i] * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream[i]);
        cudaEventRecord(time6[i], stream[i]);

        cudaEventSynchronize(time5[i]);
        cudaEventSynchronize(time6[i]);
        cudaEventElapsedTime(resultTime + i, time5[i], time6[i]);
    }

    float averGpu = 0;
    float averTime = 0;
    float totalTime = 0;
    for (int i = 0; i < userNb; i++)
    {
        cudaMemcpy(timeHost, timeDev, sizeof(float) * userNb, cudaMemcpyDeviceToHost);
        timeGpu[i] = (float)timeHost[i] / pow(10, 6);
        if (i != 0)
        {
            // printf("kernel=%d,Batch=%d,NF=%s,Ptm=%d,packet拷贝时间=%fms,kernel固定开销=%fms,GPU端执行时间=%fms,结果拷贝时间=%fms,CPU端执行时间=%fms,pacPerTh=%d,kernel固定开销/GPU端执行时间=%f,\n", i, taskEntry[i].nbPackets, GET_NF_NAME(taskEntry[i].nfId), taskEntry[i].currPtm, cpTime[i], exeTime[i] - timeGpu[i], timeGpu[i], resultTime[i], exeTime[i], pacPerTh, 100 * (exeTime[i] - timeGpu[i]) / timeGpu[i]);
        }
        averGpu += timeGpu[i];
        averTime += (cpTime[i] + exeTime[i]);
    }
    averGpu /= userNb;
    averTime /= userNb;

    float temp = 0;
    cudaEventElapsedTime(&temp, time1[0], time2[userNb - 1]);
    totalTime += temp;

    float totalExec = 0;
    for (int i = 0; i < userNb; i++)
    {
        for (int j = 0; j < userNb; j++)
        {
            cudaEventElapsedTime(&temp, time3[i], time4[j]);
            if (temp > totalExec)
            {
                totalExec = temp;
            }
        }
    }
    totalTime += totalExec;
    // printf("averTime=%fms,averGpu=%fms,totalTime=%fms\n", averTime, averGpu, totalTime);
    printf("exp=%d,spatial_temporal=%d,time=%lfms,baseBatchsize=%d\n", exp, spatial_temporal, 1000 * calTime(time_start, time_end), baseBatchsize);

    for (int i = 0; i < userNb; i++)
    {
        cudaFreeHost(_pInPackets[i]);
        free(_pPackets[i]);
        free(_pResults[i]);
        free(taskEntry[i].pOutHost);
        cudaFree(taskEntry[i].pInPackets);
        cudaFree(taskEntry[i].pPackets);
        cudaFree(taskEntry[i].pOutDev);
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(time1[i]);
        cudaEventDestroy(time2[i]);
        cudaEventDestroy(time3[i]);
        cudaEventDestroy(time4[i]);
        cudaEventDestroy(time5[i]);
        cudaEventDestroy(time6[i]);
    }

    cudaFree(timeDev);
    free(blockNb);
    free(blockSize);
    free(userId);
    free(vnfUser_t);
    free(cpTime);
    free(exeTime);
    free(resultTime);
    free(timeHost);
    free(timeGpu);
    free(time1);
    free(time2);
    free(time3);
    free(time4);
    free(time5);
    free(time6);
    free(taskEntry);
    free(stream);
    free(totalLen);
    free(resultLen);
    free(_pInPackets);
    free(_pPackets);
    free(_pResults);
}

void profiler()
{
    userNb = 4;
    int flowThreadNb = 1024;
    int mode = CONCURRENT;

    CPacket *pac = NULL;

    taskEntry = (struct STaskEntry *)malloc(sizeof(struct STaskEntry) * userNb);
    stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * userNb);
    totalLen = (uint64_t *)malloc(sizeof(uint64_t) * userNb);
    resultLen = (uint64_t *)malloc(sizeof(uint64_t) * userNb);
    _pInPackets = (uint8_t **)malloc(sizeof(uint8_t *) * userNb);
    _pPackets = (uint64_t **)malloc(sizeof(uint64_t *) * userNb);
    _pResults = (uint64_t **)malloc(sizeof(uint64_t *) * userNb);

    time1 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);
    time2 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);
    time3 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);
    time4 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);
    time5 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);
    time6 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);

    cpTime = (float *)malloc(sizeof(float) * userNb);
    exeTime = (float *)malloc(sizeof(float) * userNb);
    resultTime = (float *)malloc(sizeof(float) * userNb);

    cudaMalloc((void **)&timeDev, sizeof(int) * userNb);
    timeHost = (int *)malloc(sizeof(int) * userNb);
    timeGpu = (float *)malloc(sizeof(float) * userNb);
    memset(timeHost, 0, sizeof(int) * userNb);
    cudaMemcpy(timeDev, timeHost, sizeof(int) * userNb, cudaMemcpyHostToDevice);

    vnfUser_t = (pthread_t *)malloc(sizeof(pthread_t) * userNb);
    userId = (int *)malloc(sizeof(int) * userNb);

    blockNb = (int *)malloc(sizeof(int) * userNb);
    blockSize = (int *)malloc(sizeof(int) * userNb);

    //采用ADDR_MAP方式的VNF的数量
    int amNb = 0;
    assert(amNb <= userNb);

    int pacPerTh = 0;

    for (int i = 0; i < userNb; i++)
    {
        taskEntry[i].init();
        // taskEntry[i].nbPackets = 256 * i + 256;
        taskEntry[i].nbPackets = 256;
        taskEntry[i].nbPackets = flowThreadNb;
        // taskEntry[i].nfId = i % 4;
        // taskEntry[i].nfId = i - 1 >= 0 ? i - 1 : 0;
        taskEntry[i].nfId = 2;

        pacPerTh = taskEntry[i].nbPackets / flowThreadNb == float(taskEntry[i].nbPackets) / flowThreadNb ? (taskEntry[i].nbPackets / flowThreadNb) : (taskEntry[i].nbPackets / flowThreadNb + 1);

        if (amNb != 0)
        {
            taskEntry[i].currPtm = ADDR_MAP;
            amNb--;
        }
        else
        {
            taskEntry[i].currPtm = DATA_COPY;
        }

        cudaStreamCreate(stream + i);

        cudaEventCreate(time1 + i);
        cudaEventCreate(time2 + i);
        cudaEventCreate(time3 + i);
        cudaEventCreate(time4 + i);
        cudaEventCreate(time5 + i);
        cudaEventCreate(time6 + i);

        pac = new CPacket[taskEntry[i].nbPackets];
        totalLen[i] = 0;
        for (int j = 0; j < taskEntry[i].nbPackets; j++)
        {
            totalLen[i] += pac[j].bytesLen;
        }

        float scaleHd = 1.5;
        if (taskEntry[i].nfId == IPSEC_GPU)
        {
            // _pInPackets[i] = (uint8_t *)malloc(sizeof(uint8_t) * scaleHd * totalLen[i]);
            cudaMallocHost((void **)(_pInPackets + i), sizeof(uint8_t) * scaleHd * totalLen[i], cudaHostAllocMapped);
            cudaMalloc((void **)&(taskEntry[i].pInPackets), sizeof(uint8_t) * scaleHd * totalLen[i]);
        }
        else
        {
            // _pInPackets[i] = (uint8_t *)malloc(sizeof(uint8_t) * totalLen[i]);
            cudaMallocHost((void **)(_pInPackets + i), sizeof(uint8_t) * totalLen[i], cudaHostAllocMapped);
            cudaMalloc((void **)&(taskEntry[i].pInPackets), sizeof(uint8_t) * totalLen[i]);
        }
        _pPackets[i] = (uint64_t *)malloc(sizeof(uint64_t) * taskEntry[i].nbPackets);
        cudaMalloc((void **)&(taskEntry[i].pPackets), sizeof(uint64_t) * taskEntry[i].nbPackets);

        uint8_t *p = _pInPackets[i];
        for (int j = 0; j < taskEntry[i].nbPackets; j++)
        {
            uint16_t bytesLen = pac[j].bytesLen;
            memcpy(p, pac[j].bytes(), bytesLen * sizeof(uint8_t));
            p += bytesLen;
        }
        delete[] pac;

        if (taskEntry[i].nfId == IPSEC_GPU)
        {
            uint8_t *out;
            uint64_t result = espEncap(_pInPackets[i], taskEntry[i].nbPackets, out);
            memcpy(_pInPackets[i], out, result);
            free(out);
            totalLen[i] = result;
        }

        _pPackets[i][0] = 0;
        for (uint32_t j = 0; j < taskEntry[i].nbPackets - 1; j++)
        {
            struct ether_header *ethh = (struct ether_header *)(_pInPackets[i] + _pPackets[i][j]);
            struct iphdr *iph = (struct iphdr *)(ethh + 1);
            uint32_t packetLen = sizeof(struct ether_header) + ntohs(iph->tot_len);
            _pPackets[i][j + 1] = _pPackets[i][j] + packetLen;
        }

        userId[i] = i;

        // blockNb[i] = (taskEntry[i].nbPackets / 1024 == float(taskEntry[i].nbPackets) / 1024) ? taskEntry[i].nbPackets / 1024 : taskEntry[i].nbPackets / 1024 + 1;
        // blockSize[i] = (taskEntry[i].nbPackets >= 1024 ? 1024 : taskEntry[i].nbPackets);

        blockNb[i] = (flowThreadNb / 1024 == float(flowThreadNb) / 1024) ? flowThreadNb / 1024 : flowThreadNb / 1024 + 1;
        blockSize[i] = (flowThreadNb >= 1024 ? 1024 : flowThreadNb);

        /*分配处理结果的内存*/
        _pResults[i] = (uint64_t *)malloc(sizeof(uint64_t) * taskEntry[i].nbPackets);
        cudaMalloc((void **)&(taskEntry[i].pResults), sizeof(uint64_t) * taskEntry[i].nbPackets);

        float scaleDh = 1.8;
        if (taskEntry[i].nfId == IPV4_ROUTER_GPU)
        {
            resultLen[i] = taskEntry[i].nbPackets * sizeof(uint32_t);
        }
        else if (taskEntry[i].nfId == IPSEC_GPU)
        {
            resultLen[i] = scaleDh * totalLen[i];
        }
        else if (taskEntry[i].nfId == IDS_GPU)
        {
            resultLen[i] = taskEntry[i].nbPackets * sizeof(uint32_t);
        }
        else
        {
            resultLen[i] = totalLen[i];
        }

        for (uint32_t j = 0; j < taskEntry[i].nbPackets; j++)
        {
            if (taskEntry[i].nfId == IPV4_ROUTER_GPU)
            {
                _pResults[i][j] = j * sizeof(uint32_t);
            }
            else if (taskEntry[i].nfId == IPSEC_GPU)
            {
                _pResults[i][j] = _pPackets[i][j] * scaleDh;
            }
            else if (taskEntry[i].nfId == IDS_GPU)
            {
                _pResults[i][j] = j * sizeof(uint32_t);
            }
            else
            {
                _pResults[i][j] = _pPackets[i][j];
            }
        }

        taskEntry[i].pOutHost = (uint32_t *)malloc(sizeof(uint8_t) * resultLen[i]);
        cudaMalloc((void **)&(taskEntry[i].pOutDev), sizeof(uint8_t) * resultLen[i]);
    }

    for (int i = 0; i < userNb; i++)
    {
        if (taskEntry[i].currPtm == DATA_COPY)
        {
            cudaEventRecord(time1[i], stream[i]);
            cudaMemcpyAsync(taskEntry[i].pPackets, _pPackets[i], taskEntry[i].nbPackets * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i]);
            cudaMemcpyAsync(taskEntry[i].pInPackets, _pInPackets[i], totalLen[i] * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i]);
            cudaEventRecord(time2[i], stream[i]);
        }
        else if (taskEntry[i].currPtm == ADDR_MAP)
        {
            cudaEventRecord(time1[i], stream[i]);
            cudaMemcpyAsync(taskEntry[i].pPackets, _pPackets[i], taskEntry[i].nbPackets * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i]);
            cudaHostGetDevicePointer<uint8_t>(&(taskEntry[i].pInPackets), (void *)_pInPackets[i], 0);
            cudaEventRecord(time2[i], stream[i]);
        }
        cudaEventSynchronize(time1[i]);
        cudaEventSynchronize(time2[i]);
        cudaEventElapsedTime(cpTime + i, time1[i], time2[i]);
    }

    if (mode == NORMAL)
    {
        for (int i = 0; i < userNb; i++)
        {
            cudaEventRecord(time3[i], stream[i]);
            //注意一个TB所能容纳的最大线程的数量是1024
            gpuNF<<<blockNb[i], blockSize[i], 0, stream[i]>>>(1000 * deviceProp.clockRate, timeDev, i, taskEntry[i].nbPackets, taskEntry[i].nfId, taskEntry[i].pInPackets, taskEntry[i].pPackets, taskEntry[i].pOutDev, taskEntry[i].pResults, taskEntry[i].nbPackets / flowThreadNb);
            cudaEventRecord(time4[i], stream[i]);
        }
    }
    else if (mode == CONCURRENT)
    {
        for (int i = 0; i < userNb; i++)
        {
            cudaEventRecord(time3[i], stream[i]);
        }

        for (int i = 0; i < userNb; i++)
        {
            //注意一个TB所能容纳的最大线程的数量是1024
            gpuNF<<<blockNb[i], blockSize[i], 0, stream[i]>>>(1000 * deviceProp.clockRate, timeDev, i, taskEntry[i].nbPackets, taskEntry[i].nfId, taskEntry[i].pInPackets, taskEntry[i].pPackets, taskEntry[i].pOutDev, taskEntry[i].pResults, taskEntry[i].nbPackets / flowThreadNb);
        }

        for (int i = 0; i < userNb; i++)
        {
            cudaEventRecord(time4[i], stream[i]);
        }
    }
    else if (mode == MULTITHREAD)
    {
        for (int i = 0; i < userNb; i++)
        {
            pthread_create(vnfUser_t + i, NULL, vnfUser, (void *)(userId + i));
        }

        for (int i = 0; i < userNb; i++)
        {
            pthread_join(vnfUser_t[i], NULL);
        }
    }

    for (int i = 0; i < userNb; i++)
    {
        cudaEventSynchronize(time3[i]);
        cudaEventSynchronize(time4[i]);
        cudaEventElapsedTime(exeTime + i, time3[i], time4[i]);

        cudaEventRecord(time5[i], stream[i]);
        cudaMemcpyAsync(_pResults[i], taskEntry[i].pResults, taskEntry[i].nbPackets * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream[i]);
        cudaMemcpyAsync(taskEntry[i].pOutHost, taskEntry[i].pOutDev, resultLen[i] * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream[i]);
        cudaEventRecord(time6[i], stream[i]);

        cudaEventSynchronize(time5[i]);
        cudaEventSynchronize(time6[i]);
        cudaEventElapsedTime(resultTime + i, time5[i], time6[i]);
    }

    float averGpu = 0;
    float averTime = 0;
    float totalTime = 0;
    for (int i = 0; i < userNb; i++)
    {
        cudaMemcpy(timeHost, timeDev, sizeof(float) * userNb, cudaMemcpyDeviceToHost);
        timeGpu[i] = (float)timeHost[i] / pow(10, 6);
        if (i != 0)
        {
            printf("kernel=%d,Batch=%d,NF=%s,Ptm=%d,packet拷贝时间=%fms,kernel固定开销=%fms,GPU端执行时间=%fms,结果拷贝时间=%fms,CPU端执行时间=%fms,pacPerTh=%d,kernel固定开销/GPU端执行时间=%f,\n", i, taskEntry[i].nbPackets, GET_NF_NAME(taskEntry[i].nfId), taskEntry[i].currPtm, cpTime[i], exeTime[i] - timeGpu[i], timeGpu[i], resultTime[i], exeTime[i], pacPerTh, 100 * (exeTime[i] - timeGpu[i]) / timeGpu[i]);
        }
        averGpu += timeGpu[i];
        averTime += (cpTime[i] + exeTime[i]);
    }
    averGpu /= userNb;
    averTime /= userNb;

    float temp = 0;
    cudaEventElapsedTime(&temp, time1[0], time2[userNb - 1]);
    totalTime += temp;

    float totalExec = 0;
    for (int i = 0; i < userNb; i++)
    {
        for (int j = 0; j < userNb; j++)
        {
            cudaEventElapsedTime(&temp, time3[i], time4[j]);
            if (temp > totalExec)
            {
                totalExec = temp;
            }
        }
    }
    totalTime += totalExec;
    printf("averTime=%fms,averGpu=%fms,totalTime=%fms\n", averTime, averGpu, totalTime);

    for (int i = 0; i < userNb; i++)
    {
        cudaFreeHost(_pInPackets[i]);
        free(_pPackets[i]);
        free(_pResults[i]);
        free(taskEntry[i].pOutHost);
        cudaFree(taskEntry[i].pInPackets);
        cudaFree(taskEntry[i].pPackets);
        cudaFree(taskEntry[i].pOutDev);
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(time1[i]);
        cudaEventDestroy(time2[i]);
        cudaEventDestroy(time3[i]);
        cudaEventDestroy(time4[i]);
        cudaEventDestroy(time5[i]);
        cudaEventDestroy(time6[i]);
    }

    cudaFree(timeDev);
    free(blockNb);
    free(blockSize);
    free(userId);
    free(vnfUser_t);
    free(cpTime);
    free(exeTime);
    free(resultTime);
    free(timeHost);
    free(timeGpu);
    free(time1);
    free(time2);
    free(time3);
    free(time4);
    free(time5);
    free(time6);
    free(taskEntry);
    free(stream);
    free(totalLen);
    free(resultLen);
    free(_pInPackets);
    free(_pPackets);
    free(_pResults);
}

void *vnfFlow(void *args)
{
    int i = *((int *)args);

    cudaEventRecord(startTime[i], stream[i]);

    cudaMemcpyAsync(taskEntry[i].pPackets, _pPackets[i], taskEntry[i].nbPackets * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i]);
    cudaMemcpyAsync(taskEntry[i].pInPackets, _pInPackets[i], totalLen[i] * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i]);

    //注意一个TB所能容纳的最大线程的数量是1024
    gpuNF<<<blockNb[i], blockSize[i], 0, stream[i]>>>(1000 * deviceProp.clockRate, timeDev, i, taskEntry[i].nbPackets, taskEntry[i].nfId, taskEntry[i].pInPackets, taskEntry[i].pPackets, taskEntry[i].pOutDev, taskEntry[i].pResults, pacPerTh[i]);

    cudaMemcpyAsync(_pResults[i], taskEntry[i].pResults, taskEntry[i].nbPackets * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream[i]);
    cudaMemcpyAsync(taskEntry[i].pOutHost, taskEntry[i].pOutDev, resultLen[i] * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream[i]);

    cudaEventRecord(endTime[i], stream[i]);

    return NULL;
}

//gose_delay_prediction 0
// static float flowRate[4] = {8, 6, 4, 2};
int getflowBatchSize(int flowId)
{
    return 124 + flowId * 60;
}

//gose_delay_drop 1
// static float flowRate[4] = {8, 16, 24, 32};
static float flowRate[4] = {12, 24, 36, 48};
static float flowBatchSize[4] = {512, 1024, 1536, 2048};

void gose(int type, int cond, bool isLSSTG = false)
{
    char str[2];
    sprintf(str, "%d", cond);
    char fileName[40] = "../../gose/delay_prediction/system0.txt";
    fileName[34] = str[0];

    //gose_delay_prediction
    if (type == 0)
    {
        userNb = 16;
        printf("%s\n", fileName);
    }
    //gose_delay_drop
    else if (type == 1)
    {
        userNb = 4 * 9;
    }

    CPacket *pac = NULL;

    taskEntry = (struct STaskEntry *)malloc(sizeof(struct STaskEntry) * userNb);
    stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * userNb);
    totalLen = (uint64_t *)malloc(sizeof(uint64_t) * userNb);
    resultLen = (uint64_t *)malloc(sizeof(uint64_t) * userNb);
    _pInPackets = (uint8_t **)malloc(sizeof(uint8_t *) * userNb);
    _pPackets = (uint64_t **)malloc(sizeof(uint64_t *) * userNb);
    _pResults = (uint64_t **)malloc(sizeof(uint64_t *) * userNb);

    startTime = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);
    endTime = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * userNb);

    flowTime = (float *)malloc(sizeof(float) * userNb);

    cudaMalloc((void **)&timeDev, sizeof(int) * userNb);
    timeHost = (int *)malloc(sizeof(int) * userNb);
    memset(timeHost, 0, sizeof(int) * userNb);
    cudaMemcpy(timeDev, timeHost, sizeof(int) * userNb, cudaMemcpyHostToDevice);

    vnfUser_t = (pthread_t *)malloc(sizeof(pthread_t) * userNb);
    userId = (int *)malloc(sizeof(int) * userNb);

    threadNb = (int *)malloc(sizeof(int) * userNb);
    blockNb = (int *)malloc(sizeof(int) * userNb);
    blockSize = (int *)malloc(sizeof(int) * userNb);
    pacPerTh = (int *)malloc(sizeof(int) * userNb);

    int turn = userNb / 4;

    if (turn == 0)
    {
        turn = 1;
    }

    for (int i = 0; i < userNb; i++)
    {
        taskEntry[i].init();
        taskEntry[i].currPtm = DATA_COPY;
        if (type == 0)
        {
            taskEntry[i].nbPackets = getflowBatchSize(i);
        }
        else if (type == 1)
        {
            taskEntry[i].nbPackets = flowBatchSize[i / turn];
        }

        threadNb[i] = taskEntry[i].nbPackets;

        if (type == 1 && isLSSTG == true)
        {
            //根据LSSTG算法得到线程配置
            if (cond == 5)
            {
                threadNb[22] = 1024;
                threadNb[31] = 1024;
            }
            else
            {
                threadNb[19] = 1024;
                threadNb[29] = 1536;
            }
        }
        if (cond == 5)
        {
            taskEntry[i].nfId = i / turn;
        }
        else
        {
            taskEntry[i].nfId = cond - 1;
        }

        pacPerTh[i] = (float(taskEntry[i].nbPackets) / threadNb[i] == taskEntry[i].nbPackets / threadNb[i]) ? taskEntry[i].nbPackets / threadNb[i] : int(taskEntry[i].nbPackets / threadNb[i]) + 1;

        cudaStreamCreate(stream + i);

        cudaEventCreate(startTime + i);
        cudaEventCreate(endTime + i);

        pac = new CPacket[taskEntry[i].nbPackets];
        totalLen[i] = 0;
        for (int j = 0; j < taskEntry[i].nbPackets; j++)
        {
            totalLen[i] += pac[j].bytesLen;
        }

        float scaleHd = 1.5;
        if (taskEntry[i].nfId == IPSEC_GPU)
        {
            // _pInPackets[i] = (uint8_t *)malloc(sizeof(uint8_t) * scaleHd * totalLen[i]);
            cudaMallocHost((void **)(_pInPackets + i), sizeof(uint8_t) * scaleHd * totalLen[i], cudaHostAllocMapped);
            cudaMalloc((void **)&(taskEntry[i].pInPackets), sizeof(uint8_t) * scaleHd * totalLen[i]);
        }
        else
        {
            // _pInPackets[i] = (uint8_t *)malloc(sizeof(uint8_t) * totalLen[i]);
            cudaMallocHost((void **)(_pInPackets + i), sizeof(uint8_t) * totalLen[i], cudaHostAllocMapped);
            cudaMalloc((void **)&(taskEntry[i].pInPackets), sizeof(uint8_t) * totalLen[i]);
        }
        _pPackets[i] = (uint64_t *)malloc(sizeof(uint64_t) * taskEntry[i].nbPackets);
        cudaMalloc((void **)&(taskEntry[i].pPackets), sizeof(uint64_t) * taskEntry[i].nbPackets);

        uint8_t *p = _pInPackets[i];
        for (int j = 0; j < taskEntry[i].nbPackets; j++)
        {
            uint16_t bytesLen = pac[j].bytesLen;
            memcpy(p, pac[j].bytes(), bytesLen * sizeof(uint8_t));
            p += bytesLen;
        }
        delete[] pac;

        if (taskEntry[i].nfId == IPSEC_GPU)
        {
            uint8_t *out;
            uint64_t result = espEncap(_pInPackets[i], taskEntry[i].nbPackets, out);
            memcpy(_pInPackets[i], out, result);
            free(out);
            totalLen[i] = result;
        }

        _pPackets[i][0] = 0;
        for (uint32_t j = 0; j < taskEntry[i].nbPackets - 1; j++)
        {
            struct ether_header *ethh = (struct ether_header *)(_pInPackets[i] + _pPackets[i][j]);
            struct iphdr *iph = (struct iphdr *)(ethh + 1);
            uint32_t packetLen = sizeof(struct ether_header) + ntohs(iph->tot_len);
            _pPackets[i][j + 1] = _pPackets[i][j] + packetLen;
        }

        userId[i] = i;

        blockNb[i] = (threadNb[i] / 1024 == float(threadNb[i]) / 1024) ? threadNb[i] / 1024 : threadNb[i] / 1024 + 1;
        blockSize[i] = (threadNb[i] >= 1024 ? 1024 : threadNb[i]);

        /*分配处理结果的内存*/
        _pResults[i] = (uint64_t *)malloc(sizeof(uint64_t) * taskEntry[i].nbPackets);
        cudaMalloc((void **)&(taskEntry[i].pResults), sizeof(uint64_t) * taskEntry[i].nbPackets);

        float scaleDh = 1.8;
        if (taskEntry[i].nfId == IPV4_ROUTER_GPU)
        {
            resultLen[i] = taskEntry[i].nbPackets * sizeof(uint32_t);
        }
        else if (taskEntry[i].nfId == IPSEC_GPU)
        {
            resultLen[i] = scaleDh * totalLen[i];
        }
        else if (taskEntry[i].nfId == IDS_GPU)
        {
            resultLen[i] = taskEntry[i].nbPackets * sizeof(uint32_t);
        }
        else
        {
            resultLen[i] = totalLen[i];
        }

        for (uint32_t j = 0; j < taskEntry[i].nbPackets; j++)
        {
            if (taskEntry[i].nfId == IPV4_ROUTER_GPU)
            {
                _pResults[i][j] = j * sizeof(uint32_t);
            }
            else if (taskEntry[i].nfId == IPSEC_GPU)
            {
                _pResults[i][j] = _pPackets[i][j] * scaleDh;
            }
            else if (taskEntry[i].nfId == IDS_GPU)
            {
                _pResults[i][j] = j * sizeof(uint32_t);
            }
            else
            {
                _pResults[i][j] = _pPackets[i][j];
            }
        }

        taskEntry[i].pOutHost = (uint32_t *)malloc(sizeof(uint8_t) * resultLen[i]);
        cudaMalloc((void **)&(taskEntry[i].pOutDev), sizeof(uint8_t) * resultLen[i]);
    }

    for (int i = 0; i < userNb; i++)
    {
        pthread_create(vnfUser_t + i, NULL, vnfFlow, (void *)(userId + i));
    }

    for (int i = 0; i < userNb; i++)
    {
        pthread_join(vnfUser_t[i], NULL);
    }

    FILE *fp;
    if (type == 0)
    {
        remove(fileName);
        fp = fopen(fileName, "a+");
    }
    float averFlowTime = 0;
    for (int i = 0; i < userNb; i++)
    {
        cudaEventSynchronize(startTime[i]);
        cudaEventSynchronize(endTime[i]);
        cudaEventElapsedTime(flowTime + i, startTime[i], endTime[i]);
        float vary = 1;
        if (type == 0)
        {
            vary = 0.01 * (rand() % (110 - 90 + 1) + 90);
        }
        flowTime[i] += (vary * taskEntry[i].nbPackets / flowRate[i / turn]);
        if (type == 0)
        {
            printf("kernel=%d,NF=%s,Batch=%d,threadNb=%d,flowTime=%fms\n", i, GET_NF_NAME(taskEntry[i].nfId), taskEntry[i].nbPackets, threadNb[i], flowTime[i]);
            fprintf(fp, "%f\n", flowTime[i]);
        }
        averFlowTime += flowTime[i];
    }
    if (type == 0)
    {
        fclose(fp);
    }
    else if (type == 1)
    {
        averFlowTime /= userNb;
        printf("cond=%d,isLSSTG=%d,flowTime=%fms\n", cond, isLSSTG, averFlowTime);
    }

    for (int i = 0; i < userNb; i++)
    {
        cudaFreeHost(_pInPackets[i]);
        free(_pPackets[i]);
        free(_pResults[i]);
        free(taskEntry[i].pOutHost);
        cudaFree(taskEntry[i].pInPackets);
        cudaFree(taskEntry[i].pPackets);
        cudaFree(taskEntry[i].pOutDev);
        cudaStreamDestroy(stream[i]);
        cudaEventDestroy(startTime[i]);
        cudaEventDestroy(endTime[i]);
    }

    cudaFree(timeDev);
    free(threadNb);
    free(blockNb);
    free(blockSize);
    free(pacPerTh);
    free(userId);
    free(vnfUser_t);
    free(flowTime);
    free(timeHost);
    free(taskEntry);
    free(stream);
    free(totalLen);
    free(resultLen);
    free(_pInPackets);
    free(_pPackets);
    free(_pResults);
    free(startTime);
    free(endTime);
}

void delay_prediction()
{
    initNf();
    gose(0, 1);
    freeIpv4Router();
    freeIpsec();
    freeIds();
    freeNat();
}

void delay_drop(int cond)
{
    initNf();
    // gose(1, cond);
    gose(1, cond, true);
    freeIpv4Router();
    freeIpsec();
    freeIds();
    freeNat();
}

int main()
{
    initNf();
    // profiler();
    temporal_overhead_and_spatial_limitations(8, 0, 256);
    freeIpv4Router();
    freeIpsec();
    freeIds();
    freeNat();
    // delay_drop(5);
    return 0;
}