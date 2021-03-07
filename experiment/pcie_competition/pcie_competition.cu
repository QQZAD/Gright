#include <stdio.h>
#include "../../gpu_core/gpu_schedule.hpp"

#define INIT_BATCH_SIZE 1024
#define INCRE_BATCH_SIZE 1024
#define BATCH_NB 2

static int folwsNB = 2;

static cudaStream_t *stream;
static struct timespec *start;
static struct timespec *end;
static double *folwsTime;
static pthread_t *thread;
static int *num;

static uint8_t **pHost = NULL;
static uint8_t **pDevice = NULL;
static uint64_t totalLen = 0;

// static char dc[512] = "../../data/pcie_competition/experiment";
static char dc[512] = "../../gose/pcie_competition.txt";

static double calTime(struct timespec start, struct timespec end)
{
    return end.tv_sec - start.tv_sec + pow(10, -9) * (end.tv_nsec - start.tv_nsec);
}

static void saveTime(int folwsNB, uint32_t batchSize, double nsec)
{
    FILE *fp = fopen(dc, "a+");
    if (batchSize == INIT_BATCH_SIZE)
    {
        fprintf(fp, "%d %lf ", folwsNB, nsec);
    }
    else if (batchSize == INIT_BATCH_SIZE + INCRE_BATCH_SIZE * (BATCH_NB - 1))
    {
        fprintf(fp, "%lf\n", nsec);
    }
    fclose(fp);
}

/*pcie数据拷贝之间的竞争*/
static void dcCompetition(uint32_t batchSize)
{
    CPacket *pac = new CPacket[batchSize];
    totalLen = 0;
    for (uint32_t i = 0; i < batchSize; i++)
    {
        totalLen += pac[i].bytesLen;
    }

    for (int i = 0; i < folwsNB; i++)
    {
        pHost[i] = (uint8_t *)malloc(totalLen * sizeof(uint8_t));
        cudaMalloc((void **)&(pDevice[i]), totalLen * sizeof(uint8_t));
        uint8_t *p = pHost[i];
        for (uint32_t i = 0; i < batchSize; i++)
        {
            uint16_t bytesLen = pac[i].bytesLen;
            memcpy(p, pac[i].bytes(), bytesLen * sizeof(uint8_t));
            p += bytesLen;
        }
    }
    delete[] pac;

    sleep(1);

    for (int i = 0; i < folwsNB; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &(start[i]));
        cudaMemcpyAsync(pDevice[i], pHost[i], totalLen * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i]);
    }

    for (int i = 0; i < folwsNB; i++)
    {
        cudaStreamSynchronize(stream[i]);
        clock_gettime(CLOCK_MONOTONIC, &(end[i]));
    }

    for (int i = 0; i < folwsNB; i++)
    {
        folwsTime[i] = calTime(start[i], end[i]);
    }

    sleep(1);

    double averTime = 0;

    for (int i = 0; i < folwsNB; i++)
    {
        averTime += folwsTime[i];
    }
    averTime /= folwsNB;
    printf("folwsNB=%d batchSize=%u averTime=%lf\n", folwsNB, batchSize, averTime);
    saveTime(folwsNB, batchSize, averTime);

    for (int i = 0; i < folwsNB; i++)
    {
        free(pHost[i]);
        cudaFree(pDevice[i]);
    }
}

void pcieSerial()
{
    uint32_t batchSize = 1024;
    CPacket *pac = new CPacket[batchSize];
    totalLen = 0;
    for (uint32_t i = 0; i < batchSize; i++)
    {
        totalLen += pac[i].bytesLen;
    }

    for (int i = 0; i < folwsNB; i++)
    {
        pHost[i] = (uint8_t *)malloc(totalLen * sizeof(uint8_t));
        cudaMalloc((void **)&(pDevice[i]), totalLen * sizeof(uint8_t));
        uint8_t *p = pHost[i];
        for (uint32_t i = 0; i < batchSize; i++)
        {
            uint16_t bytesLen = pac[i].bytesLen;
            memcpy(p, pac[i].bytes(), bytesLen * sizeof(uint8_t));
            p += bytesLen;
        }
    }
    delete[] pac;

    struct timespec st;
    struct timespec ed;

    clock_gettime(CLOCK_MONOTONIC, &st);

    for (int i = 0; i < folwsNB; i++)
    {
        cudaMemcpy(pDevice[i], pHost[i], totalLen * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    clock_gettime(CLOCK_MONOTONIC, &ed);

    double serialTime = calTime(st, ed);

    printf("folwsNB=%d batchSize=%u serialTime=%lf\n", folwsNB, batchSize, serialTime);

    FILE *fp = fopen(dc, "a+");
    fprintf(fp, "%lf ", serialTime * pow(10, 3));
    fclose(fp);

    for (int i = 0; i < folwsNB; i++)
    {
        free(pHost[i]);
        cudaFree(pDevice[i]);
    }
}

void pcieParallel()
{
    uint32_t batchSize = 1024;
    CPacket *pac = new CPacket[batchSize];
    totalLen = 0;
    for (uint32_t i = 0; i < batchSize; i++)
    {
        totalLen += pac[i].bytesLen;
    }

    for (int i = 0; i < folwsNB; i++)
    {
        pHost[i] = (uint8_t *)malloc(totalLen * sizeof(uint8_t));
        cudaMalloc((void **)&(pDevice[i]), totalLen * sizeof(uint8_t));
        uint8_t *p = pHost[i];
        for (uint32_t i = 0; i < batchSize; i++)
        {
            uint16_t bytesLen = pac[i].bytesLen;
            memcpy(p, pac[i].bytes(), bytesLen * sizeof(uint8_t));
            p += bytesLen;
        }
    }
    delete[] pac;

    struct timespec st;
    struct timespec ed;

    clock_gettime(CLOCK_MONOTONIC, &st);

    for (int i = 0; i < folwsNB; i++)
    {
        cudaMemcpyAsync(pDevice[i], pHost[i], totalLen * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i]);
    }

    for (int i = 0; i < folwsNB; i++)
    {
        cudaStreamSynchronize(stream[i]);
    }

    clock_gettime(CLOCK_MONOTONIC, &ed);

    double parallelTime = calTime(st, ed);

    printf("folwsNB=%d batchSize=%u parallelTime=%lf\n", folwsNB, batchSize, parallelTime);

    FILE *fp = fopen(dc, "a+");
    fprintf(fp, "%lf\n", parallelTime * pow(10, 3));
    fclose(fp);

    for (int i = 0; i < folwsNB; i++)
    {
        free(pHost[i]);
        cudaFree(pDevice[i]);
    }
}

void pcieHybrid()
{
    uint32_t batchSize = 512;
    CPacket *pac = new CPacket[batchSize];
    totalLen = 0;
    for (uint32_t i = 0; i < batchSize; i++)
    {
        totalLen += pac[i].bytesLen;
    }

    for (int i = 0; i < folwsNB; i++)
    {
        pHost[i] = (uint8_t *)malloc(totalLen * sizeof(uint8_t));
        cudaMalloc((void **)&(pDevice[i]), totalLen * sizeof(uint8_t));
        uint8_t *p = pHost[i];
        for (uint32_t i = 0; i < batchSize; i++)
        {
            uint16_t bytesLen = pac[i].bytesLen;
            memcpy(p, pac[i].bytes(), bytesLen * sizeof(uint8_t));
            p += bytesLen;
        }
    }
    delete[] pac;

    struct timespec st;
    struct timespec ed;

    int unit = folwsNB / 2;

    clock_gettime(CLOCK_MONOTONIC, &st);

    for (int i = 0; i < unit; i++)
    {
        cudaMemcpyAsync(pDevice[i], pHost[i], totalLen * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i]);
    }

    for (int i = 0; i < unit; i++)
    {
        cudaStreamSynchronize(stream[i]);
        cudaMemcpyAsync(pDevice[i + unit], pHost[i + unit], totalLen * sizeof(uint8_t), cudaMemcpyHostToDevice, stream[i + unit]);
    }

    for (int i = unit; i < folwsNB; i++)
    {
        cudaStreamSynchronize(stream[i]);
    }

    clock_gettime(CLOCK_MONOTONIC, &ed);

    double parallelTime = calTime(st, ed);

    printf("folwsNB=%d batchSize=%u hybridTime=%lf\n", folwsNB, batchSize, parallelTime);

    FILE *fp = fopen(dc, "a+");
    fprintf(fp, "%lf\n", parallelTime);
    fclose(fp);

    for (int i = 0; i < folwsNB; i++)
    {
        free(pHost[i]);
        cudaFree(pDevice[i]);
    }
}

/*
[并发NF数量]对[并行PCIe总完成时间]和[串行PCIe总完成时间]的影响
批处理大小为1024
*/

int main(int argc, char *argv[])
{
    // memcpy(dc + strlen(dc), argv[1], strlen(argv[1]));
    // memcpy(dc + strlen(dc), ".txt", strlen(".txt"));

    FILE *file;
    if (file = fopen(dc, "r"))
    {
        remove(dc);
    }

    for (folwsNB = 5; folwsNB <= 40; folwsNB += 5)
    {
        pHost = (uint8_t **)malloc(folwsNB * sizeof(uint8_t *));
        pDevice = (uint8_t **)malloc(folwsNB * sizeof(uint8_t *));

        stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * folwsNB);
        start = (struct timespec *)malloc(sizeof(struct timespec) * folwsNB);
        end = (struct timespec *)malloc(sizeof(struct timespec) * folwsNB);
        folwsTime = (double *)malloc(sizeof(double) * folwsNB);
        thread = (pthread_t *)malloc(sizeof(pthread_t) * folwsNB);
        num = (int *)malloc(sizeof(int) * folwsNB);

        for (int i = 0; i < folwsNB; i++)
        {
            num[i] = i;
            cudaStreamCreate(&(stream[i]));
        }

        // for (uint32_t batchSize = INIT_BATCH_SIZE; batchSize <= INIT_BATCH_SIZE + INCRE_BATCH_SIZE * (BATCH_NB - 1); batchSize += INCRE_BATCH_SIZE)
        // {
        //     dcCompetition(batchSize);
        // }

        pcieSerial();
        pcieParallel();
        // pcieHybrid();

        for (int i = 0; i < folwsNB; i++)
        {
            cudaStreamDestroy(stream[i]);
        }

        free(thread);
        free(num);
        free(stream);
        free(start);
        free(end);
        free(folwsTime);

        free(pHost);
        free(pDevice);
    }

    return 0;
}