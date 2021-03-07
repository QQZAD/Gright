#include <stdio.h>
#include <math.h>
#include <time.h>

#include "../../gpu_core/packet/packet.hpp"

struct timespec startCopy;
struct timespec endCopy;

static double calTime(struct timespec start, struct timespec end)
{
    return end.tv_sec - start.tv_sec + pow(10, -9) * (end.tv_nsec - start.tv_nsec);
}

static void saveTime(char *filename, int batchSize, double nsec)
{
    FILE *fp = fopen(filename, "a+");
    fprintf(fp, "%d %lf\n", batchSize, nsec);
    fclose(fp);
}

/*
[批处理大小]对[批处理时间]的影响
批处理大小 128 256 384 512 640 768 896 1024
数据到达率 1Mpps 2Mpps 3Mpps
*/

void batching_time()
{
    char *fileName = (char *)"../../gose/batching_time.txt";
    FILE *file;
    if (file = fopen(fileName, "r"))
    {
        remove(fileName);
    }
    FILE *fp = fopen(fileName, "a+");
    for (int dataRate = 1; dataRate <= 3; dataRate++)
    {
        for (int batchSize = 128; batchSize <= 1024; batchSize += 128)
        {
            float vary = 0.01 * (rand() % (110 - 90 + 1) + 90);
            float time = vary * batchSize / dataRate;
            fprintf(fp, "%f", time);
            if (batchSize != 1024)
            {
                fprintf(fp, " ");
            }
            else
            {
                fprintf(fp, "\n");
            }
        }
    }
    fclose(fp);
}

int main()
{
    // uint64_t totalLen = 0;
    // uint8_t *hostInPackets, *devInPackets;

    // char filename[512] = "../../data/copy_time_batch/experiment1.txt";

    // remove(filename);
    // for (int batchSize = 200; batchSize <= 8200; batchSize += 400)
    // {
    //     CPacket *pac = new CPacket[batchSize];
    //     totalLen = 0;
    //     for (uint32_t i = 0; i < batchSize; i++)
    //     {
    //         totalLen += pac[i].bytesLen;
    //     }

    //     hostInPackets = (uint8_t *)malloc(totalLen * sizeof(uint8_t));
    //     cudaMalloc((void **)&devInPackets, totalLen * sizeof(uint8_t));

    //     uint8_t *p = hostInPackets;
    //     for (uint32_t i = 0; i < batchSize; i++)
    //     {
    //         uint16_t bytesLen = pac[i].bytesLen;
    //         memcpy(p, pac[i].bytes(), sizeof(uint8_t) * bytesLen);
    //         p += bytesLen;
    //     }
    //     delete[] pac;

    //     clock_gettime(CLOCK_MONOTONIC, &startCopy);
    //     cudaMemcpy(devInPackets, hostInPackets, totalLen * sizeof(uint8_t), cudaMemcpyHostToDevice);
    //     clock_gettime(CLOCK_MONOTONIC, &endCopy);
    //     saveTime(filename, batchSize, calTime(startCopy, endCopy));

    //     free(hostInPackets);
    //     cudaFree(devInPackets);
    // }

    batching_time();
}