#include <cstdlib>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include "flows.hpp"
#include "../gpu_core/header/gpu_config.hpp"
#include "../gpu_core/packet/packet.hpp"

extern uint32_t NUM_OF_SM;

uint32_t flowsBatch[FLOWS_NB] = {0};

uint8_t *pFlowsPackets[FLOWS_NB] = {NULL};

int flowsOrder[FLOWS_NB] = {0};

int flowsNf[FLOWS_NB] = {0};

int flowsSm[FLOWS_NB] = {0};

int flowsPtm[FLOWS_NB] = {0};

void readFlowsInfo(bool gpuCore)
{
    FILE *fp;
    if (gpuCore == true)
    {
        fp = fopen("../data/raw_data/flows.txt", "r");
    }
    else
    {
        fp = fopen("data/raw_data/flows.txt", "r");
    }
    for (int i = 0; i < FLOWS_NB; i++)
    {
        fscanf(fp, "%d", &flowsOrder[i]);
        fscanf(fp, "%d", &flowsBatch[i]);
        // flowsBatch[i] = rand() % (1792 - 10 + 1) + 10;
        assert(flowsBatch[i] <= MAX_BATCH_SIZE);
        fscanf(fp, "%d", &flowsNf[i]);
        // flowsNf[i] = rand() % (2 - 0 + 1) + 0;
        fscanf(fp, "%d", &flowsSm[i]);
        // flowsSm[i] = rand() % (NUM_OF_SM - 1 - 0 + 1) + 0;
        fscanf(fp, "%d", &flowsPtm[i]);
    }
    fclose(fp);
}

void printFlowsInfo()
{
    printf("[gpu schedule] 获取流的参数信息\n");
    printf("BATCH %d\n", BATCH);
    printf("FLOWS_NB %d\n", FLOWS_NB);
    printf("MAX_TASKS_PER_SM %d\n\n", MAX_TASKS_PER_SM);

    for (int i = 0; i < FLOWS_NB; i++)
    {
        printf("[流%d]\n", i);
        printf("flowsOrder=%d\n", flowsOrder[i]);
        printf("flowsBatch=%d\n", flowsBatch[i]);
        printf("flowsNf=%s\n", GET_NF_NAME(flowsNf[i]));
        printf("flowsSm=%d\n", flowsSm[i]);
        printf("flowsPtm=%s\n", GET_PTM_NAME(flowsPtm[i]));
    }
    printf("\n");
}