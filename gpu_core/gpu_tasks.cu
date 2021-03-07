#include "gpu_schedule.hpp"

extern uint32_t NUM_OF_SM;
extern struct STaskTable taskTable;
extern cudaStream_t streamDh;

bool scheduleExit = false;

double calTime(struct timespec start, struct timespec end)
{
    return end.tv_sec - start.tv_sec + pow(10, -9) * (end.tv_nsec - start.tv_nsec);
}

void *getGpuTasks(void *argc)
{
    for (int batch = 0;; batch++)
    {
        if (batch == BATCH)
        {
            break;
        }
        for (int i = 0; i < FLOWS_NB; i++)
        {
            int flowId = flowsOrder[i];
#ifndef GRIGHT
            // flowsBatch[flowId] = rand() % (1792 - 10 + 1) + 10;
            // assert(flowsBatch[flowId] <= MAX_BATCH_SIZE);
            // flowsNf[flowId] = rand() % (2 - 0 + 1) + 0;
            // flowsNf[flowId] = IPSEC_GPU;
            // flowsNf[flowId] = IDS_GPU;
            // flowsNf[flowId] = IPV4_ROUTER_GPU;
            // if (flowsNf[flowId] == IPSEC_GPU)
            // {
            //     flowsNf[flowId] = IDS_GPU;
            // }
            // flowsSm[flowId] = rand() % (NUM_OF_SM - 1 - 0 + 1) + 0;
#endif
            printf("[gpu tasks] batch-%d\n", batch + 1);
            taskTable.insertNewTask(batch, flowId, flowsBatch[flowId], flowsNf[flowId], flowsPtm[flowId], flowsSm[flowId]);
        }
    }
    return NULL;
}

/*保存处理结果*/
#ifdef SAVE_RESULTS
static void saveResults(uint32_t *results, uint32_t workId, uint32_t taskIndex, uint32_t smId, uint32_t nfId, uint32_t nbPackets, int nbExes)
{
    struct stat st = {0};
    if (stat("processing_results", &st) == -1)
    {
        mkdir("processing_results", 0755);
    }

    char fileName[45] = "processing_results/workid";
    int len = sizeof("processing_results/workid") - 1;
    char id[3];
    sprintf(id, "%02d", workId);
    char ind[4];
    sprintf(ind, "%03d", taskIndex);
    mempcpy(fileName + len, id, sizeof(id));
    len += sizeof(id) - 1;
    mempcpy(fileName + len, "_taskindex", sizeof("_taskindex"));
    len += sizeof("_taskindex") - 1;
    mempcpy(fileName + len, ind, sizeof(ind));
    len += sizeof(ind) - 1;
    mempcpy(fileName + len, ".txt", sizeof(".txt"));

#ifdef DEBUG_GPU_SCHEDULE
    printf("[gpu schedule] 正在保存workId-%d-taskIndex-%d的处理结果到", workId, taskIndex);
    printf("%s\n", fileName);
#endif

    remove(fileName);
    FILE *fp = fopen(fileName, "a+");
    fprintf(fp, "smId-%u\tnfName-%s\tnbPackets-%u\tnbExes-%u\n\n", smId, GET_NF_NAME(nfId), nbPackets, nbExes);
    /*一行显示多少数据包*/
    int rowNb = 20;
    for (int i = 0; i < nbPackets; i++)
    {
        if (i % rowNb == 0)
        {
            if (i / rowNb == nbPackets / rowNb)
            {
                fprintf(fp, "[packets index] %d - %d\n", i + 1, i + nbPackets % rowNb);
            }
            else
            {
                fprintf(fp, "[packets index] %d - %d\n", i + 1, i + rowNb);
            }
        }
        fprintf(fp, "%u", results[i]);
        if (i % rowNb != rowNb - 1)
        {
            fprintf(fp, "\t");
        }
        else
        {
            fprintf(fp, "\n\n");
        }
    }
    fclose(fp);
}
#endif

void *finishGpuTasks(void *argc)
{
    int smId = *((int *)argc);
    while (scheduleExit == false)
    {
        int taskIndex = taskTable.pTaskStartPerSm[smId];
        int flowId = taskTable.pTaskEntry[taskIndex].workId;
        /*taskIndex对应的任务槽存在任务*/
        if (taskTable.pNbExes[taskIndex] > 0)
        {
            /*等待该任务完成*/
            while (taskTable.pTaskEntry[taskIndex].isSave == false)
            {
            }
            clock_gettime(CLOCK_MONOTONIC, &(taskTable.pTaskEntry[taskIndex].endExec));
            printf("%d %lf\n", flowId, calTime(taskTable.pTaskEntry[taskIndex].startTransfer, taskTable.pTaskEntry[taskIndex].endExec));
#ifdef SAVE_RESULTS
            cudaMemcpyAsync(taskTable.pTaskEntry[taskIndex].pOutHost, taskTable.pTaskEntry[taskIndex].pOutDev, taskTable.pTaskEntry[taskIndex].nbPackets * sizeof(uint32_t), cudaMemcpyDeviceToHost, streamDh);
            if (err != 0)
            {
                printf("[cudaError] pOutHost-cudaMemcpyAsync返回0x%x\n", err);
                exit(1);
            }
            cudaStreamSynchronize(streamDh);
            saveResults(taskTable.pTaskEntry[taskIndex].pOutHost, taskTable.pTaskEntry[taskIndex].workId, taskIndex, smId, taskTable.pTaskEntry[taskIndex].nfId, taskTable.pTaskEntry[taskIndex].nbPackets, taskTable.pNbExes[taskIndex]);
#endif
            taskTable.pNbExes[taskIndex] = 0;
            taskTable.pTaskStartPerSm[smId] = NEXT_TASK_ID(taskTable.pTaskStartPerSm[smId], smId, 1);
        }
    }
    return NULL;
}