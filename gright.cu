#include <stdio.h>
#include <assert.h>

#include "config/flows.hpp"
#include "gpu_core/gpu_schedule.hpp"

/*
config/flows.txt
flowsOrder flowsBatch flowsNf flowsSm ptm
*/

/*
flowsNf
IPV4_ROUTER_GPU 0
IPSEC_GPU 1
IDS_GPU 2
NAT_GPU 3
*/

/*
flowsSm
0,1,...,44,45
*/

/*
ptm
DATA_COPY=0
ADRR_MAP=1
*/

extern bool *tasksExit;
extern bool scheduleExit;
extern void *launchDpdk(void *_argc);
extern void initGpuSchedule();
extern __global__ void gpuSchedule(volatile struct STaskEntry *devTaskEntry, int *devNbExes, uint32_t *devTaskStartPerSm, uint32_t *devTaskEndPerSm, bool *devTasksExit, uint32_t EXECUTORS_PER_BLOCK);
extern void freeGpuSchedule();
extern void *getGpuTasks(void *argc);
extern void *finishGpuTasks(void *argc);

/*main函数的运行参数必须和dpdk保持一致，需要sudo权限*/
int main(int argc, char *argv[])
{
    SPar par(argc, argv);

    readFlowsInfo(false);
    printFlowsInfo();
    initGpuSchedule();

#ifdef PAUSE_WAIT
    printf("[gpu schedule] 按回车键继续...\n");
    system("read REPLY");
#endif

    pthread_t dpdk_t, getTasks_t;
    pthread_t *finishTasks_t = (pthread_t *)malloc(sizeof(pthread_t) * NUM_OF_SM);
    int *smId = (int *)malloc(sizeof(int) * NUM_OF_SM);
    for (int i = 0; i < NUM_OF_SM; i++)
    {
        smId[i] = i;
        pthread_create(finishTasks_t + i, NULL, finishGpuTasks, (void *)(smId + i));
    }
    gpuSchedule<<<NUM_OF_SM * BLOCKS_PER_SM, THREADS_PER_BLOCK, 0, streamKernel>>>(devTaskEntry, devNbExes, devTaskStartPerSm, devTaskEndPerSm, devTasksExit, EXECUTORS_PER_BLOCK);
    pthread_create(&getTasks_t, NULL, getGpuTasks, NULL);
    pthread_create(&dpdk_t, NULL, launchDpdk, (void *)&par);

    pthread_join(dpdk_t, NULL);
    printf("[dpdk] launchDpdk已经退出\n");
    pthread_join(getTasks_t, NULL);
    printf("[gpu schedule] getGpuTasks已经退出\n");
    tasksExit[0] = true;
    cudaDeviceSynchronize();
    printf("[gpu schedule] gpuSchedule已经退出\n");
    scheduleExit = true;
    for (int i = 0; i < NUM_OF_SM; i++)
    {
        pthread_join(finishTasks_t[i], NULL);
    }
    printf("[gpu schedule] finishGpuTasks已经退出\n");

    free(finishTasks_t);
    free(smId);
    freeGpuSchedule();
    return 0;
}
/*
在运行之前需要确保dpdk网卡enp2s0的灯是亮的
cd ~/private_shell;./server.sh 2
cd /home/adzhu/source/Gright/;sudo make clean
cd /home/adzhu/source/Gright/;sudo make;sudo ./gright -l 0-3
ping 172.27.237.118
sudo chown adzhu: -R /home/adzhu/source/Gright/;cd /home/adzhu/source/Gright/gpu_core/processing_results/
*/