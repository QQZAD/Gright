#include <cuda_runtime.h>
#include <assert.h>
#include <string.h>

#include "gpu_schedule.hpp"
#include "gpu_tasks.hpp"
#include "packet/packet.hpp"
#include "ipv4_router/ipv4_router.cuh"
#include "ids/ids.cuh"
#include "ipsec/ipsec.cuh"
#include "nat/nat.cuh"

uint32_t NUM_OF_SM = 0;
uint32_t WARP_SIZE = 0;
uint32_t THREADS_PER_BLOCK = 0;
uint32_t BLOCKS_PER_SM = 0;
uint32_t EXECUTORS_PER_SM = 0;
uint32_t EXECUTORS_PER_BLOCK = 0;

/*主机端->设备端内存拷贝流*/
cudaStream_t streamHd;
/*设备端->主机端内存拷贝流*/
cudaStream_t streamDh;
/*内核执行流*/
cudaStream_t streamKernel;
/*CUDA异常处理*/
cudaError err;

struct STaskTable taskTable;
bool *tasksExit = NULL;
bool *devTasksExit = NULL;

struct STaskEntry *devTaskEntry;
uint32_t *devTaskStartPerSm;
uint32_t *devTaskEndPerSm;
int *devNbExes;

struct STaskEntry *dpdkTaskEntry;
uint32_t *dpdkTaskEndPerSm;

struct timespec startCopy;
struct timespec endCopy;

#ifdef PRINTF_TO_FILE
int stdDup;
FILE *outLog;
#endif

static char workDir[256];
static char fileDir[256];

double calTime(struct timespec start, struct timespec end);

void initDevicePara(uint32_t gpu)
{
	printf("[gpu schedule] 获取GPU %u的参数信息\n", gpu);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, gpu);
	NUM_OF_SM = deviceProp.multiProcessorCount;
	printf("NUM_OF_SM %u\n", NUM_OF_SM);
	WARP_SIZE = deviceProp.warpSize;
	printf("WARP_SIZE %u\n", WARP_SIZE);
	THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
	/*GeForce RTX 2080中BLOCKS_PER_SM的值为1*/
	BLOCKS_PER_SM = deviceProp.maxThreadsPerMultiProcessor / THREADS_PER_BLOCK;
	/*只考虑BLOCKS_PER_SM为1的情况*/
	assert(BLOCKS_PER_SM == 1);
	printf("BLOCKS_PER_SM %u\n", BLOCKS_PER_SM);
	EXECUTORS_PER_BLOCK = (THREADS_PER_BLOCK - WARP_SIZE) / THREADS_PER_EXECUTOR;
	printf("EXECUTORS_PER_BLOCK %u\n", EXECUTORS_PER_BLOCK);
	EXECUTORS_PER_SM = BLOCKS_PER_SM * EXECUTORS_PER_BLOCK;
	printf("EXECUTORS_PER_SM %u\n", EXECUTORS_PER_SM);
	THREADS_PER_BLOCK = EXECUTORS_PER_BLOCK * THREADS_PER_EXECUTOR + WARP_SIZE;
	printf("THREADS_PER_BLOCK %u\n", THREADS_PER_BLOCK);
	printf("每个线程块上共享内存的最大值 %zu bytes\n", deviceProp.sharedMemPerBlock);
	if (!deviceProp.deviceOverlap)
	{
		printf("该GPU不支持Device Overlap功能\n");
		exit(1);
	}
	else
	{
		printf("该GPU支持Device Overlap功能\n");
	}
	printf("THREADS_PER_EXECUTOR %u\n", THREADS_PER_EXECUTOR);
	printf("PACKETS_PER_EXECUTOR %u\n", PACKETS_PER_EXECUTOR);
	printf("MAX_TASKS_PER_SM %u\n\n", MAX_TASKS_PER_SM);
#ifdef SAVE_RESULTS
	printf("[gpu schedule] 本次运行将保存处理结果到processing_results\n\n");
#else
	printf("[gpu schedule] 本次运行不会保存处理结果\n\n");
#endif
}

static __device__ __inline__ uint32_t getSmid()
{
	uint32_t smId;
	asm volatile("mov.u32 %0, %%smid;"
				 : "=r"(smId));
	return smId;
}

__global__ void gpuSchedule(volatile struct STaskEntry *devTaskEntry, int *devNbExes, uint32_t *devTaskStartPerSm, uint32_t *devTaskEndPerSm, bool *devTasksExit, uint32_t EXECUTORS_PER_BLOCK)
{
	/*为了简化问题只考虑BLOCKS_PER_SM==1的情况*/
	/*即一个SM中最多只有一个线程块*/

	/*得到当前SM的编号*/
	uint32_t smId = getSmid();

	/*得到当前线程的编号*/
	uint32_t threadId = threadIdx.x;

	/*每个线程块中的共享内存*/
	/*共享数组变量的长度必须为const类型或常数*/
	__shared__ SExecTable execTable[DEV_EXECUTORS_PER_BLOCK];
	uint32_t initExecId = threadId;
	if (initExecId >= EXECUTORS_PER_BLOCK)
	{
		initExecId = EXECUTORS_PER_BLOCK - 1;
	}
	execTable[initExecId].status = EXEC_NOT_USE;
	execTable[initExecId].taskIndex = -1;
	execTable[initExecId].workId = 0;
	execTable[initExecId].nfId = 0;
	execTable[initExecId].pInPackets = NULL;
	execTable[initExecId].packetsNb = PACKETS_PER_EXECUTOR;
	execTable[initExecId].skipPacketsNb = 0;
	execTable[initExecId].pOutDev = NULL;

	/*每个线程块中任务数量的状态变量*/
	__shared__ uint32_t taskStatePerBlock[1];

	/*1表示该线程块中的任务数量需要更新*/
	taskStatePerBlock[0] = 0;

	/*每个线程块中executors的状态变量*/
	/*共享数组变量的长度必须为const类型或常数*/
	__shared__ uint32_t execEndPerBlock[DEV_EXECUTORS_PER_BLOCK];

	/*该值表示该executor中已经完成任务的线程数量*/
	execEndPerBlock[initExecId] = THREADS_PER_EXECUTOR;

	/*每个线程块中dispatcher线程是否退出*/
	/*共享数组变量的长度必须为const类型或常数*/
	__shared__ uint32_t dispaStatePerBlock[DEV_EXECUTORS_PER_BLOCK];

	/*1表示该dispatcher线程已退出*/
	dispaStatePerBlock[initExecId] = 0;

	/*共享内存的修改对块内所有线程可见*/
	__threadfence_block();

	if (threadId < EXECUTORS_PER_BLOCK)
	{
		uint32_t tasksPerSm = GET_TASK_NB(devTaskStartPerSm[smId], devTaskEndPerSm[smId]);
		if (tasksPerSm != 0)
		{
			/*输出全局内存中该SM的任务数量*/
#ifdef DEBUG_DISPATCHER
			if (threadId == 0)
			{
				printf("[dspatcher] threadId-%u-TasksPerSm[%u]-%u\n", threadId, smId, tasksPerSm);
			}
#endif
		}

		/*dispatcher线程的循环周期*/
		while (1)
		{
			/*若dispatcher线程的执行表状态为EXEC_NOT_USE则从全局内存获取任务*/
			if (execTable[threadId].status == EXEC_NOT_USE)
			{
				/*若全局内存中存在该SM的任务则继续*/
				if (GET_TASK_NB(devTaskStartPerSm[smId], devTaskEndPerSm[smId]) > 0)
				{
#ifdef ALLOW_ASYN_ALLOC
					uint32_t taskIndex = devTaskStartPerSm[smId];

					/*初始化每个线程块中任务数量的状态变量*/
					if (taskStatePerBlock[0] == 0)
					{
						taskStatePerBlock[0] = 1;
					}

					/*任务所需的executors数量必须为正数且不超过该线程块中executors数量的上限*/
					if (devTaskEntry[taskIndex].nbExes <= EXECUTORS_PER_BLOCK)
					{
						/*若该dispatcher线程获取了一个任务则为其executor分配任务*/
						/*每次成功的任务获取都会导致该任务所需的executors数量减1直到为0*/
						int oldNbExes = atomicSub((int *)(&(devTaskEntry[taskIndex].nbExes)), 1);
						if (oldNbExes > 0)
						{
							/*每个dispatcher线程根据所获取的任务设定自己的执行表*/
							execTable[threadId].taskIndex = taskIndex;
							execTable[threadId].nfId = devTaskEntry[taskIndex].nfId;
							execTable[threadId].workId = devTaskEntry[taskIndex].workId;
							execTable[threadId].pInPackets = devTaskEntry[taskIndex].pInPackets;
							execTable[threadId].pOutDev = devTaskEntry[taskIndex].pOutDev;
							/*第一个获取该任务的dispatcher线程需要确定分配到的数据包的个数*/
							if (oldNbExes == devNbExes[taskIndex])
							{
								uint32_t temp = (devTaskEntry[taskIndex].nbPackets) % PACKETS_PER_EXECUTOR;
								if (temp != 0)
								{
									execTable[threadId].packetsNb = temp;
								}
							}
							execTable[threadId].skipPacketsNb = (oldNbExes - 1) * PACKETS_PER_EXECUTOR;

							/*状态赋值语句必须在最后以保证executors线程读取正确的值*/
							execTable[threadId].status = EXEC_USING;
#ifdef DEBUG_DISPATCHER
							printf("[dspatcher] smId-%u-threadId-%u获得workId-%u-taskIndex-%u-nf-%s\n",
								   smId, threadId, execTable[threadId].workId, taskIndex, GET_NF_NAME(execTable[threadId].nfId));
#endif
						}
					}

					/*当任务所需的executors数量为0时表示该任务已全部分配完成*/
					/*每个线程块中任务数量的状态变量为1表示需要更新*/
					/*若已有dispatcher线程对任务数量的状态变量进行更新则阻止其他dispatcher线程重复修改*/
					// if (devTaskEntry[taskIndex].nbExes <= 0 && atomicCAS(taskStatePerBlock, 1, 0) == 1)
					if (devTaskEntry[devTaskStartPerSm[smId]].nbExes <= 0 && atomicCAS(taskStatePerBlock, 1, 0) == 1)
					{
						devTaskEntry[taskIndex].nbExes = 0;
#ifdef BACKUP
#ifdef DEBUG_DISPATCHER
						printf("[dspatcher] threadId-%u-TasksPerSm[%u]-%u\n", threadId, smId, GET_TASK_NB(devTaskStartPerSm[smId], devTaskEndPerSm[smId]));
#endif
						/*全局内存中的任务数量减1*/
						devTaskStartPerSm[smId] = NEXT_TASK_ID(devTaskStartPerSm[smId], smId, 1);

						/*全局内存的修改对所有线程可见*/
						__threadfence();
#ifdef DEBUG_DISPATCHER
						printf("[dspatcher] threadId-%u-TasksPerSm[%u]-%u\n", threadId, smId, GET_TASK_NB(devTaskStartPerSm[smId], devTaskEndPerSm[smId]));
#endif
#endif
					}

#else
					/*计算该SM所有任务的起止编号*/
					uint32_t minIndex = devTaskStartPerSm[smId];
					uint32_t maxIndex = devTaskEndPerSm[smId];
					uint32_t taskIndex = minIndex;
					uint32_t nbExes = 0;
					uint32_t nbTask = 0;
					bool isExceed = false;

					/*确定该次调度的任务总数*/
					for (uint32_t i = minIndex; i != maxIndex; i = NEXT_TASK_ID(i, smId, 1))
					{
						nbExes += devTaskEntry[i].nbExes;
						if (nbExes <= EXECUTORS_PER_BLOCK)
						{
							nbTask = i - minIndex + 1;
							taskStatePerBlock[0] = 1;
						}
						else
						{
							break;
						}
					}

#ifdef DEBUG_DISPATCHER
					if (threadId == 0)
					{
						printf("[dspatcher] smId-%u-nbTask-%u\n", smId, nbTask);
					}
#endif

					nbExes = 0;

					/*确定每个dispatcher线程对应的任务编号taskIndex*/
					for (uint32_t i = minIndex; i != maxIndex; i = NEXT_TASK_ID(i, smId, 1))
					{
						nbExes += devTaskEntry[i].nbExes;
						if (EXECUTORS_PER_BLOCK < nbExes)
						{
							isExceed = true;
							break;
						}
						else
						{
							if (threadId < nbExes)
							{
								break;
							}
							taskIndex++;
						}
					}

					/*若SM满足任务的资源需求则将该任务分配给对应的dispatcher线程*/
					if (!isExceed && threadId < nbExes)
					{
						/*每个dispatcher线程根据所获取的任务设定自己的执行表*/
						execTable[threadId].nfId = devTaskEntry[taskIndex].nfId;
						execTable[threadId].workId = devTaskEntry[taskIndex].workId;
						execTable[threadId].pInPackets = devTaskEntry[taskIndex].pInPackets;
						execTable[threadId].pOutDev = devTaskEntry[taskIndex].pOutDev;

						/*状态赋值语句必须在最后以保证executors线程读取正确的值*/
						execTable[threadId].status = EXEC_USING;

						/*共享内存的修改对块内所有线程可见*/
						__threadfence_block();
#ifdef DEBUG_DISPATCHER
						printf("[dspatcher] smId-%u-threadId-%u获得workId-%u-taskIndex-%u-nf-%s\n",
							   smId, threadId, execTable[threadId].workId, taskIndex, GET_NF_NAME(execTable[threadId].nfId));
#endif

#ifdef DEBUG_DISPATCHER
						/*该任务对应的最后一个dispatcher线程释放该任务*/
						if (threadId == nbExes - 1)
						{
							printf("[dspatcher] smId-%u-threadId-%u释放workId-%u-taskIndex-%u-nf-%s\n",
								   smId, threadId, execTable[threadId].workId, taskIndex, GET_NF_NAME(execTable[threadId].nfId));
						}
#endif

						/*第一个dispatcher线程一定参与*/
						/*该线程块中第一个dispatcher线程更新全局内存中的任务队列*/
						if (threadId == 0)
						{
							devTaskStartPerSm[smId] = NEXT_TASK_ID(devTaskStartPerSm[smId], smId, nbTask);

							/*全局内存的修改对所有线程可见*/
							__threadfence();

							taskStatePerBlock[0] = 0;

							/*共享内存的修改对块内所有线程可见*/
							__threadfence_block();
						}

						while (taskStatePerBlock[0] == 1)
						{
						}
					}
#endif
				}
			}

			if (devTasksExit[0] == true)
			{
				if (GET_TASK_NB(devTaskStartPerSm[smId], devTaskEndPerSm[smId]) == 0)
				{
#ifdef DEBUG_EXIT
					if (smId == DEBUG_SMID)
					{
						printf("[dspatcher] smId-%u-threadId-%u退出\n", smId, threadId);
					}
#endif
					/*该SM上的全部任务已分配给executors*/
					/*该dispatcher线程应该退出*/
					dispaStatePerBlock[threadId] = 1;
					break;
				}
			}
		}
	}
	else if (threadId >= DEV_WARP_SIZE)
	{
		/*排除dispatcher线程后该线程的编号*/
		uint32_t _threadId = threadId - DEV_WARP_SIZE;

		/*该线程所属executor的编号*/
		uint32_t execId = _threadId / THREADS_PER_EXECUTOR;

		/*executor中线程的编号*/
		uint32_t exeThread = _threadId % THREADS_PER_EXECUTOR;

		/*executors线程的循环周期*/
		while (1)
		{

			if (execTable[execId].status == EXEC_USING)
			{
				/*一个线程要处理的数据包个数*/
				uint32_t pacPerTh = PACKETS_PER_EXECUTOR / THREADS_PER_EXECUTOR;

				/*数据包数量开始变得不正常的线程*/
				uint32_t keyThread = (execTable[execId].packetsNb) / pacPerTh;

				/*当前线程要处理的数据包个数*/
				uint32_t packetsPerTh;

				if (exeThread == keyThread)
				{
					packetsPerTh = (execTable[execId].packetsNb) % pacPerTh;
				}
				else if (exeThread > keyThread)
				{
					packetsPerTh = 0;
				}
				else
				{
					packetsPerTh = pacPerTh;
				}

				int taskIndex = execTable[execId].taskIndex;

				if (packetsPerTh != 0)
				{
					uint32_t skipPacketsNb = execTable[execId].skipPacketsNb;
					skipPacketsNb += pacPerTh * exeThread;
					uint64_t cursor = (devTaskEntry[taskIndex].pPackets)[skipPacketsNb];
					uint8_t *inPackets = execTable[execId].pInPackets + cursor;
					uint32_t *outPackets = execTable[execId].pOutDev + skipPacketsNb;

					if (execTable[execId].nfId == IPV4_ROUTER_GPU)
					{
						gpuIpv4Router(packetsPerTh, inPackets, outPackets);
#ifdef DEBUG_NF
						printf("[executors] smId-%u-executors-%u-threadId-%u-workId-%u-taskIndex-%u-packetsPerTh-%u-ipv4_router...\n", smId, execId, exeThread, execTable[execId].workId, taskIndex, packetsPerTh);
#endif
					}
					else if (execTable[execId].nfId == IPSEC_GPU)
					{
						gpuIpsec(packetsPerTh, inPackets, outPackets);
#ifdef DEBUG_NF
						printf("[executors] smId-%u-executors-%u-threadId-%u-workId-%u-taskIndex-%u-packetsPerTh-%u-ipsec...\n", smId, execId, exeThread, execTable[execId].workId, taskIndex, packetsPerTh);
#endif
					}
					else if (execTable[execId].nfId == IDS_GPU)
					{
						gpuIds(packetsPerTh, inPackets, outPackets);
#ifdef DEBUG_NF
						printf("[executors] smId-%u-executors-%u-threadId-%u-workId-%u-taskIndex-%u-packetsPerTh-%u-ids...\n", smId, execId, exeThread, execTable[execId].workId, taskIndex, packetsPerTh);
#endif
					}
					else if (execTable[execId].nfId == NAT_GPU)
					{
						gpuNat(packetsPerTh, inPackets, outPackets);
#ifdef DEBUG_NF
						printf("[executors] smId-%u-executors-%u-threadId-%u-workId-%u-taskIndex-%u-packetsPerTh-%u-nat...\n", smId, execId, exeThread, execTable[execId].workId, taskIndex, packetsPerTh);
#endif
					}
				}

				/*选择executors中第一个线程*/
				if (exeThread == 0)
				{
					uint32_t old = atomicAdd((uint32_t *)(&(devTaskEntry[taskIndex].finishedExes)), uint32_t(1));
					/*最后完成任务的executor输出信息*/
					if (devTaskEntry[taskIndex].finishedExes == devNbExes[taskIndex] && old + 1 == devNbExes[taskIndex])
					{
#ifdef DEBUG_EXECUTORS
						printf("[executors] smId-%u-threadId-%u-workId-%u-taskIndex-%u-finishedExes-%u完成\n", smId, threadId, execTable[execId].workId, taskIndex, devTaskEntry[taskIndex].finishedExes);
#endif

#ifdef GRIGHT
						devTaskEntry[taskIndex].isBatched = false;
#endif
						devTaskEntry[taskIndex].isSave = true;
					}
				}

				/*executors中最快的线程初始化executors的状态变量为0*/
				atomicCAS((execEndPerBlock + execId), THREADS_PER_EXECUTOR, 0);

				/*executors中线程完成任务后更新executors的状态变量*/
				uint32_t execEnd = atomicAdd((execEndPerBlock + execId), uint32_t(1));

				/*等待executors中所有线程全部完成任务*/
				/*同一个executors内部的所有线程进行同步*/
				if (execEnd != THREADS_PER_EXECUTOR - 1)
				{
					while (execEndPerBlock[execId] != THREADS_PER_EXECUTOR)
					{
					}
				}

				/*在进入下次循环之前更新执行表状态*/
				if (execTable[execId].status == EXEC_USING)
				{
					execTable[execId].status = EXEC_NOT_USE;
				}
			}

			/*若该executors线程对应的dispatcher线程已退出且其执行表为空*/
			if (dispaStatePerBlock[execId] == 1 && execTable[execId].status == EXEC_NOT_USE)
			{
#ifdef DEBUG_EXIT
				if (smId == DEBUG_SMID)
				{
					printf("[executors] 退出smId-%u-executors-%u-threadId-%u\n", smId, execId, threadId);
				}
#endif
				/*该executor线程应该退出*/
				break;
			}
		}
	}
}

/*检查工作目录*/
void checkWorkDir()
{
	getcwd(workDir, sizeof(workDir));
	int workLen = strlen(workDir);
	printf("[gpu schedule] 当前工作目录%s\n", workDir);

	memcpy(fileDir, __FILE__, sizeof(__FILE__));
	int fileLen = strlen(fileDir);
	bool isDir = false;
	for (int i = fileLen - 1; i >= 0; i--)
	{
		if (fileDir[i] == '/')
		{
			fileDir[i] = '\0';
			isDir = true;
			break;
		}
	}
	if (isDir == true)
	{
		fileLen = strlen(fileDir);
		memcpy(fileDir + workLen + 1, fileDir, fileLen + 1);
		fileDir[workLen] = '/';
	}
	memcpy(fileDir, workDir, workLen);
	printf("[gpu schedule] 当前文件目录%s\n", fileDir);

	if (strlen(fileDir) != strlen(workDir))
	{
		printf("[gpu schedule] 切换工作目录\n");
		chdir(fileDir);
	}
}

/*恢复工作目录*/
void backWorkDir()
{
	if (strlen(fileDir) != strlen(workDir))
	{
		printf("[gpu schedule] 恢复工作目录\n");
		chdir(workDir);
	}
}

/*检查初始参数*/
static void checkInitPar()
{
	if (WARP_SIZE != DEV_WARP_SIZE)
	{
		printf("[gpu schedule error] WARP_SIZE 和 DEV_WARP_SIZE 的值不相等");
		printf("，访问gpu_config.hpp修改 DEV_WARP_SIZE 的值\n");
		exit(1);
	}
	if (THREADS_PER_EXECUTOR != 128)
	{
		printf("[gpu schedule error] 建议将 THREADS_PER_EXECUTOR 的值设置为128");
		printf("，访问gpu_config.hpp修改 THREADS_PER_EXECUTOR 的值\n");
		exit(1);
	}
	if (PACKETS_PER_EXECUTOR % THREADS_PER_EXECUTOR != 0 || PACKETS_PER_EXECUTOR < THREADS_PER_EXECUTOR)
	{
		printf("[gpu schedule error] PACKETS_PER_EXECUTOR 不是 THREADS_PER_EXECUTOR 的整数倍");
		printf("，访问gpu_config.hpp修改 PACKETS_PER_EXECUTOR 的值\n");
		exit(1);
	}
	if (EXECUTORS_PER_BLOCK != DEV_EXECUTORS_PER_BLOCK)
	{
		printf("[gpu schedule error] EXECUTORS_PER_BLOCK和DEV_EXECUTORS_PER_BLOCK的值不相等");
		printf("，访问gpu_config.hpp修改DEV_EXECUTORS_PER_BLOCK的值\n");
		exit(1);
	}
}

void initGpuSchedule()
{
	checkWorkDir();
	checkInitPar();

	printf("\n");

	initIpv4Router("ipv4_router/routing_info.txt");
	initIpsec();
	initIds("ids/ac_rule.txt");
	initNat();

	printf("\n");

	cudaMallocHost((void **)&taskTable.pTaskEntry, taskTable.nbTaskEntry * sizeof(struct STaskEntry), cudaHostAllocMapped);

	uint32_t bytesInPackets = MAX_BATCH_SIZE * MAX_PAC_LEN * sizeof(uint8_t);
	int bytesPackets = MAX_BATCH_SIZE * sizeof(uint64_t);
	int bytesOutDev = MAX_BATCH_SIZE * sizeof(uint32_t);

	uint8_t *initInPackets = (uint8_t *)malloc(bytesInPackets);
	uint64_t *initPackets = (uint64_t *)malloc(bytesPackets);
	uint32_t *initOutDev = (uint32_t *)malloc(bytesOutDev);

	memset(initInPackets, 0, bytesInPackets);
	memset(initPackets, 0, bytesPackets);
	memset(initOutDev, 0, bytesOutDev);

	for (uint32_t i = 0; i < taskTable.nbTaskEntry; i++)
	{
		taskTable.pTaskEntry[i].init();
		err = cudaMalloc((void **)&(taskTable.pTaskEntry[i].pInPackets), bytesInPackets);
		if (err != 0)
		{
			printf("[cudaError] cudaMalloc返回0x%x\n", err);
			exit(1);
		}
		taskTable.pTaskEntry[i].pInPacketsBack = taskTable.pTaskEntry[i].pInPackets;
		err = cudaMemcpyAsync(taskTable.pTaskEntry[i].pInPackets, initInPackets, bytesInPackets, cudaMemcpyHostToDevice, streamHd);
		if (err != 0)
		{
			printf("[cudaError] pInPackets-cudaMemcpyAsync返回0x%x\n", err);
			exit(1);
		}
		err = cudaMallocHost((void **)&(taskTable.pTaskEntry[i].pInPacketsHost), bytesInPackets, cudaHostAllocMapped);
		if (err != 0)
		{
			printf("[cudaError] cudaMallocHost返回0x%x\n", err);
			exit(1);
		}
		memset(taskTable.pTaskEntry[i].pInPacketsHost, 0, bytesInPackets);

		err = cudaMalloc((void **)&(taskTable.pTaskEntry[i].pPackets), bytesPackets);
		if (err != 0)
		{
			printf("[cudaError] cudaMalloc返回0x%x\n", err);
			exit(1);
		}
		err = cudaMemcpyAsync(taskTable.pTaskEntry[i].pPackets, initPackets, bytesPackets, cudaMemcpyHostToDevice, streamHd);
		if (err != 0)
		{
			printf("[cudaError] pPackets-cudaMemcpyAsync返回0x%x\n", err);
			exit(1);
		}

		err = cudaMalloc((void **)&(taskTable.pTaskEntry[i].pOutDev), bytesOutDev);
		if (err != 0)
		{
			printf("[cudaError] cudaMalloc返回0x%x\n", err);
			exit(1);
		}
		err = cudaMemcpyAsync(taskTable.pTaskEntry[i].pOutDev, initOutDev, bytesOutDev, cudaMemcpyHostToDevice, streamHd);
		if (err != 0)
		{
			printf("[cudaError] pOutDev-cudaMemcpyAsync返回0x%x\n", err);
			exit(1);
		}

		taskTable.pTaskEntry[i].pOutHost = (uint32_t *)malloc(MAX_BATCH_SIZE * sizeof(uint32_t));
		memset(taskTable.pTaskEntry[i].pOutHost, 0, MAX_BATCH_SIZE * sizeof(uint32_t));
	}

	free(initInPackets);
	free(initPackets);
	free(initOutDev);

	cudaMallocHost((void **)&taskTable.pTaskStartPerSm, NUM_OF_SM * sizeof(uint32_t), cudaHostAllocMapped);
	cudaMallocHost((void **)&taskTable.pTaskEndPerSm, NUM_OF_SM * sizeof(uint32_t), cudaHostAllocMapped);
	for (uint32_t i = 0; i < NUM_OF_SM; i++)
	{
		taskTable.pTaskStartPerSm[i] = taskTable.pTaskEndPerSm[i] = i * ADD(MAX_TASKS_PER_SM);
	}
	cudaMallocHost((void **)&taskTable.pNbExes, taskTable.nbTaskEntry * sizeof(int), cudaHostAllocMapped);
	for (uint32_t i = 0; i < taskTable.nbTaskEntry; i++)
	{
		taskTable.pNbExes[i] = taskTable.pTaskEntry[i].nbExes;
	}

	cudaStreamCreate(&streamHd);
	cudaStreamCreate(&streamDh);
	cudaStreamCreate(&streamKernel);

	cudaMallocHost((void **)&tasksExit, sizeof(bool), cudaHostAllocMapped);
	tasksExit[0] = false;

	cudaHostGetDevicePointer<struct STaskEntry>(&devTaskEntry, (void *)taskTable.pTaskEntry, 0);
	cudaHostGetDevicePointer<uint32_t>(&devTaskStartPerSm, (void *)taskTable.pTaskStartPerSm, 0);
	cudaHostGetDevicePointer<uint32_t>(&devTaskEndPerSm, (void *)taskTable.pTaskEndPerSm, 0);
	cudaHostGetDevicePointer<int>(&devNbExes, (void *)taskTable.pNbExes, 0);
	cudaHostGetDevicePointer<bool>(&devTasksExit, (void *)tasksExit, 0);

	dpdkTaskEntry = taskTable.pTaskEntry;
	dpdkTaskEndPerSm = taskTable.pTaskEndPerSm;
}

void freeGpuSchedule()
{
	for (uint32_t i = 0; i < taskTable.nbTaskEntry; i++)
	{
		cudaFree(taskTable.pTaskEntry[i].pInPackets);
		cudaFreeHost(taskTable.pTaskEntry[i].pInPacketsHost);
		cudaFree(taskTable.pTaskEntry[i].pPackets);
		cudaFree(taskTable.pTaskEntry[i].pOutDev);
		free(taskTable.pTaskEntry[i].pOutHost);
	}

	cudaStreamDestroy(streamHd);
	cudaStreamDestroy(streamDh);
	cudaStreamDestroy(streamKernel);

	cudaFreeHost(tasksExit);
	cudaFreeHost(taskTable.pTaskEntry);
	cudaFreeHost(taskTable.pTaskStartPerSm);
	cudaFreeHost(taskTable.pTaskEndPerSm);
	cudaFreeHost(taskTable.pNbExes);

	freeIpv4Router();
	freeIpsec();
	freeIds();
	freeNat();

#ifdef PRINTF_TO_FILE
	fflush(stdout);
	fclose(outLog);
	dup2(stdDup, 1);
	close(stdDup);
#endif

	backWorkDir();
}

void *launchGpuSchedule(void *argc)
{
	readFlowsInfo(true);
	printFlowsInfo();
	initGpuSchedule();

#ifdef PAUSE_WAIT
	printf("[gpu schedule] 按回车键继续...\n");
	system("read REPLY");
#endif

	struct timespec startGet;
	struct timespec endGet;
	pthread_t getTasks_t;
	struct timespec startFin;
	struct timespec endFin;
	pthread_t *finishTasks_t = (pthread_t *)malloc(sizeof(pthread_t) * NUM_OF_SM);
	int *smId = (int *)malloc(sizeof(int) * NUM_OF_SM);
	clock_gettime(CLOCK_MONOTONIC, &startFin);
	for (int i = 0; i < NUM_OF_SM; i++)
	{
		smId[i] = i;
		pthread_create(finishTasks_t + i, NULL, finishGpuTasks, (void *)(smId + i));
	}
	struct timespec startGpu;
	struct timespec endGpu;
	clock_gettime(CLOCK_MONOTONIC, &startGpu);
	gpuSchedule<<<NUM_OF_SM * BLOCKS_PER_SM, THREADS_PER_BLOCK, 0, streamKernel>>>(devTaskEntry, devNbExes, devTaskStartPerSm, devTaskEndPerSm, devTasksExit, EXECUTORS_PER_BLOCK);
	clock_gettime(CLOCK_MONOTONIC, &startGet);
	pthread_create(&getTasks_t, NULL, getGpuTasks, NULL);

	pthread_join(getTasks_t, NULL);
	clock_gettime(CLOCK_MONOTONIC, &endGet);
	printf("[gpu schedule] getGpuTasks已经退出 time=%lf\n", calTime(startGet, endGet));
	tasksExit[0] = true;
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &endGpu);
	printf("[gpu schedule] gpuSchedule已经退出 time=%lf\n", calTime(startGpu, endGpu));
	scheduleExit = true;
	for (int i = 0; i < NUM_OF_SM; i++)
	{
		pthread_join(finishTasks_t[i], NULL);
	}
	clock_gettime(CLOCK_MONOTONIC, &endFin);
	printf("[gpu schedule] finishGpuTasks已经退出 time=%lf\n", calTime(startFin, endFin));

	free(finishTasks_t);
	free(smId);
	freeGpuSchedule();
	return NULL;
}

#ifndef GRIGHT
int main()
{
	launchGpuSchedule(NULL);
	return 0;
}
#endif

/*
cd ~/private_shell;./server.sh 2
cd /home/adzhu/source/Gright/gpu_core;sudo make;sudo ./gpu_core
ping 172.27.220.149
sudo chown adzhu: -R /home/adzhu/source/Gright/;cd /home/adzhu/source/Gright/gpu_core/processing_results/
*/
