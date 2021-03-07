#pragma once
#include <stdio.h>
#include <assert.h>
#include <sys/stat.h>
#include <time.h>

#include "header/task_entry.hpp"
#include "packet/packet.hpp"
#include "../config/flows.hpp"

#define GPU_SCHEDULE
#define ADD(NB) (NB + 1)
#define NEXT_TASK_ID(CURR_TASK_ID, SMID, CUROSR) ((CURR_TASK_ID - SMID * ADD(MAX_TASKS_PER_SM) + CUROSR) % ADD(MAX_TASKS_PER_SM) + SMID * ADD(MAX_TASKS_PER_SM))
#define GET_TASK_NB(TASK_START, TASK_END) ((TASK_END >= TASK_START) ? (TASK_END - TASK_START) : (ADD(MAX_TASKS_PER_SM) + TASK_END - TASK_START))

extern uint32_t NUM_OF_SM;
extern uint32_t THREADS_PER_BLOCK;
extern uint32_t BLOCKS_PER_SM;
extern uint32_t EXECUTORS_PER_SM;
extern uint32_t EXECUTORS_PER_BLOCK;
extern cudaStream_t streamHd;
extern cudaStream_t streamKernel;
extern cudaError err;

extern struct STaskEntry *devTaskEntry;
extern uint32_t *devTaskStartPerSm;
extern uint32_t *devTaskEndPerSm;
extern int *devNbExes;
extern bool *devTasksExit;

#ifdef PRINTF_TO_FILE
extern int stdDup;
extern FILE *outLog;
#endif

extern void initDevicePara(uint32_t gpu);
/*对一定数量的数据包封装ESP头*/
extern int espEncap(uint8_t *in, uint32_t nbPackets, uint8_t *&out);

// extern double calTime(struct timespec start, struct timespec end);

extern struct timespec startCopy;
extern struct timespec endCopy;

/*任务队列*/

struct STaskTable
{
	/*任务表项的指针*/
	struct STaskEntry *pTaskEntry;

	/*任务表项的数量*/
	uint32_t nbTaskEntry;

	/*SM当前任务起点*/
	uint32_t *pTaskStartPerSm;

	/*SM当前任务终点*/
	uint32_t *pTaskEndPerSm;

	/*所有任务所需executors数量*/
	int *pNbExes;

	STaskTable()
	{
#ifdef PRINTF_TO_FILE
		struct stat st = {0};
		char fileName[25] = "terminal_info/debug.log";
		if (stat("terminal_info", &st) == -1)
		{
			mkdir("terminal_info", 0700);
		}
		else
		{
			remove(fileName);
		}
		printf("[gpu schedule] 所有printf的输出信息已经被重定向到");
		printf("%s\n", fileName);
		stdDup = dup(1);
		outLog = fopen(fileName, "a");
		dup2(fileno(outLog), 1);
#endif
		initDevicePara(0);
		nbTaskEntry = ADD(MAX_TASKS_PER_SM) * NUM_OF_SM;
		/*不能在构造函数中为pTaskEntry pTaskStartPerSm pTaskEndPerSm pNbExes四个指针申请pinned内存*/
	}

	int insertNewTask(int _batch, uint32_t _workId, uint32_t _nbPackets, uint32_t _nfId, int _ptm, uint32_t smId)
	{
		if (smId >= NUM_OF_SM)
		{
			printf("[error] workId-%u-SM的编号错误！\n", _workId);
			exit(1);
		}
		if (_nbPackets <= 0)
		{
			printf("[error] workId-%u-数据包的数量不合法！\n", _workId);
			exit(1);
		}
		if (_nbPackets > MAX_BATCH_SIZE)
		{
			printf("[error] 发现批处理大小大于MAX_BATCH_SIZE %d的流，对应的工作线程为%u\n", MAX_BATCH_SIZE, _workId);
			exit(1);
		}
		if (_nbPackets > EXECUTORS_PER_SM * PACKETS_PER_EXECUTOR)
		{
			printf("[error] workId-%u-数据包的数量过多以至于一个SM无法处理！\n", _workId);
			printf("解决方案\n");
			printf("1.通过minibatch将该任务拆分成多个子任务\n");
			printf("2.访问gpu_config.hpp将PACKETS_PER_EXECUTOR修改为更大的值\n");
			exit(1);
		}

		bool temp = false;
		while (NEXT_TASK_ID(pTaskEndPerSm[smId], smId, 1) == pTaskStartPerSm[smId])
		{
			if (temp == false)
			{
				/*修改MAX_TASKS_PER_SM来增大SM的任务缓冲区*/
				printf("[gpu schedule] SM-%d的任务缓冲区已满\n", smId);
				temp = true;
			}
		}

		uint32_t nowTaskPerSm = pTaskEndPerSm[smId];

		/*
		数据流的总长度
		这个数可能会很大
		*/
		uint64_t totalLen = 0;

		uint8_t *_pInPackets = pTaskEntry[nowTaskPerSm].pInPacketsHost;
		/*
		_pInPackets应该是多个数据包连接起来的一个大数组
		通过一个指针分配空间
		通过一个指针释放空间
		*/

#ifdef GRIGHT
		/*等待dpdk的数据包批处理完成*/
		while (pTaskEntry[nowTaskPerSm].isBatched == false)
		{
		}
		// _pInPackets = pTaskEntry[nowTaskPerSm].pInPacketsHost;
#else
		/*随机生成数据包*/
#ifdef DEBUG_GPU_SCHEDULE
		printf("[gpu schedule] workId-%u-taskIndex-%u传入的数据包指针为NULL，将随机生成数据包\n", _workId, nowTaskPerSm);
#endif

		CPacket *pac = new CPacket[_nbPackets];

		for (uint32_t i = 0; i < _nbPackets; i++)
		{
			totalLen += pac[i].bytesLen;
		}

		/*一批次数据包的总长度不能超过最大值*/
		assert(totalLen < MAX_BATCH_SIZE * MAX_PAC_LEN);

		// _pInPackets = (uint8_t *)malloc(totalLen * sizeof(uint8_t));

		uint8_t *p = _pInPackets;

		for (uint32_t i = 0; i < _nbPackets; i++)
		{
			uint16_t bytesLen = pac[i].bytesLen;
			if (bytesLen > MAX_PAC_LEN)
			{
				printf("[error] 发现字节长度大于MAX_PAC_LEN %d的数据包，其字节长度为%d\n", MAX_PAC_LEN, bytesLen);
				exit(1);
			}
			memcpy(p, pac[i].bytes(), sizeof(uint8_t) * bytesLen);
			p += bytesLen;
		}
		delete[] pac;
#endif

		/*执行ipsec的数据包需要先进行esp头的封装*/
		if (_nfId == IPSEC_GPU)
		{
			uint8_t *out;
			uint64_t result = espEncap(_pInPackets, _nbPackets, out);
			/*封装后的长度不能超过总长度*/
			assert(result < MAX_BATCH_SIZE * MAX_PAC_LEN);

			if (result == 1)
			{
				printf("[error] 当前数据包的3层协议不是ipv4！\n");
				exit(1);
			}
			else if (result == 2)
			{
				printf("[error] 在Hash表中找不到当前数据包对应的ESP封装结构！\n");
				exit(1);
			}
			else
			{
				memcpy(_pInPackets, out, result);
				free(out);
				// _pInPackets = out;
				totalLen = result;
			}
		}

#ifdef ASSERT_PACKETS
		/*检查一个批处理过程的数据包*/
		checkBatch(_pInPackets, _nbPackets);
#endif

		/*计算每个数据包的相对起始位置*/
		uint64_t *_pPackets = (uint64_t *)malloc(sizeof(uint64_t) * _nbPackets);
		_pPackets[0] = 0;

		for (uint32_t i = 0; i < _nbPackets - 1; i++)
		{
			struct ether_header *ethh = (struct ether_header *)(_pInPackets + _pPackets[i]);
			struct iphdr *iph = (struct iphdr *)(ethh + 1);
			uint32_t packetLen = sizeof(struct ether_header) + ntohs(iph->tot_len);
			_pPackets[i + 1] = _pPackets[i] + packetLen;
		}

		/*DPDK计算totalLen的值*/
		if (totalLen == 0)
		{
			totalLen = _pPackets[_nbPackets - 1];
			struct ether_header *ethh = (struct ether_header *)(_pInPackets + totalLen);
			struct iphdr *iph = (struct iphdr *)(ethh + 1);
			uint32_t packetLen = sizeof(struct ether_header) + ntohs(iph->tot_len);
			totalLen += packetLen;
		}

		/*为STaskEntry赋值*/
		pTaskEntry[nowTaskPerSm].setValue(_batch, _workId, _nbPackets, _nfId, _ptm);
		/*为pNbExes赋值*/
		pNbExes[nowTaskPerSm] = pTaskEntry[nowTaskPerSm].nbExes;

		/*PCIe数据包拷贝*/
		clock_gettime(CLOCK_MONOTONIC, &(pTaskEntry[nowTaskPerSm].startTransfer));
		if (flowsPtm[_workId] == DATA_COPY)
		{
			pTaskEntry[nowTaskPerSm].pInPackets = pTaskEntry[nowTaskPerSm].pInPacketsBack;
			if (_nfId == IPV4_ROUTER_GPU)
			{
				/*仅拷贝IP头*/
				uint8_t *pDes, *pSrc;
				for (int i = 0; i < _nbPackets; i++)
				{
					pDes = pTaskEntry[nowTaskPerSm].pInPackets + _pPackets[i] + sizeof(struct ether_header);
					pSrc = _pInPackets + _pPackets[i] + sizeof(struct ether_header);
					err = cudaMemcpyAsync(pDes, pSrc, sizeof(struct iphdr), cudaMemcpyHostToDevice, streamHd);
					if (err != 0)
					{
						printf("[cudaError] pInPackets-cudaMemcpyAsync返回0x%x\n", err);
						exit(1);
					}
				}
			}
			else
			{
				/*IDS IPsec 拷贝整个数据包*/
				err = cudaMemcpyAsync(pTaskEntry[nowTaskPerSm].pInPackets, _pInPackets, totalLen * sizeof(uint8_t), cudaMemcpyHostToDevice, streamHd);
				if (err != 0)
				{
					printf("[cudaError] pInPackets-cudaMemcpyAsync返回0x%x\n", err);
					exit(1);
				}
			}
			// cudaStreamSynchronize(streamHd);
			// clock_gettime(CLOCK_MONOTONIC, &endCopy);
			// printf("copyTime=%lf\n", calTime(startCopy, endCopy));
		}
		else if (flowsPtm[_workId] == ADDR_MAP)
		{
			cudaHostGetDevicePointer<uint8_t>(&(pTaskEntry[nowTaskPerSm].pInPackets), (void *)_pInPackets, 0);
		}

		err = cudaMemcpyAsync(pTaskEntry[nowTaskPerSm].pPackets, _pPackets, _nbPackets * sizeof(uint64_t), cudaMemcpyHostToDevice, streamHd);
		if (err != 0)
		{
			printf("[cudaError] pPackets-cudaMemcpyAsync返回0x%x\n", err);
			exit(1);
		}
		err = cudaMemcpyAsync(pTaskEntry[nowTaskPerSm].pOutDev, pTaskEntry[nowTaskPerSm].pOutHost, _nbPackets * sizeof(uint32_t), cudaMemcpyHostToDevice, streamHd);
		if (err != 0)
		{
			printf("[cudaError] pOutDev-cudaMemcpyAsync返回0x%x\n", err);
			exit(1);
		}
		/*等待cudaMemcpyAsync执行完毕*/
		cudaStreamSynchronize(streamHd);

#ifdef GRIGHT
		if (flowsPtm[_workId] == DATA_COPY)
		{
			/*标识任务缓冲区中的数据包已经被取走*/
			pTaskEntry[nowTaskPerSm].isBatched = false;
		}
#endif

#ifdef DEBUG_GPU_SCHEDULE
		printf("[gpu schedule] workId-%u-taskIndex-%u-nbPackets-%u-nfName-%s-pInPackets-%p-smId-%u-flowsPtm-%s\n", _workId, nowTaskPerSm, _nbPackets, GET_NF_NAME(_nfId), pTaskEntry[nowTaskPerSm].pInPackets, smId, GET_PTM_NAME(flowsPtm[_workId]));
#endif
		// free(_pInPackets);
		free(_pPackets);

		/*添加任务后更新游标*/
		pTaskEndPerSm[smId] = NEXT_TASK_ID(pTaskEndPerSm[smId], smId, 1);
		return 0;
	}
};

/*执行表项*/

struct SExecTable
{
	/*当前执行者的状态*/
	uint32_t status;

	/*当前任务在任务队列中的编号*/
	int taskIndex;

	/*工作线程的编号*/
	uint32_t workId;

	/*当前NF的编号*/
	uint32_t nfId;

	/*输入数据包的指针*/
	uint8_t *pInPackets;

	/*该executor需要处理的数据包个数*/
	uint32_t packetsNb;

	/*该executor需要跳过的数据包个数*/
	uint32_t skipPacketsNb;

	/*device端处理结果*/
	uint32_t *pOutDev;
};