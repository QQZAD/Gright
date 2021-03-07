#pragma once
/*该宏定义用于对接到Gright*/
// #define GRIGHT

/*将printf的输出信息重定向到文件*/
// #define PRINTF_TO_FILE

/*在开始运行前暂停等待*/
#define PAUSE_WAIT

#define ALLOW_ASYN_ALLOC

/*保存处理结果*/
// #define SAVE_RESULTS

#ifdef PRINTF_TO_FILE
#define DEBUG_NF
#define DEBUG_EXIT
#endif

/*调试时的输出信息*/
#define DEBUG_GPU_SCHEDULE
#define DEBUG_EXECUTORS
#define DEBUG_DISPATCHER
#define DEBUG_SMID 0

/*调试数据包是否正确*/
#define ASSERT_PACKETS

/*NF的ID*/
#define IPV4_ROUTER_GPU 0
#define IPSEC_GPU 1
#define IDS_GPU 2
#define NAT_GPU 3

/*执行表的状态*/
#define EXEC_NOT_USE 0
#define EXEC_USING 1

/*gpu的warp的大小*/
#define DEV_WARP_SIZE 32

/*一个executor包含的线程数量*/
#define THREADS_PER_EXECUTOR 128

/*
一个executor同时处理数据包的数量
PACKETS_PER_EXECUTOR必须为THREADS_PER_EXECUTOR的整数倍
*/
#define PACKETS_PER_EXECUTOR 256

/*一个SM最大允许分配的任务数量*/
#define MAX_TASKS_PER_SM 10

/*每个线程块中executor的数量*/
#define DEV_EXECUTORS_PER_BLOCK 7
/*
一个SM最大能处理的数据包的数量为256*7=1792
*/