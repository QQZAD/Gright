#pragma once
#include <pthread.h>
#include <stdint.h>

#define DATA_COPY 0
#define ADDR_MAP 1

#define GET_NF_NAME(NF_ID) (NF_ID == 0 ? "ipv4_router" : (NF_ID == 1 ? "ipsec" : (NF_ID == 2 ? "ids" : "nat")))
#define GET_PTM_NAME(PTM_ID) (PTM_ID == 0 ? "DATA_COPY" : "ADRR_MAP")

/*指定批次的数量*/
#define BATCH 1

/*流的数量*/
#define FLOWS_NB 3 //对应sudo ./dpdk -l 0-3
// #define FLOWS_NB 1 //对应sudo ./dpdk -l 0

/*流的批处理大小*/
extern uint32_t flowsBatch[FLOWS_NB];

/*流的数据包指针*/
extern uint8_t *pFlowsPackets[FLOWS_NB];

/*流的PCIe传输顺序*/
extern int flowsOrder[FLOWS_NB];

/*流的NF类型*/
extern int flowsNf[FLOWS_NB];

/*流的数据包执行SM*/
extern int flowsSm[FLOWS_NB];

/*流访问数据包的方式*/
extern int flowsPtm[FLOWS_NB];

struct SPar
{
    int argc;
    char **argv;
    SPar(int _argc, char **_argv)
    {
        argc = _argc;
        argv = _argv;
    }
};

extern void readFlowsInfo(bool gpuCore);

extern void printFlowsInfo();