#pragma once
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

#include "gpu_config.hpp"

extern uint32_t EXECUTORS_PER_SM;

/*任务表项*/

struct STaskEntry
{
    /*工作线程的编号*/
    uint32_t workId;

    /*数据包的数量*/
    uint32_t nbPackets;

    /*执行者的数量*/
    int nbExes;

    /*执行NF编号*/
    uint32_t nfId;

    /*输入数据包的指针*/
    uint8_t *pInPackets;

    /*输入数据包的指针（备份）*/
    uint8_t *pInPacketsBack;

    /*输入数据包的指针（pinned内存）*/
    uint8_t *pInPacketsHost;

    /*每个数据包的相对起始位置*/
    uint64_t *pPackets;

    /*device端处理结果*/
    uint32_t *pOutDev;

    /*host端处理结果*/
    uint32_t *pOutHost;

    /*每个处理结果的相对起始位置*/
    uint64_t *pResults;

    /*已经完成任务的executor个数*/
    uint16_t finishedExes;

    /*true表示该任务的数据包批处理已经完成，false表示该任务的数据包批处理没有完成或者为空*/
    bool isBatched;

    /*true表示该任务已经完成但没有保存，false表示该任务没有完成或者为空*/
    bool isSave;

    /*当前任务对应的批次*/
    int currBatch;

    /*当前任务对应的ptm*/
    int currPtm;

    /*开始传输数据包的时间戳*/
    struct timespec startTransfer;

    /*传输结束的时间戳*/
    struct timespec endTransfer;

    /*执行结束的时间戳*/
    struct timespec endExec;

    void init()
    {
        workId = 0;
        nbPackets = 0;
        nbExes = 0;
        nfId = 0;
        pInPackets = NULL;
        pInPacketsBack = NULL;
        pInPacketsHost = NULL;
        pPackets = NULL;
        pOutDev = NULL;
        pOutHost = NULL;
        pResults = NULL;
        finishedExes = 0;
        isBatched = false;
        isSave = false;
        currBatch = 0;
        currPtm = 0;
    }

    void setValue(int _batch, uint32_t _workId, uint32_t _nbPackets, uint32_t _nfId, int _ptm)
    {
        workId = _workId;
        nbPackets = _nbPackets;
        nbExes = (nbPackets % PACKETS_PER_EXECUTOR == 0) ? (nbPackets / PACKETS_PER_EXECUTOR) : (nbPackets / PACKETS_PER_EXECUTOR + 1);
#ifndef PACKET_TRANSFER_MODE
        if (nbExes > (int)EXECUTORS_PER_SM)
        {
            printf("[error] workId-%u-执行者的数量过大！\n", _workId);
            exit(1);
        }
#endif
        nfId = _nfId;
        finishedExes = 0;
        isBatched = false;
        isSave = false;
        currBatch = _batch;
        currPtm = _ptm;
    }
};