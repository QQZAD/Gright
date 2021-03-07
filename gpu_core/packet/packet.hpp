#pragma once
#include <stdint.h>
#include <assert.h>

#include "../header/net_struct.hpp"

/*用于构建ipsec hash表的源地址和目的地址*/
#define SRC_ADDR 0x7f1b0001u
#define DEST_ADDR_NB 1024

/*单个数据包的最大字节长度*/
#define MAX_PAC_LEN 1500

/*batch size的最大值*/
#define MAX_BATCH_SIZE 3072

extern std::string u32ToIp(uint32_t ip);

/*此函数用于计算IP头的校验值*/
extern uint16_t getIpCheckSum(uint8_t *ptr, int size);

/*此函数用于计算TCP头的校验值*/
extern uint16_t getTcpCheckSum(int lenTcp, uint8_t *srcAddr, uint8_t *destAddr, bool padding, uint8_t *buff);

/*此函数用于检查一批次数据包是否正确*/
extern void checkBatch(uint8_t *inPackets, uint32_t batchSize);

/*自定义CPacket结构用于简单的测试*/
class CPacket
{
    /*
    内存的基本单位是一个字节
    每个字节的空间都对应唯一的地址
    网络字节顺序为__BIG_ENDIAN 地址的低位存储值的高位
    x86_64主机字节顺序为__LITTLE_ENDIAN 地址的低位存储值的低位
    uint8_t是单个字节，不需要转换顺序
    uint16_t主机字节转网络字节用函数htons
    uint16_t网络字节转主机字节用函数ntohs
    uint32_t主机字节转网络字节用函数htonl
    uint32_t网络字节转主机字节用函数ntohl
    目的硬件地址为ff:ff:ff:ff:ff:ff表示广播
    */
public:
    uint8_t *pBytes;
    uint16_t bytesLen;
    /*14字节*/
    struct ether_header eth;
    /*20字节*/
    struct iphdr iph;
    /*8字节*/
    struct udphdr uph;
    /*20字节*/
    struct tcphdr tph;
    uint8_t *pData;
    uint16_t dataLen;
    /*运输层协议 0 udp 1 tcp*/
    uint32_t tranLayPro;

    CPacket(uint32_t tlp = 1);

    ~CPacket();

    void setEtherHeader();

    void setIphdr();

    void setUdphdr();

    void setTcphdr();

    void generateData(uint16_t st, uint16_t ed);

    /*可输入数据包数据部分（20字节）*/
    void inputData(int _dataLen);

    /*通过tranBytesFlow函数可以将CPacket中的内容转换为uint8_t流*/
    void setBytesFlow(uint16_t _bytesLen, uint8_t *_pBytes);

    void tranBytesFlow();

    uint8_t *bytes();

    inline uint32_t u8tu32(uint8_t *data);

    void printInfo();
};