#include <stdio.h>
#include <string.h>
#include <iomanip>
#include <stdlib.h>
#include <math.h>

#include "packet.hpp"

static uint8_t *acRule[5] = {
    (uint8_t *)"alert tcp $EXTERNAL_NET any -> $HOME_NET 53 ( msg:\" PROTOCOL - DNS named version attempt \"; flow:to_server,established; content:\" | 07 | version \",offset 12,nocase; content:\" | 04 | bind | 00 | \",offset 12,nocase; metadata:ruleset community; service:dns; reference:nessus,10028; classtype:attempted-recon; sid:257; rev:17; )",
    (uint8_t *)"alert udp $EXTERNAL_NET any -> $HOME_NET 20433 ( msg:\"MALWARE-OTHER shaft agent to handler\"; flow:to_server; content:\"alive\"; metadata:ruleset community; reference:cve,2000-0138; classtype:attempted-dos; sid:240; rev:10; )",
    (uint8_t *)"alert udp $EXTERNAL_NET 3345 -> $HOME_NET 3344 ( msg:\"MALWARE-BACKDOOR Matrix 2.0 Server access\"; flow:to_server; content:\"logged in\"; metadata:ruleset community; classtype:misc-activity; sid:162; rev:10; )",
    (uint8_t *)"alert tcp $EXTERNAL_NET any -> $HOME_NET 53 ( msg:\"OS-SOLARIS EXPLOIT sparc overflow attempt\"; flow:to_server,established; content:\"|90 1A C0 0F 90 02| |08 92 02| |0F D0 23 BF F8|\",fast_pattern,nocase; metadata:ruleset community; service:dns; classtype:attempted-admin; sid:267; rev:13; )",
    (uint8_t *)"alert udp $EXTERNAL_NET any -> $HOME_NET 635 ( msg:\"OS-LINUX x86 Linux mountd overflow\"; flow:to_server; content:\"|EB|V^VVV1|D2 88|V|0B 88|V|1E|\"; metadata:ruleset community; reference:bugtraq,121; reference:cve,1999-0002; classtype:attempted-admin; sid:316; rev:10; )"};

static void randHost(uint8_t *_dh, uint8_t *_sh)
{
    for (int i = 0; i < 6; i++)
    {
        int buf1, buf2;
        buf1 = rand() % 256;
        buf2 = rand() % 256;
        _dh[i] = uint8_t(buf1);
        _sh[i] = uint8_t(buf2);
    }
    /*将目的硬件设置为单播*/
    _dh[0] = 0x00;
}

static uint32_t ipToU32(std::string ip)
{
    char *result8 = (char *)malloc(sizeof(char) * 8);
    char *p = result8;
    char temp[3];
    temp[2] = '\0';
    char *ips = strtok((char *)ip.c_str(), ".");
    while (ips != NULL)
    {
        sprintf(temp, "%02x", atoi(ips));
        (*p++) = temp[0];
        (*p++) = temp[1];
        ips = strtok(NULL, ".");
    }
    *p = '\0';
    uint32_t result = 0;
    sscanf(result8, "%x", &result);
    free(result8);
    return result;
}

std::string u32ToIp(uint32_t ip)
{
    std::string result = "";
    uint8_t ips[4];
    uint32_t old_ip, new_ip;
    old_ip = ip;
    for (int i = 0; i < 4; i++)
    {
        if (i == 0)
        {
            new_ip = old_ip;
        }
        else
        {
            new_ip = old_ip - ips[i - 1] * pow(16, 8 - 2 * i);
        }
        ips[i] = new_ip / pow(16, 6 - 2 * i);
        result += std::to_string(int(ips[i]));
        if (i != 3)
        {
            result += '.';
        }
        old_ip = new_ip;
    }
    return result;
}

/*此函数用于计算IP头的校验值*/
uint16_t getIpCheckSum(uint8_t *ptr, int size)
{
    int cksum = 0;
    int index = 0;
    *(ptr + 10) = 0;
    *(ptr + 11) = 0;
    if (size % 2 != 0)
        return 0;
    while (index < size)
    {
        cksum += *(ptr + index + 1);
        cksum += *(ptr + index) << 8;
        index += 2;
    }
    while (cksum > 0xffff)
    {
        cksum = (cksum >> 16) + (cksum & 0xffff);
    }
    return ~cksum;
}

/*此函数用于计算TCP头的校验值*/
uint16_t getTcpCheckSum(int lenTcp, uint8_t *srcAddr, uint8_t *destAddr, bool padding, uint8_t *buff)
{
    uint16_t prot_tcp = 6;
    uint16_t padd = 0;
    uint16_t word16;
    uint32_t sum;

    /*找出数据的长度是偶数还是奇数。如果是奇数，在包的末尾添加填充字节= 0*/
    if ((padding & 1) == 1)
    {
        padd = 1;
        buff[lenTcp] = 0;
    }

    sum = 0;
    int i;

    /*由每两个相邻的8位字组成16位字，并计算所有16个vit字的和*/
    for (i = 0; i < lenTcp + padd; i = i + 2)
    {
        word16 = ((buff[i] << 8) & 0xFF00) + (buff[i + 1] & 0xFF);
        sum = sum + (unsigned long)word16;
    }

    /*添加TCP伪标头，其中包含:IP源地址和目的地址*/
    for (i = 0; i < 4; i = i + 2)
    {
        word16 = ((srcAddr[i] << 8) & 0xFF00) + (srcAddr[i + 1] & 0xFF);
        sum = sum + word16;
    }
    for (i = 0; i < 4; i = i + 2)
    {
        word16 = ((destAddr[i] << 8) & 0xFF00) + (destAddr[i + 1] & 0xFF);
        sum = sum + word16;
    }

    /*协议编号和TCP包的长度*/
    sum = sum + prot_tcp + lenTcp;

    /*只保留32位计算和的最后16位并添加进位*/
    while (sum >> 16)
        sum = (sum & 0xFFFF) + (sum >> 16);

    sum = ~sum;

    return ((unsigned short)sum);
}

/*此函数用于检查一批次数据包是否正确*/
void checkBatch(uint8_t *inPackets, uint32_t batchSize)
{
    uint8_t *p = inPackets;
    struct ether_header *eth_hdr;
    struct iphdr *iph_hdr;

    for (int i = 0; i < batchSize; i++)
    {
        eth_hdr = (struct ether_header *)p;
        iph_hdr = (struct iphdr *)(eth_hdr + 1);
        uint16_t ether_type = ntohs(eth_hdr->ether_type);
        uint16_t packe_len = sizeof(struct ether_header) + ntohs(iph_hdr->tot_len);
        if (ether_type != ETHERTYPE_IP)
        {
            printf("[packet error] packetId-%d-packeLen-%u-etherType-0x%x\n", i, packe_len, ether_type);
            exit(1);
        }
        p += packe_len;
    }
}

CPacket::CPacket(uint32_t tlp)
{
    pBytes = NULL;
    pData = NULL;
    tranLayPro = tlp;
    // generateData(18, 548);
    generateData(548, 548);
    if (tranLayPro == 0)
    {
        setUdphdr();
    }
    else
    {
        setTcphdr();
    }
    setIphdr();
    setEtherHeader();
    tranBytesFlow();
    // printInfo();
}

CPacket::~CPacket()
{
    if (pBytes)
    {
        free(pBytes);
        pBytes = NULL;
    }
    if (pData)
    {
        free(pData);
        pData = NULL;
    }
}

void CPacket::setEtherHeader()
{
    uint8_t dh[6];
    uint8_t sh[6];
    randHost(dh, sh);
    for (int i = 0; i < 6; i++)
    {
        eth.ether_dhost[i] = dh[i];
        eth.ether_shost[i] = sh[i];
    }
    eth.ether_type = htons(ETHERTYPE_IP);
}

void CPacket::setIphdr()
{
    iph.ihl = 5;
    iph.version = 4;
    iph.tos = 0;
    if (tranLayPro == 0)
    {
        iph.tot_len = htons(dataLen + sizeof(struct iphdr) + sizeof(struct udphdr));
    }
    else
    {
        iph.tot_len = htons(dataLen + sizeof(struct iphdr) + sizeof(struct tcphdr));
    }
    iph.id = htons(0x8d2f);
    iph.frag_off = htons(0x0000);
    iph.ttl = 0x80;
    /*
    设定4层协议类型
    TCP 0x06
    UDP 0x11
    ICMP 0x01
    IGMP 0x02
    */
    if (tranLayPro == 0)
    {
        iph.protocol = 0x11;
    }
    else
    {
        iph.protocol = 0x06;
    }
    iph.check = htons(0);
    iph.saddr = htonl(SRC_ADDR);
    iph.daddr = htonl(0x7f1b0000u | (rand() % DEST_ADDR_NB + 1));
    uint8_t *p_ip = (uint8_t *)malloc(sizeof(uint8_t) * sizeof(struct iphdr));
    memcpy(p_ip, (uint8_t *)(&iph), sizeof(uint8_t) * sizeof(struct iphdr));
    iph.check = htons(getIpCheckSum(p_ip, sizeof(struct iphdr)));
    free(p_ip);
    p_ip = NULL;
}

void CPacket::setUdphdr()
{
    uph.uh_sport = htons(443);
    uph.uh_dport = htons(59622);
    uph.uh_ulen = htons(dataLen + sizeof(struct udphdr));
    uph.uh_sum = htons(0xcb2d);
}

void CPacket::setTcphdr()
{
    tph.source = htons(rand() % MAX_NAT_ENTRIES);
    tph.dest = htons(rand() % MAX_NAT_ENTRIES);
    tph.seq = htonl(rand() % 2048);
    tph.ack_seq = htonl(rand() % 2048);
    tph.res1 = htons(rand() % 16);
    tph.doff = htons(rand() % 16);
    tph.fin = htons(rand() % 2);
    tph.syn = htons(rand() % 2);
    tph.rst = htons(rand() % 2);
    tph.psh = htons(rand() % 2);
    tph.ack = htons(rand() % 2);
    tph.urg = htons(rand() % 2);
    tph.res2 = htons(rand() % 4);
    tph.window = htons(rand() % 2048);
    tph.check = htons(0);
    tph.urg_ptr = htons(rand() % 2048);
    uint8_t *p_tcp = (uint8_t *)malloc(sizeof(uint8_t) * sizeof(struct tcphdr));
    memcpy(p_tcp, (uint8_t *)(&tph), sizeof(uint8_t) * sizeof(struct tcphdr));
    uint8_t *psaddr = (uint8_t *)malloc(sizeof(uint32_t));
    uint8_t *pdaddr = (uint8_t *)malloc(sizeof(uint32_t));
    uint32_t saddr = ntohl(iph.saddr);
    uint32_t daddr = ntohl(iph.daddr);
    memcpy(psaddr, (uint8_t *)(&saddr), sizeof(uint32_t));
    memcpy(pdaddr, (uint8_t *)(&daddr), sizeof(uint32_t));
    tph.check = htons(getTcpCheckSum(sizeof(struct tcphdr), psaddr, pdaddr, 0, p_tcp));
    free(p_tcp);
    free(psaddr);
    free(pdaddr);
}

void CPacket::generateData(uint16_t st, uint16_t ed)
{
    if (st == ed)
    {
        dataLen = st;
    }
    else
    {
        dataLen = rand() % (ed - st + 1) + st;
    }
    pData = (uint8_t *)malloc(sizeof(uint8_t) * dataLen);
    for (int i = 0; i < dataLen; i++)
    {
        if (st != ed)
        {
            pData[i] = 125;
        }
        else
        {
            /*剔除反斜杆*/
            while (1)
            {
                /*随机数32,33,...,125,126*/
                pData[i] = rand() % (126 - 32 + 1) + 32;
                if (pData[i] != 92)
                    break;
            }
        }
    }
    short nb = rand() % (5 - 1 + 1) + 1;
    short len = strlen((char *)(acRule[nb - 1])) - 1;
    assert(dataLen >= len);
    memcpy(pData + nb, acRule[nb - 1], len);
}

/*可输入数据包数据部分（20字节）*/
void CPacket::inputData(int _dataLen)
{
    dataLen = _dataLen;
    pData = (uint8_t *)malloc(sizeof(uint8_t) * dataLen);
    printf("\n>>[测试包]\n请输入数据部分(%d个字符，勿输入反斜杠)：", _dataLen);
    for (int i = 0; i < dataLen; i++)
    {
        scanf("%c", &pData[i]);
        printf("%d ", (int)pData[i]);
    }
    printf("\n");
}

/*通过tranBytesFlow函数可以将CPacket中的内容转换为uint8_t流*/
void CPacket::setBytesFlow(uint16_t _bytesLen, uint8_t *_pBytes)
{
    if (pBytes)
    {
        free(pBytes);
        pBytes = NULL;
    }
    if (pData)
    {
        free(pData);
        pData = NULL;
    }
    bytesLen = _bytesLen;
    pBytes = (uint8_t *)malloc(sizeof(uint8_t) * bytesLen);
    memcpy(pBytes, _pBytes, sizeof(uint8_t) * bytesLen);
}

void CPacket::tranBytesFlow()
{
    bytesLen = sizeof(struct ether_header) + ntohs(iph.tot_len);
    pBytes = (uint8_t *)malloc(sizeof(uint8_t) * bytesLen);
    memcpy(pBytes, (uint8_t *)(&eth), sizeof(uint8_t) * sizeof(struct ethhdr));
    memcpy(pBytes + sizeof(struct ethhdr), (uint8_t *)(&iph), sizeof(uint8_t) * sizeof(struct iphdr));
    if (tranLayPro == 0)
    {
        memcpy(pBytes + sizeof(struct ethhdr) + sizeof(struct iphdr), (uint8_t *)(&uph), sizeof(uint8_t) * sizeof(struct udphdr));
        memcpy(pBytes + sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct udphdr), pData, sizeof(uint8_t) * dataLen);
    }
    else
    {
        memcpy(pBytes + sizeof(struct ethhdr) + sizeof(struct iphdr), (uint8_t *)(&tph), sizeof(uint8_t) * sizeof(struct tcphdr));
        memcpy(pBytes + sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct tcphdr), pData, sizeof(uint8_t) * dataLen);
    }
}

uint8_t *CPacket::bytes()
{
    return pBytes;
}

inline uint32_t CPacket::u8tu32(uint8_t *data)
{
    uint32_t result = 0;
    for (int i = 0; i < 4; i++)
    {
        result += data[i] * pow(16, 6 - 2 * i);
    }
    return result;
}

void CPacket::printInfo()
{
    printf("bytesLen-%u\n", bytesLen);
    printf("dataLen-%u\n", dataLen);
    printf("iph.daddr-%u\n", ntohl(iph.daddr));
    printf("iph.check-%u\n", ntohs(iph.check));
    for (uint32_t i = 0; i < bytesLen; i++)
    {
        printf("%02X ", pBytes[i]);
    }
    printf("\n");
}
