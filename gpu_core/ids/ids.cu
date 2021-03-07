#include <stdio.h>
#include <assert.h>

#include "ac.hpp"
#include "ids.cuh"
#include "../header/net_struct.hpp"
#include "../header/gpu_config.hpp"

// static uint32_t *res;
// static uint32_t *_res;
// static __device__ uint32_t *devRes;

static ac_machine_t *cacm;
static ac_machine_t *_cacm;
static __device__ ac_machine_t *devCacm;

static ac_state_t *_states;
static int *_transitions;
static int *_outputs;
static char *_dim1Patterns;

/*AC规则库*/
static char **ruleArgv;

/*规则的个数*/
static int ruleNb;

int loadAcFromFile(const char *acRule)
{
    printf("[ids] 正在打开文件%s\n", acRule);
    FILE *fp = fopen(acRule, "r");

    if (fp == NULL)
    {
        printf("[ids] 打开文件%s出错\n", acRule);
    }
    assert(fp != NULL);

    ruleNb = 1;
    int ch;
    while (EOF != (ch = getc(fp)))
    {
        if ('\n' == ch)
        {
            ruleNb++;
        }
    }
    // printf("%u\n", ruleNb);
    fclose(fp);

    if (ruleNb != DEV_RULE_NB)
    {
        printf("[ids error] ac_rule.txt中规则数量为%u,", ruleNb);
        printf("和DEV_RULE_NB的值不相等，访问ids/ids.cuh修改DEV_RULE_NB的值\n");
        exit(1);
    }

    ch = 0;

    ruleArgv = (char **)malloc(sizeof(char *) * ruleNb);

    fp = fopen(acRule, "r");
    char buf[RULE_STRING_MAXSIZE];
    while (fgets(buf, RULE_STRING_MAXSIZE, fp))
    {
        if (ch >= ruleNb)
        {
            printf("[ids error] ac_rule.txt存在长度大于RULE_STRING_MAXSIZE的规则！\n");
            exit(1);
        }
        for (int i = 0; i < RULE_STRING_MAXSIZE; i++)
        {
            if (buf[i] == '\n')
            {
                buf[i] = '\0';
                break;
            }
        }
        ruleArgv[ch] = (char *)malloc(sizeof(char) * RULE_STRING_MAXSIZE);
        memcpy(ruleArgv[ch], buf, sizeof(buf));
        // printf("%s\n", ruleArgv[ch]);
        ch++;
    }
    fclose(fp);
    printf("[ids] 已经成功导入%d条规则\n", ruleNb);
    return 0;
}

void initIds(const char *acRule)
{
    loadAcFromFile(acRule);
    ac_machine_t acm;
    cacm = &acm;
    ac_build_machine(cacm, ruleArgv, ruleNb, 0);

    for (int i = 0; i < ruleNb; i++)
    {
        free(ruleArgv[i]);
    }
    free(ruleArgv);

    // res = (uint32_t *)malloc(sizeof(uint32_t) * ruleNb);
    // memset(res, 0, sizeof(uint32_t) * ruleNb);

    /*将二维数组压平*/
    cacm->dim1_patterns = (char *)malloc(sizeof(char) * RULE_STRING_MAXSIZE * cacm->npatterns);
    for (int i = 0; i < cacm->npatterns; i++)
    {
        for (int j = 0; j < RULE_STRING_MAXSIZE; j++)
        {
            cacm->dim1_patterns[i * RULE_STRING_MAXSIZE + j] = cacm->patterns[i][j];
        }
    }

    /*1.分配设备结构体变量*/
    cudaMalloc((void **)&_cacm, sizeof(ac_machine_t));

    /*2.分配设备指针*/
    cudaMalloc((void **)&_states, sizeof(ac_state_t) * cacm->nstates);
    cudaMalloc((void **)&_transitions, sizeof(int) * cacm->nstates * AC_ALPHABET_SIZE);
    cudaMalloc((void **)&_outputs, sizeof(int) * cacm->noutputs);
    cudaMalloc((void **)&_dim1Patterns, sizeof(char) * RULE_STRING_MAXSIZE * cacm->npatterns);

    // cudaMalloc((void **)&_res, sizeof(uint32_t) * ruleNb);

    /*3.将指针内容从主机复制到设备*/
    cudaMemcpy(_states, cacm->states, sizeof(ac_state_t) * cacm->nstates, cudaMemcpyHostToDevice);
    cudaMemcpy(_transitions, cacm->transitions, sizeof(int) * cacm->nstates * AC_ALPHABET_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(_outputs, cacm->outputs, sizeof(int) * cacm->noutputs, cudaMemcpyHostToDevice);
    cudaMemcpy(_dim1Patterns, cacm->dim1_patterns, sizeof(char) * RULE_STRING_MAXSIZE * cacm->npatterns, cudaMemcpyHostToDevice);

    free(cacm->dim1_patterns);

    /*4.指向主机结构中的设备指针*/
    cacm->states = _states;
    cacm->transitions = _transitions;
    cacm->outputs = _outputs;
    cacm->dim1_patterns = _dim1Patterns;

    /*5.将结构体从主机复制到设备*/
    cudaMemcpy(_cacm, cacm, sizeof(ac_machine_t), cudaMemcpyHostToDevice);

    // cudaMemcpy(_res, res, sizeof(uint32_t) * ruleNb, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(devCacm, &_cacm, sizeof(_cacm));
    // cudaMemcpyToSymbol(devRes, &_res, sizeof(_res));

    // free(res);
    ac_release_machine(cacm);
}

void freeIds()
{
    cudaFree(_states);
    cudaFree(_transitions);
    cudaFree(_outputs);
    cudaFree(_dim1Patterns);
    cudaFree(_cacm);
    // cudaFree(_res);
}

static __device__ int devStrlen(const char *str)
{
    int count = 0;
    while (*str)
    {
        count++;
        str++;
    }
    return count;
}

static __device__ __inline__ uint16_t u8ToU16(uint8_t *data)
{
    /*网络字节顺序为__BIG_ENDIAN 地址的低位存储值的高位*/
    uint16_t result = 0;
    for (int i = 0; i < 2; i++)
    {
        result += data[i] * pow(16, 2 - 2 * i);
    }
    return result;
}

static __device__ int devAcMatch(uint8_t *str, int len, int once, ac_machine_t *acm, uint32_t *res)
{
    int nm = 0;
    char c;
    ac_state_t *st = acm_state(acm, 0);
    for (int i = 0; i < len; i++)
    {
        c = str[i];
        if (c < 0)
        {
            return -1;
        }

        int nid = acm_state_transitions(acm, st->id)[(int)c];
        st = acm_state(acm, nid);
        if (st->noutput > 0)
        {
            if (res)
            {
                for (int j = 0; j < st->noutput; j++)
                {
                    int ot = acm_state_output(acm, st->output)[j];
                    res[ot] = 0x80000000 | (i + 1 - devStrlen(DEV_ACM_PATTERN(acm, ot)));
                }
            }
            if (!nm && once)
            {
                return st->noutput;
            }
            nm += st->noutput;
        }
    }
    return nm;
}

__device__ void gpuIds(uint32_t packetsPerTh, uint8_t *in, uint32_t *out)
{
    for (uint32_t i = 0; i < packetsPerTh; i++)
    {
        struct iphdr *iph = (struct iphdr *)((struct ether_header *)in + 1);

        /*对于uint16_t在这里不要使用iph->tot_len方式，这样会出现CUDA_EXCEPTION_6 Warp Misaligned Address
        另外，如果使用u8ToU16则不需要转换字节顺序*/
        uint16_t packetLen = sizeof(struct ether_header) + u8ToU16((uint8_t *)iph + 2 * sizeof(uint8_t));
        assert(packetLen != sizeof(struct ether_header));
        // uint16_t packetLen = sizeof(struct ether_header) + ntohsGpu(iph->tot_len);
        // printf("%u\n",packetLen);

        // #ifdef ASSERT_PACKETS
        //         uint16_t etherType = u8ToU16((uint8_t *)iph - sizeof(uint16_t));
        //         if (etherType != ETHERTYPE_IP)
        //         {
        //             printf("[ids error] etherType-0x%x\n", etherType);
        //         }
        //         assert(etherType == ETHERTYPE_IP);
        // #endif
        uint8_t protocol = ((uint8_t *)(iph + 1) - 2 * sizeof(uint32_t) - sizeof(uint16_t) - sizeof(uint8_t))[0];
        uint32_t tranLen = 0;
        if (protocol == 0x11)
        {
            tranLen = sizeof(struct udphdr);
        }
        else if (protocol == 0x06)
        {
            tranLen = sizeof(struct tcphdr);
        }
        // #ifdef ASSERT_PACKETS
        //         if (packetLen != DEBUG_DATA_LEN + sizeof(struct ether_header) + sizeof(struct iphdr) + tranLen)
        //         {
        //             printf("[ids error] packetLen-0x%x\n", packetLen);
        //         }
        //         assert(packetLen == DEBUG_DATA_LEN + sizeof(struct ether_header) + sizeof(struct iphdr) + tranLen);
        // #endif

        /*计算每个包的数据部分长度*/
        int dataLen = packetLen - sizeof(struct ether_header) - sizeof(struct iphdr) - tranLen;

        // printf("%d\n", dataLen);

        /*定位每个包的数据部分*/
        uint8_t *packetData = in + sizeof(struct ether_header) + sizeof(struct iphdr) + tranLen;

        // printf("%s\n", packetData);

        /*
        定义局部动态数组必须保证devRuleNb是const类型或常数
        且devRuleNb的值较小
        res是一个local变量
        */
        uint32_t res[DEV_RULE_NB];

        for (int j = 0; j < DEV_RULE_NB; j++)
        {
            res[j] = 0;
        }

        int r = devAcMatch(packetData, dataLen, 0, devCacm, res);

        if (out != NULL)
        {
            out[i] = 0;
        }

        if (r > 0)
        {
            /*ACMatch匹配到ac_rule.txt中的字符串*/
            for (int j = 0; j < DEV_RULE_NB; j++)
            {
                if (ac_res_found(res[j]))
                {
                    // char *matchedRule = devCacm->dim1_patterns + j * RULE_STRING_MAXSIZE;
                    // uint32_t matchedLoc = ac_res_location(res[j]);
                    if (out != NULL)
                    {
                        out[i]++;
                    }
                }
            }
        }

        in += packetLen;
    }
}
