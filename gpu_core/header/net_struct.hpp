#pragma once
#include <iostream>

#include <arpa/inet.h>
#include <net/ethernet.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <netinet/tcp.h>

#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

// #define DEFAULT_EVP_CIPHER_CTX
//开启以使用系统默认的EVP_CIPHER_CTX

#define ESP_IV_LENGTH 16
#define HMAC_KEY_SIZE 64
#define IGNORED_IP 0xFFffFFffu

#define MAX_NAT_ENTRIES 2048
/*255.255.0.0*/
#define LAN_NET_MASK 0xFfFf0000
/*172.27.0.0*/
#define LAN_NET_IP 0x7f1b0000
/*172.27.237.118*/
#define HOST_IP_ADDR 0x7f1bed76

struct ipaddr_pair
{
    uint32_t src_addr;
    uint32_t dest_addr;
    bool operator==(const ipaddr_pair &other) const
    {
        return (src_addr == other.src_addr && dest_addr == other.dest_addr);
    }
};

namespace std
{
    template <>
    struct hash<ipaddr_pair>
    {
    public:
        std::size_t operator()(ipaddr_pair const &p) const
        {
            return std::hash<uint64_t>()((((uint64_t)p.src_addr) << 32) | (p.dest_addr));
        }
    };
} // namespace std

struct esphdr
{
    uint32_t esp_spi;              /* Security Parameters Index */
    uint32_t esp_rpl;              /* Replay counter */
    uint8_t esp_iv[ESP_IV_LENGTH]; /* initial vector */
};

struct espencap_sa_entry
{
    uint32_t spi;                      /* Security Parameters Index */
    uint32_t rpl; /* Replay counter */ // XXX: is this right to use this one?
    uint32_t gwaddr;                   // XXX: not used yet; when this value is used?
    uint64_t entry_idx;
};

struct alignas(8) hmac_sa_entry
{
    uint8_t hmac_key[HMAC_KEY_SIZE];
    int entry_idx;
};

#ifndef DEFAULT_EVP_CIPHER_CTX
struct evp_cipher_ctx_st
{
    const EVP_CIPHER *cipher;
    //ENGINE *engine;       /* functional reference if 'cipher' is ENGINE-provided */
    void *engine;
    int encrypt; /* encrypt or decrypt */
    int buf_len; /* number we have left */

    unsigned char oiv[EVP_MAX_IV_LENGTH];    /* original iv */
    unsigned char iv[EVP_MAX_IV_LENGTH];     /* working iv */
    unsigned char buf[EVP_MAX_BLOCK_LENGTH]; /* saved partial block */
    int num;                                 /* used by cfb/ofb/ctr mode */

    void *app_data;      /* application stuff */
    int key_len;         /* May change for variable length cipher */
    unsigned long flags; /* Various flags */
    void *cipher_data;   /* per EVP data */
    int final_used;
    int block_mask;
    unsigned char final[EVP_MAX_BLOCK_LENGTH]; /* possible final block */
} /* EVP_CIPHER_CTX */;
#endif

struct alignas(8) aes_sa_entry
{
    AES_KEY aes_key_t; // Prepared for AES library function.
    EVP_CIPHER_CTX evpctx;
    int entry_idx; // Index of current flow: value for verification.
};