#include "ws_frame.h"

#include <stdio.h>
#include <string.h>
#include <strings.h>

#define WS_GUID "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

size_t ws_header_build(uint8_t *out, uint8_t opcode, uint64_t len, const uint8_t mask[4])
{
    size_t n = 0;
    out[n++] = (uint8_t)(0x80 | opcode);
    if (len < 126) {
        out[n++] = (uint8_t)(0x80 | len);
    } else if (len <= 0xFFFF) {
        out[n++] = 0x80 | 126;
        out[n++] = (uint8_t)(len >> 8);
        out[n++] = (uint8_t)len;
    } else {
        out[n++] = 0x80 | 127;
        for (int i = 7; i >= 0; i--) out[n++] = (uint8_t)(len >> (8 * i));
    }
    memcpy(out + n, mask, 4);
    return n + 4;
}

size_t ws_header_size(const uint8_t first[2])
{
    const uint8_t l = first[1] & 0x7F;
    return 2 + (l == 126 ? 2 : l == 127 ? 8 : 0) + ((first[1] & 0x80) ? 4 : 0);
}

size_t ws_header_parse(const uint8_t *buf, size_t n, ws_header_t *h)
{
    if (n < 2) return 0;
    const size_t size = ws_header_size(buf);
    if (n < size) return 0;
    h->fin = (buf[0] & 0x80) != 0;
    h->opcode = buf[0] & 0x0F;
    h->masked = (buf[1] & 0x80) != 0;
    const uint8_t l = buf[1] & 0x7F;
    size_t at = 2;
    if (l == 126) {
        h->len = (uint64_t)buf[2] << 8 | buf[3];
        at = 4;
    } else if (l == 127) {
        h->len = 0;
        for (int i = 0; i < 8; i++) h->len = h->len << 8 | buf[2 + i];
        at = 10;
    } else {
        h->len = l;
    }
    if (h->masked) memcpy(h->mask, buf + at, 4);
    return size;
}

void ws_mask(uint8_t *p, size_t n, const uint8_t mask[4], uint64_t at)
{
    for (size_t i = 0; i < n; i++) p[i] ^= mask[(at + i) % 4];
}

// ---------------------------------------------------------------- the handshake

static uint32_t rol(uint32_t v, int s)
{
    return v << s | v >> (32 - s);
}

// SHA-1 (FIPS 180-4) of a message shorter than 2^32 bytes.
static void sha1(const uint8_t *msg, size_t len, uint8_t digest[20])
{
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    const size_t total = (len + 8) / 64 * 64 + 64;  // the message, 0x80, zeros, the 64-bit length
    for (size_t block = 0; block < total; block += 64) {
        uint8_t b[64];
        for (size_t i = 0; i < 64; i++) {
            const size_t at = block + i;
            b[i] = at < len ? msg[at] : at == len ? 0x80 : 0;
        }
        if (block + 64 == total) {
            const uint64_t bits = (uint64_t)len * 8;
            for (int i = 0; i < 8; i++) b[56 + i] = (uint8_t)(bits >> (56 - 8 * i));
        }
        uint32_t w[80];
        for (int i = 0; i < 16; i++) {
            w[i] = (uint32_t)b[4 * i] << 24 | (uint32_t)b[4 * i + 1] << 16 | (uint32_t)b[4 * i + 2] << 8 | b[4 * i + 3];
        }
        for (int i = 16; i < 80; i++) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) {
                f = (bb & c) | (~bb & d);
                k = 0x5A827999;
            } else if (i < 40) {
                f = bb ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (i < 60) {
                f = (bb & c) | (bb & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = bb ^ c ^ d;
                k = 0xCA62C1D6;
            }
            const uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rol(bb, 30);
            bb = a;
            a = t;
        }
        h[0] += a;
        h[1] += bb;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 4; j++) digest[4 * i + j] = (uint8_t)(h[i] >> (24 - 8 * j));
    }
}

// Standard base64 with padding; `out` takes 4 * ceil(n / 3) characters and a NUL.
static void base64(const uint8_t *in, size_t n, char *out)
{
    static const char ABC[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t)in[i] << 16 | (i + 1 < n ? (uint32_t)in[i + 1] << 8 : 0) | (i + 2 < n ? in[i + 2] : 0);
        out[o++] = ABC[v >> 18 & 63];
        out[o++] = ABC[v >> 12 & 63];
        out[o++] = i + 1 < n ? ABC[v >> 6 & 63] : '=';
        out[o++] = i + 2 < n ? ABC[v & 63] : '=';
    }
    out[o] = '\0';
}

void ws_key(const uint8_t nonce[16], char key[25])
{
    base64(nonce, 16, key);
}

void ws_accept(const char *key, char accept[29])
{
    char text[64 + sizeof WS_GUID];
    const int n = snprintf(text, sizeof text, "%s" WS_GUID, key);
    uint8_t digest[20];
    sha1((const uint8_t *)text, (size_t)n, digest);
    base64(digest, sizeof digest, accept);
}

size_t ws_request(char *out, size_t cap, const char *host, unsigned port, const char *path, const char *key)
{
    const int n = snprintf(out, cap,
                           "GET %s HTTP/1.1\r\n"
                           "Host: %s:%u\r\n"
                           "Upgrade: websocket\r\n"
                           "Connection: Upgrade\r\n"
                           "Sec-WebSocket-Key: %s\r\n"
                           "Sec-WebSocket-Version: 13\r\n"
                           "\r\n",
                           path, host, port, key);
    return n > 0 && (size_t)n < cap ? (size_t)n : 0;
}

bool ws_response_ok(const char *response, const char *key)
{
    if (strncmp(response, "HTTP/1.1 101 ", 13) != 0) return false;
    char want[29];
    ws_accept(key, want);
    static const char NAME[] = "Sec-WebSocket-Accept:";
    for (const char *line = strstr(response, "\r\n"); line != NULL; line = strstr(line, "\r\n")) {
        line += 2;
        if (strncasecmp(line, NAME, sizeof NAME - 1) != 0) continue;
        const char *v = line + sizeof NAME - 1;
        while (*v == ' ' || *v == '\t') v++;
        return strncmp(v, want, 28) == 0 && (v[28] == '\r' || v[28] == ' ' || v[28] == '\t');
    }
    return false;
}
