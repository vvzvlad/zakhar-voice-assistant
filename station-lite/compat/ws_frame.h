// The WebSocket wire format (RFC 6455) of the compat esp_websocket_client: frame headers, the
// client's masking and the opening handshake. Pure: host-tested (make station-host-test).
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define WS_OP_TEXT 0x1
#define WS_OP_BINARY 0x2
#define WS_OP_CLOSE 0x8
#define WS_OP_PING 0x9
#define WS_OP_PONG 0xA

#define WS_HEADER_MAX 14  // 2 bytes, 8 of extended length, 4 of mask

typedef struct {
    bool fin;
    uint8_t opcode;
    bool masked;
    uint64_t len;
    uint8_t mask[4];
} ws_header_t;

// A client frame's header, FIN set and masked with `mask`, for a payload of `len` bytes, into
// `out` (WS_HEADER_MAX bytes); returns its length.
size_t ws_header_build(uint8_t *out, uint8_t opcode, uint64_t len, const uint8_t mask[4]);
// The length of the header that starts with these two bytes.
size_t ws_header_size(const uint8_t first[2]);
// The header at the start of `buf`: its length, or 0 when the `n` bytes do not hold all of it.
size_t ws_header_parse(const uint8_t *buf, size_t n, ws_header_t *h);
// XORs `n` payload bytes with the mask; `at` is the offset of p[0] in the payload.
void ws_mask(uint8_t *p, size_t n, const uint8_t mask[4], uint64_t at);

// Sec-WebSocket-Key for a 16-byte nonce: 24 characters of base64 and a NUL.
void ws_key(const uint8_t nonce[16], char key[25]);
// The Sec-WebSocket-Accept the server answers `key` with: 28 characters and a NUL.
void ws_accept(const char *key, char accept[29]);
// The upgrade request for ws://<host>:<port><path>; its length, 0 when it does not fit `cap`.
size_t ws_request(char *out, size_t cap, const char *host, unsigned port, const char *path, const char *key);
// Whether the response headers (NUL-terminated) accept the upgrade asked for with `key`: status 101
// and the Sec-WebSocket-Accept of that key.
bool ws_response_ok(const char *response, const char *key);
