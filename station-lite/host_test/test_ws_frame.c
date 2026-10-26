// Host test of the compat WebSocket wire format (station-lite/compat/ws_frame.c) against RFC 6455's
// examples. `make station-host-test`.
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "ws_frame.h"

static const uint8_t MASK[4] = {0x37, 0xfa, 0x21, 0x3d};  // RFC 6455 section 5.7

static void handshake(void)
{
    // RFC 6455 section 1.3: the sample nonce, its key and the server's accept.
    char key[25], accept[29];
    ws_key((const uint8_t *)"the sample nonce", key);
    assert(strcmp(key, "dGhlIHNhbXBsZSBub25jZQ==") == 0);
    ws_accept(key, accept);
    assert(strcmp(accept, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") == 0);

    char req[512];
    const size_t n = ws_request(req, sizeof req, "10.0.0.5", 8765, "/device", key);
    assert(n == strlen(req));
    assert(strcmp(req, "GET /device HTTP/1.1\r\nHost: 10.0.0.5:8765\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                       "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n") == 0);
    assert(ws_request(req, 40, "10.0.0.5", 8765, "/device", key) == 0);

    // aiohttp's answer, the header name in any case.
    assert(ws_response_ok("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: upgrade\r\n"
                          "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n",
                          key));
    assert(ws_response_ok("HTTP/1.1 101 Switching Protocols\r\nsec-websocket-accept:s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n",
                          key));
    assert(!ws_response_ok("HTTP/1.1 101 Switching Protocols\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOX=\r\n\r\n",
                           key));
    assert(!ws_response_ok("HTTP/1.1 404 Not Found\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n", key));
    assert(!ws_response_ok("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\r\n", key));
}

static void frames(void)
{
    // RFC 6455 section 5.7: a masked "Hello" from a client.
    uint8_t f[WS_HEADER_MAX + 5];
    const size_t head = ws_header_build(f, WS_OP_TEXT, 5, MASK);
    assert(head == 6);
    memcpy(f + head, "Hello", 5);
    ws_mask(f + head, 5, MASK, 0);
    static const uint8_t HELLO[] = {0x81, 0x85, 0x37, 0xfa, 0x21, 0x3d, 0x7f, 0x9f, 0x4d, 0x51, 0x58};
    assert(memcmp(f, HELLO, sizeof HELLO) == 0);

    ws_header_t h;
    assert(ws_header_size(HELLO) == 6);
    assert(ws_header_parse(HELLO, 5, &h) == 0);  // the mask not in yet
    assert(ws_header_parse(HELLO, 6, &h) == 6);
    assert(h.fin && h.opcode == WS_OP_TEXT && h.masked && h.len == 5 && memcmp(h.mask, MASK, 4) == 0);
    uint8_t p[5];
    memcpy(p, HELLO + 6, 5);
    ws_mask(p, 2, h.mask, 0);  // unmasked in two pieces
    ws_mask(p + 2, 3, h.mask, 2);
    assert(memcmp(p, "Hello", 5) == 0);

    // A server's unmasked frames: a single one, and the first of a fragmented message.
    static const uint8_t SERVER[] = {0x81, 0x05};
    assert(ws_header_parse(SERVER, 2, &h) == 2 && h.fin && h.opcode == WS_OP_TEXT && !h.masked && h.len == 5);
    static const uint8_t FIRST[] = {0x01, 0x03};
    assert(ws_header_parse(FIRST, 2, &h) == 2 && !h.fin && h.opcode == WS_OP_TEXT);
    // RFC 6455's 256-byte and 64 KiB binary frames, unmasked.
    static const uint8_t B256[] = {0x82, 0x7E, 0x01, 0x00};
    assert(ws_header_size(B256) == 4 && ws_header_parse(B256, 4, &h) == 4 && h.opcode == WS_OP_BINARY && h.len == 256);
    static const uint8_t B64K[] = {0x82, 0x7F, 0, 0, 0, 0, 0, 0x01, 0x00, 0x00};
    assert(ws_header_size(B64K) == 10 && ws_header_parse(B64K, 9, &h) == 0);
    assert(ws_header_parse(B64K, 10, &h) == 10 && h.len == 65536);

    // Every length class round-trips through build and parse.
    static const uint64_t LENS[] = {0, 125, 126, 2053, 65535, 65536, 1u << 20};
    for (size_t i = 0; i < sizeof LENS / sizeof LENS[0]; i++) {
        uint8_t b[WS_HEADER_MAX];
        const size_t n = ws_header_build(b, WS_OP_BINARY, LENS[i], MASK);
        assert(n == (LENS[i] < 126 ? 6u : LENS[i] <= 0xFFFF ? 8u : 14u));
        assert(ws_header_size(b) == n && ws_header_parse(b, n, &h) == n);
        assert(h.fin && h.opcode == WS_OP_BINARY && h.masked && h.len == LENS[i] && memcmp(h.mask, MASK, 4) == 0);
    }
    // Control frames.
    assert(ws_header_build(f, WS_OP_PONG, 0, MASK) == 6 && f[0] == 0x8A && f[1] == 0x80);
    assert(ws_header_build(f, WS_OP_CLOSE, 2, MASK) == 6 && f[0] == 0x88 && f[1] == 0x82);
}

int main(void)
{
    handshake();
    frames();
    puts("ws_frame: ok");
    return 0;
}
