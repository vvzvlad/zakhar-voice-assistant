// Host test of the zakhar_link pure core (zl_core.c) against docs/device-protocol.md: message
// decode and encode, the cutoff and peak numbers, the backoff, the session filter, the playback
// queue and the model receiver. `make fw-host-test` builds it with cJSON at the version ESP-IDF
// 5.5.4 ships.
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "zl_core.h"

static bool decode(const char *json, zl_msg_t *m)
{
    return zl_decode(json, strlen(json), m);
}

static void encoded_is(char *got, const char *want)
{
    if (got == NULL || strcmp(got, want) != 0) {
        fprintf(stderr, "encoded %s\n   want %s\n", got != NULL ? got : "(null)", want);
        assert(0);
    }
    zl_free(got);
}

static void test_decode(void)
{
    zl_msg_t m;
    assert(decode("{\"type\": \"config\", \"wake_cutoff\": 80, \"vad_cutoff\": 30}", &m));
    assert(m.type == ZL_MSG_CONFIG && m.wake == 204 && m.vad == 76 && !m.has_session);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"config\", \"wake_cutoff\": 100, \"vad_cutoff\": 0}", &m));
    assert(m.wake == 255 && m.vad == 0);
    zl_msg_free(&m);
    assert(!decode("{\"type\": \"config\", \"wake_cutoff\": 101, \"vad_cutoff\": 30}", &m));

    assert(decode("{\"type\": \"state\", \"session\": 17, \"value\": \"listening\"}", &m));
    assert(m.type == ZL_MSG_STATE && m.has_session && m.session == 17 && m.state == ZL_STATE_LISTENING);
    assert(m.text == NULL);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"state\", \"session\": 17, \"value\": \"error\", \"text\": \"нет связи\"}", &m));
    assert(m.state == ZL_STATE_ERROR && strcmp(m.text, "нет связи") == 0);
    zl_msg_free(&m);
    assert(!decode("{\"type\": \"state\", \"session\": 17, \"value\": \"dancing\"}", &m));
    assert(!decode("{\"type\": \"state\", \"session\": -1, \"value\": \"idle\"}", &m));

    assert(decode("{\"type\": \"reply\", \"session\": 3, \"text\": \"Включила\"}", &m));
    assert(m.type == ZL_MSG_REPLY && m.has_session && strcmp(m.text, "Включила") == 0);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"reply\", \"text\": \"Напоминание\"}", &m));  // an announcement
    assert(!m.has_session);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"transcript\", \"session\": 3, \"text\": \"свет\", \"extra\": 1}", &m));
    assert(m.type == ZL_MSG_TRANSCRIPT && strcmp(m.text, "свет") == 0);
    zl_msg_free(&m);

    assert(decode("{\"type\": \"play_start\", \"id\": 9, \"session\": 3}", &m));
    assert(m.type == ZL_MSG_PLAY_START && m.id == 9);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"play_end\", \"id\": 9}", &m) && m.type == ZL_MSG_PLAY_END && m.id == 9);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"play_stop\"}", &m) && m.type == ZL_MSG_PLAY_STOP);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"set_volume\", \"value\": 45}", &m) && m.value == 45);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"wake_peak_stream\", \"on\": true}", &m) && m.on);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"detect\", \"on\": false}", &m) && m.type == ZL_MSG_DETECT && !m.on);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"detect\", \"on\": true}", &m) && m.type == ZL_MSG_DETECT && m.on);
    zl_msg_free(&m);
    assert(!decode("{\"type\": \"detect\", \"on\": 1}", &m));
    assert(decode("{\"type\": \"log_stream\", \"on\": true}", &m) && m.type == ZL_MSG_LOG_STREAM && m.on);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"log_stream\", \"on\": false}", &m) && m.type == ZL_MSG_LOG_STREAM && !m.on);
    zl_msg_free(&m);
    assert(!decode("{\"type\": \"log_stream\", \"on\": \"yes\"}", &m));
    assert(!decode("{\"type\": \"log_stream\"}", &m));
    assert(decode("{\"type\": \"capture\", \"id\": 7, \"seconds\": 5}", &m) && m.type == ZL_MSG_CAPTURE);
    assert(m.id == 7 && m.capture_ms == 5000);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"capture\", \"id\": 7, \"seconds\": 1.5}", &m) && m.capture_ms == 1500);  // the time left
    zl_msg_free(&m);
    assert(!decode("{\"type\": \"capture\", \"id\": 7, \"seconds\": 0}", &m));
    assert(!decode("{\"type\": \"capture\", \"seconds\": 5}", &m));  // the id is required
    assert(!decode("{\"type\": \"capture\", \"id\": 0, \"seconds\": 5}", &m));
    assert(decode("{\"type\": \"capture_resume\", \"id\": 7, \"from\": 24576}", &m));
    assert(m.type == ZL_MSG_CAPTURE_RESUME && m.id == 7 && m.from == 24576);
    zl_msg_free(&m);
    assert(!decode("{\"type\": \"capture_resume\", \"id\": 7}", &m));
    assert(decode("{\"type\": \"model\", \"kind\": \"wake\", \"name\": \"zakhar.tflite\", \"size\": 61000}", &m));
    assert(m.type == ZL_MSG_MODEL && m.model == ZL_MODEL_WAKE && strcmp(m.name, "zakhar.tflite") == 0 &&
           m.size == 61000 && !m.has_session);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"model\", \"kind\": \"vad\", \"name\": \"vad.tflite\", \"size\": 1}", &m));
    assert(m.model == ZL_MODEL_VAD);
    zl_msg_free(&m);
    assert(decode("{\"type\": \"model\", \"kind\": \"stop\", \"name\": \"microWakeWord/stop/stop.tflite\", \"size\": 1}", &m));
    assert(m.model == ZL_MODEL_STOP && strcmp(m.name, "microWakeWord/stop/stop.tflite") == 0);
    zl_msg_free(&m);
    assert(!decode("{\"type\": \"model\", \"kind\": \"vad\", \"name\": \"vad.tflite\", \"size\": 0}", &m));
    assert(!decode("{\"type\": \"model\", \"kind\": \"stt\", \"name\": \"x\", \"size\": 5}", &m));
    assert(!decode("{\"type\": \"model\", \"kind\": \"wake\", \"size\": 5}", &m));

    assert(!decode("{\"type\": \"something_new\"}", &m));
    assert(!decode("[1, 2]", &m));
    assert(!decode("{\"type\": \"play_end\"", &m));
    printf("ok: decode\n");
}

static void test_encode(void)
{
    zl_config_t speaker = {.id = "main_room", .fw = "voicepe 0.1.0 (a1b2c3d)", .mic_channels = 2,
                           .play_rate = 48000, .play_buffer_ms = 4000};
    encoded_is(zl_encode_hello(&speaker, 60, 0),
               "{\"type\":\"hello\",\"id\":\"main_room\",\"fw\":\"voicepe 0.1.0 (a1b2c3d)\","
               "\"caps\":{\"speaker\":true,\"screen\":false,\"follow_up\":true},\"mic\":{\"channels\":2},"
               "\"play\":{\"rate\":48000,\"buffer_ms\":4000},\"volume\":60}");
    zl_config_t panel = {.id = "a0b1c2d3e4f5", .fw = "espanel 0.2.0", .screen = true, .mic_channels = 1};
    encoded_is(zl_encode_hello(&panel, 50, 0),
               "{\"type\":\"hello\",\"id\":\"a0b1c2d3e4f5\",\"fw\":\"espanel 0.2.0\","
               "\"caps\":{\"speaker\":false,\"screen\":true,\"follow_up\":true},\"mic\":{\"channels\":1}}");
    encoded_is(zl_encode_hello(&panel, 50, 7),  // a capture is held
               "{\"type\":\"hello\",\"id\":\"a0b1c2d3e4f5\",\"fw\":\"espanel 0.2.0\","
               "\"caps\":{\"speaker\":false,\"screen\":true,\"follow_up\":true},\"mic\":{\"channels\":1},\"capture\":7}");
    encoded_is(zl_encode_wake(17, "Захар"), "{\"type\":\"wake\",\"session\":17,\"word\":\"Захар\"}");
    encoded_is(zl_encode_wake(4294967295u, "x"), "{\"type\":\"wake\",\"session\":4294967295,\"word\":\"x\"}");
    encoded_is(zl_encode_played(9, false, 2, 125), "{\"type\":\"played\",\"id\":9,\"dry\":2,\"gap_ms\":125}");
    encoded_is(zl_encode_played(9, false, 0, 0), "{\"type\":\"played\",\"id\":9,\"dry\":0,\"gap_ms\":0}");
    encoded_is(zl_encode_played(9, true, 0, 0), "{\"type\":\"played\",\"id\":9,\"dropped\":true}");
    encoded_is(zl_encode_volume(60), "{\"type\":\"volume\",\"value\":60}");
    encoded_is(zl_encode_wake_peak(42), "{\"type\":\"wake_peak\",\"value\":42}");
    encoded_is(zl_encode_audio_end("busy", 7), "{\"type\":\"audio_end\",\"reason\":\"busy\",\"id\":7}");
    encoded_is(zl_encode_audio_end("capture", 4294967295u),
               "{\"type\":\"audio_end\",\"reason\":\"capture\",\"id\":4294967295}");
    encoded_is(zl_encode_model_loaded(ZL_MODEL_WAKE, "zakhar.tflite", true),
               "{\"type\":\"model_loaded\",\"kind\":\"wake\",\"name\":\"zakhar.tflite\",\"ok\":true}");
    encoded_is(zl_encode_model_loaded(ZL_MODEL_VAD, "vad.tflite", false),
               "{\"type\":\"model_loaded\",\"kind\":\"vad\",\"name\":\"vad.tflite\",\"ok\":false}");
    encoded_is(zl_encode_stop(true, 17), "{\"type\":\"stop\",\"session\":17}");
    encoded_is(zl_encode_stop(false, 17), "{\"type\":\"stop\"}");  // outside a session
    // Only `len` bytes go out; quotes, newlines and the colour escapes are JSON-escaped.
    const char *lines = "I (12) zakhar_link: \"капча\"\n\033[0;33mW (13) x: y\033[0m\nnot yet";
    encoded_is(zl_encode_log(lines, strlen(lines) - strlen("not yet")),
               "{\"type\":\"log\",\"text\":\"I (12) zakhar_link: \\\"капча\\\"\\n"
               "\\u001b[0;33mW (13) x: y\\u001b[0m\\n\"}");
    encoded_is(zl_encode_log("", 0), "{\"type\":\"log\",\"text\":\"\"}");
    printf("ok: encode\n");
}

static void test_numbers(void)
{
    assert(zl_cutoff_u8(60) == 153 && zl_cutoff_u8(50) == 127 && zl_cutoff_u8(0.5) == 1);
    assert(zl_peak_percent(0) == 0 && zl_peak_percent(255) == 100 && zl_peak_percent(204) == 80);
    const uint32_t want[] = {2, 4, 8, 16, 30, 30};
    uint32_t s = 1;
    for (size_t i = 0; i < sizeof want / sizeof want[0]; i++) assert((s = zl_backoff_next(s)) == want[i]);
    assert(!zl_silent(1000, 1001, 30000));  // a frame stamped after `now` was read
    assert(!zl_silent(30999, 1000, 30000) && zl_silent(31000, 1000, 30000));
    assert(zl_silent(29998, UINT32_MAX - 1, 30000));  // across the tick counter wrap
    assert(!zl_capture_done(48999, 1000, 48000) && zl_capture_done(49000, 1000, 48000));
    assert(!zl_capture_done(47997, UINT32_MAX - 1, 48000) && zl_capture_done(47998, UINT32_MAX - 1, 48000));
    // A capture that cannot be sent is dropped 2 s after its last frame, across the wrap too.
    assert(!zl_capture_expired(49000, 1000, 48000) && !zl_capture_expired(80999, 1000, 48000));
    assert(zl_capture_expired(81000, 1000, 48000));
    assert(!zl_capture_expired(79997, UINT32_MAX - 1, 48000) && zl_capture_expired(79998, UINT32_MAX - 1, 48000));
    uint8_t le[4];
    zl_put_u32le(le, 0x12345678u);
    assert(le[0] == 0x78 && le[1] == 0x56 && le[2] == 0x34 && le[3] == 0x12);
    printf("ok: numbers\n");
}

static zl_msg_t msg_of(const char *json)
{
    zl_msg_t m;
    assert(decode(json, &m));
    return m;
}

static void test_session(void)
{
    zl_sess_t s = {0};
    assert(zl_sess_wake(&s) == ZL_WAKE_START && s.session == 1 && s.open && s.mic);
    // A detection inside the phrase, before or after `listening`, starts no new session.
    assert(zl_sess_wake(&s) == ZL_WAKE_MIC && s.session == 1 && s.mic);
    zl_msg_t cur = msg_of("{\"type\": \"state\", \"session\": 1, \"value\": \"listening\"}");
    zl_msg_t old = msg_of("{\"type\": \"state\", \"session\": 0, \"value\": \"idle\"}");
    zl_msg_t any = msg_of("{\"type\": \"reply\", \"text\": \"a\"}");
    assert(zl_sess_accepts(&s, &cur) && !zl_sess_accepts(&s, &old) && zl_sess_accepts(&s, &any));

    assert(!zl_sess_state(&s, ZL_STATE_LISTENING));  // the mic is on already: no reopen
    assert(s.mic && s.open);
    assert(zl_sess_wake(&s) == ZL_WAKE_MIC && s.session == 1);
    assert(!zl_sess_capture(&s));  // busy: a session is open
    zl_sess_state(&s, ZL_STATE_TRANSCRIBING);
    assert(!s.mic && s.open);
    // A follow-up: a `listening` of the open session whose mic is off turns the mic on again.
    assert(zl_sess_state(&s, ZL_STATE_LISTENING));
    assert(s.mic && s.open);
    zl_sess_state(&s, ZL_STATE_TRANSCRIBING);

    assert(zl_sess_wake(&s) == ZL_WAKE_START && s.session == 2 && s.mic);  // a new wake replaces the session
    assert(!zl_sess_accepts(&s, &cur));
    // The replaced session's playback still passes: only the server's play_stop drops it.
    zl_msg_t play = msg_of("{\"type\": \"play_start\", \"session\": 1, \"id\": 7, \"rate\": 16000}");
    zl_msg_t end = msg_of("{\"type\": \"play_end\", \"session\": 1, \"id\": 7}");
    assert(zl_sess_accepts(&s, &play) && zl_sess_accepts(&s, &end));
    zl_sess_state(&s, ZL_STATE_IDLE);
    assert(!s.mic && !s.open);
    assert(!zl_sess_state(&s, ZL_STATE_LISTENING));  // a closed session reopens nothing
    assert(!s.mic && !s.open);

    assert(zl_sess_capture(&s) && s.capture);
    assert(zl_sess_wake_check(&s) == ZL_WAKE_CAPTURE);
    assert(zl_sess_wake(&s) == ZL_WAKE_CAPTURE && s.session == 2);  // wake words ignored during a capture
    bool open;
    zl_sess_lost(&s, &open);
    assert(!open && s.capture);  // a lost link keeps the capture
    assert(zl_sess_wake(&s) == ZL_WAKE_CAPTURE);
    s.capture = false;  // the capture ended
    assert(zl_sess_wake(&s) == ZL_WAKE_START);
    zl_sess_lost(&s, &open);
    assert(open && !s.open && !s.mic);

    // Detection off: wake words are refused and say why, a capture still starts; a lost
    // connection turns detection on again.
    const uint32_t before = s.session;
    zl_sess_detect(&s, false);
    assert(zl_sess_wake_check(&s) == ZL_WAKE_DETECT_OFF);
    assert(zl_sess_wake(&s) == ZL_WAKE_DETECT_OFF && s.session == before && !s.open && !s.mic);
    assert(zl_sess_capture(&s) && s.capture);
    zl_sess_lost(&s, &open);
    assert(zl_sess_wake_check(&s) == ZL_WAKE_CAPTURE);  // detection is on again, the capture still runs
    s.capture = false;
    assert(zl_sess_wake(&s) == ZL_WAKE_START && s.session == before + 1);
    zl_sess_detect(&s, false);
    zl_sess_detect(&s, true);
    assert(zl_sess_wake_check(&s) == ZL_WAKE_MIC);  // detection is on again; the session still streams

    zl_msg_free(&cur);
    zl_msg_free(&old);
    zl_msg_free(&any);
    zl_msg_free(&play);
    zl_msg_free(&end);
    printf("ok: session\n");
}

static void write_samples(zl_playq_t *q, const int16_t *v, size_t n)
{
    zl_playq_write(q, (const uint8_t *)v, n * sizeof *v);  // the host is little-endian, as the wire
}

static void test_playq(void)
{
    int16_t buf[8], out[16];
    uint32_t ids[ZL_PLAYQ_STREAMS];
    zl_played_t done[ZL_PLAYQ_STREAMS];
    zl_playq_t q;
    zl_playq_init(&q, buf, 8, 0);

    // A sample split across two frames, then read out across the end of the stream.
    zl_playq_write(&q, (const uint8_t *)"\x01\x00", 2);  // no stream receiving: discarded
    assert(q.count == 0);
    assert(zl_playq_start(&q, 1));
    zl_playq_write(&q, (const uint8_t *)"\x01\x00\x02", 3);
    zl_playq_write(&q, (const uint8_t *)"\x01", 1);
    assert(q.count == 2);
    assert(zl_playq_read(&q, out, 16) == 2 && out[0] == 1 && out[1] == 0x0102);
    assert(zl_playq_take_done(&q, done) == 0);  // out, but play_end has not come
    zl_playq_end(&q, 1);
    assert(zl_playq_take_done(&q, done) == 1 && done[0].id == 1);

    // Two streams back to back: each is done when its last sample is out.
    const int16_t a[] = {10, 11, 12}, b[] = {20, 21};
    assert(zl_playq_start(&q, 2));
    write_samples(&q, a, 3);
    zl_playq_end(&q, 2);
    assert(zl_playq_start(&q, 3));
    write_samples(&q, b, 2);
    zl_playq_end(&q, 3);
    assert(zl_playq_read(&q, out, 2) == 2 && zl_playq_take_done(&q, done) == 0);
    assert(zl_playq_read(&q, out, 2) == 2 && out[0] == 12 && out[1] == 20);
    assert(zl_playq_take_done(&q, done) == 1 && done[0].id == 2);
    assert(zl_playq_read(&q, out, 16) == 1 && out[0] == 21);
    assert(zl_playq_take_done(&q, done) == 1 && done[0].id == 3);

    // A full buffer discards the rest; play_stop drops queued and receiving streams.
    const int16_t many[10] = {0};
    assert(zl_playq_start(&q, 4));
    write_samples(&q, many, 10);
    assert(q.count == 8 && q.overflow == 2);
    zl_playq_end(&q, 4);
    assert(zl_playq_start(&q, 5));
    assert(zl_playq_flush(&q, ids) == 2 && ids[0] == 4 && ids[1] == 5);
    write_samples(&q, a, 3);  // the dropped stream's frames and play_end still in flight
    zl_playq_end(&q, 5);
    assert(q.count == 0 && zl_playq_read(&q, out, 16) == 0 && zl_playq_take_done(&q, done) == 0);

    // An empty stream is done at play_end; a new play_start ends one still receiving.
    assert(zl_playq_start(&q, 6));
    zl_playq_end(&q, 99);  // not the receiving stream
    zl_playq_end(&q, 6);
    assert(zl_playq_take_done(&q, done) == 1 && done[0].id == 6);
    assert(zl_playq_start(&q, 7) && zl_playq_start(&q, 8));
    assert(zl_playq_take_done(&q, done) == 1 && done[0].id == 7);

    // Queued and done streams share the slots.
    size_t started = 1;  // 8 is receiving
    while (zl_playq_start(&q, 100 + (uint32_t)started)) started++;
    assert(started == ZL_PLAYQ_STREAMS);

    // Priming at 3 samples: a stream waits for 3 samples, plays, runs dry, waits for 3 again; a
    // shorter one plays at its play_end. Reads of 4: the silence it ran into is the samples they
    // came short by once it started, 1 then 4; the wait for its first prime is not.
    zl_playq_init(&q, buf, 8, 3);
    const int16_t c[] = {30, 31, 32, 33, 34};
    assert(zl_playq_start(&q, 9));
    write_samples(&q, c, 2);
    assert(zl_playq_read(&q, out, 4) == 0);
    write_samples(&q, c + 2, 1);
    assert(zl_playq_read(&q, out, 4) == 3 && out[0] == 30 && out[2] == 32);
    write_samples(&q, c + 3, 1);
    assert(zl_playq_read(&q, out, 4) == 0);  // ran dry: one sample is not enough to resume
    write_samples(&q, c + 4, 1);
    zl_playq_end(&q, 9);
    assert(zl_playq_read(&q, out, 4) == 2 && out[0] == 33 && out[1] == 34);
    assert(zl_playq_take_done(&q, done) == 1 && done[0].id == 9 && done[0].dry == 1 && done[0].gap == 1 + 4);
    // At a stream boundary the next stream primes on its own: 2 samples of it wait, 3 play on.
    // The wait is a gap of neither: the first has played out, the next has not started.
    assert(zl_playq_start(&q, 10));
    write_samples(&q, c, 2);
    zl_playq_end(&q, 10);
    assert(zl_playq_start(&q, 11));
    write_samples(&q, c, 2);
    assert(zl_playq_read(&q, out, 16) == 2 && zl_playq_take_done(&q, done) == 1 && done[0].id == 10);
    assert(done[0].dry == 0 && done[0].gap == 0);
    write_samples(&q, c + 2, 1);
    assert(zl_playq_read(&q, out, 16) == 3 && out[0] == 30);
    printf("ok: playback queue\n");
}

static void test_modelrx(void)
{
    zl_modelrx_t r = {0};
    size_t at;
    bool done;
    assert(zl_modelrx_write(&r, 5, &at, &done) == 0 && !done);  // no model announced: takes nothing

    // Uneven writes assemble the model in order; the bytes past `size` are dropped; done fires once.
    zl_modelrx_start(&r, 10);
    assert(zl_modelrx_write(&r, 3, &at, &done) == 3 && at == 0 && !done);
    assert(zl_modelrx_write(&r, 0, &at, &done) == 0 && at == 3 && !done);
    assert(zl_modelrx_write(&r, 4, &at, &done) == 4 && at == 3 && !done);
    assert(zl_modelrx_write(&r, 7, &at, &done) == 3 && at == 7 && done && !r.active);
    assert(zl_modelrx_write(&r, 2, &at, &done) == 0 && !done);

    // A second start drops the model still receiving and counts from 0.
    zl_modelrx_start(&r, 8);
    assert(zl_modelrx_write(&r, 5, &at, &done) == 5 && !done);
    zl_modelrx_start(&r, 4);
    assert(zl_modelrx_write(&r, 6, &at, &done) == 4 && at == 0 && done);
    printf("ok: model receiver\n");
}

int main(void)
{
    test_decode();
    test_encode();
    test_numbers();
    test_session();
    test_playq();
    test_modelrx();
    printf("all zl_core tests passed\n");
    return 0;
}
