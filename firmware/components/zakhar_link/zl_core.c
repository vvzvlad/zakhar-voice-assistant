// The pure core of zakhar_link (zl_core.h).
#include "zl_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"

// ---------------------------------------------------------------- decode

static const char *const STATES[] = {
    [ZL_STATE_LISTENING] = "listening",
    [ZL_STATE_TRANSCRIBING] = "transcribing",
    [ZL_STATE_THINKING] = "thinking",
    [ZL_STATE_SPEAKING] = "speaking",
    [ZL_STATE_IDLE] = "idle",
    [ZL_STATE_ERROR] = "error",
};

static const char *const MODELS[] = {
    [ZL_MODEL_WAKE] = "wake",
    [ZL_MODEL_VAD] = "vad",
    [ZL_MODEL_STOP] = "stop",
};

static const cJSON *field(const cJSON *o, const char *name)
{
    return cJSON_GetObjectItemCaseSensitive(o, name);
}

static bool get_u32(const cJSON *v, uint32_t *out)
{
    if (!cJSON_IsNumber(v)) return false;
    const double d = v->valuedouble;
    if (d < 0 || d > UINT32_MAX || d != floor(d)) return false;
    *out = (uint32_t)d;
    return true;
}

static bool get_percent(const cJSON *v, double *out)
{
    if (!cJSON_IsNumber(v) || !(v->valuedouble >= 0 && v->valuedouble <= 100)) return false;
    *out = v->valuedouble;
    return true;
}

// An optional integer field in lo..hi: its value, or -1 when it is absent or anything else.
static int get_setting(const cJSON *v, int lo, int hi)
{
    if (!cJSON_IsNumber(v)) return -1;
    const double d = v->valuedouble;
    return d >= lo && d <= hi && d == floor(d) ? (int)d : -1;
}

// An optional boolean field: 1 true, 0 false, -1 when it is absent or anything else.
static int get_flag(const cJSON *v)
{
    return cJSON_IsBool(v) ? cJSON_IsTrue(v) : -1;
}

// A string field: *out is its text, `absent` when the field is missing.
static bool get_text(const cJSON *o, const char *name, const char *absent, const char **out)
{
    const cJSON *v = field(o, name);
    if (v == NULL) {
        *out = absent;
        return true;
    }
    if (!cJSON_IsString(v)) return false;
    *out = v->valuestring;
    return true;
}

static bool decode_fields(const cJSON *o, const char *type, zl_msg_t *m)
{
    double a, b;
    if (strcmp(type, "config") == 0) {
        m->type = ZL_MSG_CONFIG;
        if (!get_percent(field(o, "wake_cutoff"), &a) || !get_percent(field(o, "vad_cutoff"), &b)) return false;
        m->wake = zl_cutoff_u8(a);
        m->vad = zl_cutoff_u8(b);
        // Optional: a setting this device cannot take leaves its own, and the cutoffs apply.
        m->echo_cancel = get_setting(field(o, "echo_cancel"), 0, 3);
        m->echo_mode = get_setting(field(o, "echo_cancel_mode"), 0, 3);
        m->echo_filter = get_setting(field(o, "echo_cancel_filter"), 1, 8);
        m->auto_volume = get_flag(field(o, "auto_volume"));
        return true;
    }
    if (strcmp(type, "state") == 0) {
        m->type = ZL_MSG_STATE;
        const cJSON *v = field(o, "value");
        if (!cJSON_IsString(v)) return false;
        size_t i = 0;
        while (i < sizeof STATES / sizeof STATES[0] && strcmp(v->valuestring, STATES[i]) != 0) i++;
        if (i == sizeof STATES / sizeof STATES[0]) return false;
        m->state = (zl_state_t)i;
        return get_text(o, "text", NULL, &m->text);
    }
    if (strcmp(type, "transcript") == 0) {
        m->type = ZL_MSG_TRANSCRIPT;
        return get_text(o, "text", "", &m->text);
    }
    if (strcmp(type, "reply") == 0) {
        m->type = ZL_MSG_REPLY;
        return get_text(o, "text", "", &m->text);
    }
    if (strcmp(type, "play_start") == 0) {
        m->type = ZL_MSG_PLAY_START;
        return get_u32(field(o, "id"), &m->id);
    }
    if (strcmp(type, "play_end") == 0) {
        m->type = ZL_MSG_PLAY_END;
        return get_u32(field(o, "id"), &m->id);
    }
    if (strcmp(type, "play_stop") == 0) {
        m->type = ZL_MSG_PLAY_STOP;
        return true;
    }
    if (strcmp(type, "set_volume") == 0) {
        m->type = ZL_MSG_SET_VOLUME;
        if (!get_percent(field(o, "value"), &a)) return false;
        m->value = (int)a;
        return true;
    }
    if (strcmp(type, "volume_boost") == 0) {
        m->type = ZL_MSG_VOLUME_BOOST;
        if (!get_percent(field(o, "value"), &a)) return false;
        m->value = (int)a;
        return true;
    }
    if (strcmp(type, "wake_peak_stream") == 0) {
        m->type = ZL_MSG_WAKE_PEAK_STREAM;
        const cJSON *v = field(o, "on");
        if (!cJSON_IsBool(v)) return false;
        m->on = cJSON_IsTrue(v);
        return true;
    }
    if (strcmp(type, "detect") == 0) {
        m->type = ZL_MSG_DETECT;
        const cJSON *v = field(o, "on");
        if (!cJSON_IsBool(v)) return false;
        m->on = cJSON_IsTrue(v);
        return true;
    }
    if (strcmp(type, "log_stream") == 0) {
        m->type = ZL_MSG_LOG_STREAM;
        const cJSON *v = field(o, "on");
        if (!cJSON_IsBool(v)) return false;
        m->on = cJSON_IsTrue(v);
        return true;
    }
    if (strcmp(type, "capture") == 0) {
        m->type = ZL_MSG_CAPTURE;
        if (!get_u32(field(o, "id"), &m->id) || m->id == 0) return false;
        const cJSON *v = field(o, "seconds");
        if (!cJSON_IsNumber(v) || !(v->valuedouble > 0 && v->valuedouble * 1000 <= UINT32_MAX)) return false;
        m->capture_ms = (uint32_t)(v->valuedouble * 1000);
        return true;
    }
    if (strcmp(type, "capture_resume") == 0) {
        m->type = ZL_MSG_CAPTURE_RESUME;
        return get_u32(field(o, "id"), &m->id) && get_u32(field(o, "from"), &m->from);
    }
    if (strcmp(type, "model") == 0) {
        m->type = ZL_MSG_MODEL;
        const cJSON *kind = field(o, "kind"), *name = field(o, "name");
        if (!cJSON_IsString(kind) || !cJSON_IsString(name) || !get_u32(field(o, "size"), &m->size) || m->size == 0) {
            return false;
        }
        size_t i = 0;
        while (i < sizeof MODELS / sizeof MODELS[0] && strcmp(kind->valuestring, MODELS[i]) != 0) i++;
        if (i == sizeof MODELS / sizeof MODELS[0]) return false;
        m->model = (zl_model_t)i;
        m->name = name->valuestring;
        return true;
    }
    return false;
}

bool zl_decode(const char *data, size_t len, zl_msg_t *m)
{
    memset(m, 0, sizeof *m);
    cJSON *o = cJSON_ParseWithLength(data, len);
    const cJSON *type = field(o, "type");
    const cJSON *session = field(o, "session");
    bool ok = cJSON_IsObject(o) && cJSON_IsString(type);
    if (ok && session != NULL) ok = m->has_session = get_u32(session, &m->session);
    if (ok) ok = decode_fields(o, type->valuestring, m);
    if (!ok) {
        cJSON_Delete(o);
        memset(m, 0, sizeof *m);
        return false;
    }
    m->json = o;
    return true;
}

void zl_msg_free(zl_msg_t *m)
{
    cJSON_Delete(m->json);
    m->json = NULL;
    m->text = NULL;
    m->name = NULL;
}

// ---------------------------------------------------------------- encode

static cJSON *message(const char *type)
{
    cJSON *o = cJSON_CreateObject();
    if (o != NULL && cJSON_AddStringToObject(o, "type", type) == NULL) {
        cJSON_Delete(o);
        o = NULL;
    }
    return o;
}

// The text of o when every field went in (ok), else NULL; o is released either way.
static char *finish(cJSON *o, bool ok)
{
    char *s = ok ? cJSON_PrintUnformatted(o) : NULL;
    cJSON_Delete(o);
    return s;
}

char *zl_encode_hello(const zl_config_t *cfg, int volume, uint32_t capture)
{
    const bool speaker = cfg->play_rate > 0;
    cJSON *o = message("hello");
    cJSON *caps = NULL, *mic = NULL, *play = NULL;
    bool ok = o != NULL && cJSON_AddStringToObject(o, "id", cfg->id) != NULL &&
              cJSON_AddStringToObject(o, "fw", cfg->fw) != NULL &&
              (caps = cJSON_AddObjectToObject(o, "caps")) != NULL &&
              cJSON_AddBoolToObject(caps, "speaker", speaker) != NULL &&
              cJSON_AddBoolToObject(caps, "screen", cfg->screen) != NULL &&
              cJSON_AddTrueToObject(caps, "follow_up") != NULL &&
              (mic = cJSON_AddObjectToObject(o, "mic")) != NULL &&
              cJSON_AddNumberToObject(mic, "channels", cfg->mic_channels) != NULL;
    if (ok && speaker) {
        ok = (play = cJSON_AddObjectToObject(o, "play")) != NULL &&
             cJSON_AddNumberToObject(play, "rate", cfg->play_rate) != NULL &&
             cJSON_AddNumberToObject(play, "buffer_ms", cfg->play_buffer_ms) != NULL &&
             cJSON_AddNumberToObject(o, "volume", volume) != NULL;
    }
    if (ok && capture != 0) ok = cJSON_AddNumberToObject(o, "capture", capture) != NULL;
    return finish(o, ok);
}

char *zl_encode_wake(uint32_t session, const char *word)
{
    cJSON *o = message("wake");
    return finish(o, o != NULL && cJSON_AddNumberToObject(o, "session", session) != NULL &&
                         cJSON_AddStringToObject(o, "word", word) != NULL);
}

char *zl_encode_played(uint32_t id, bool dropped, uint32_t dry, uint32_t gap_ms)
{
    cJSON *o = message("played");
    bool ok = o != NULL && cJSON_AddNumberToObject(o, "id", id) != NULL;
    if (ok && dropped) ok = cJSON_AddTrueToObject(o, "dropped") != NULL;
    if (ok && !dropped) {
        ok = cJSON_AddNumberToObject(o, "dry", dry) != NULL && cJSON_AddNumberToObject(o, "gap_ms", gap_ms) != NULL;
    }
    return finish(o, ok);
}

static char *value_message(const char *type, int value)
{
    cJSON *o = message(type);
    return finish(o, o != NULL && cJSON_AddNumberToObject(o, "value", value) != NULL);
}

char *zl_encode_volume(int percent)
{
    return value_message("volume", percent);
}

char *zl_encode_wake_peak(int percent)
{
    return value_message("wake_peak", percent);
}

char *zl_encode_audio_end(const char *reason, uint32_t id)
{
    cJSON *o = message("audio_end");
    return finish(o, o != NULL && cJSON_AddStringToObject(o, "reason", reason) != NULL &&
                         cJSON_AddNumberToObject(o, "id", id) != NULL);
}

char *zl_encode_model_loaded(zl_model_t kind, const char *name, bool ok)
{
    cJSON *o = message("model_loaded");
    return finish(o, o != NULL && cJSON_AddStringToObject(o, "kind", MODELS[kind]) != NULL &&
                         cJSON_AddStringToObject(o, "name", name) != NULL &&
                         cJSON_AddBoolToObject(o, "ok", ok) != NULL);
}

char *zl_encode_stop(bool has_session, uint32_t session)
{
    cJSON *o = message("stop");
    bool ok = o != NULL;
    if (ok && has_session) ok = cJSON_AddNumberToObject(o, "session", session) != NULL;
    return finish(o, ok);
}

char *zl_encode_log(const char *text, size_t len)
{
    char *s = malloc(len + 1);  // cJSON takes NUL-terminated strings
    if (s == NULL) return NULL;
    memcpy(s, text, len);
    s[len] = '\0';
    cJSON *o = message("log");
    char *json = finish(o, o != NULL && cJSON_AddStringToObject(o, "text", s) != NULL);
    free(s);
    return json;
}

char *zl_encode_ambient(double dbfs)
{
    cJSON *o = message("ambient");
    return finish(o, o != NULL && cJSON_AddNumberToObject(o, "value", dbfs) != NULL);
}

void zl_free(char *json)
{
    cJSON_free(json);
}

// ---------------------------------------------------------------- numbers

const char *zl_state_name(zl_state_t state)
{
    return STATES[state];
}

const char *zl_model_name(zl_model_t kind)
{
    return MODELS[kind];
}

uint8_t zl_cutoff_u8(double percent)
{
    return (uint8_t)(percent / 100 * 255);
}

int zl_peak_percent(uint8_t peak)
{
    return (peak * 100 + 127) / 255;
}

uint32_t zl_backoff_next(uint32_t s)
{
    return s * 2 > 30 ? 30 : s * 2;
}

bool zl_silent(uint32_t now, uint32_t heard, uint32_t limit)
{
    return (int32_t)(now - heard) >= (int32_t)limit;
}

bool zl_capture_done(uint32_t pos, uint32_t start, uint32_t frames)
{
    return pos - start >= frames;
}

bool zl_capture_expired(uint32_t pos, uint32_t start, uint32_t frames)
{
    return (uint64_t)(pos - start) >= (uint64_t)frames + ZL_CAPTURE_KEEP_FRAMES;
}

void zl_put_u32le(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

// ---------------------------------------------------------------- the voice session

zl_wake_t zl_sess_wake_check(const zl_sess_t *s)
{
    if (s->detect_off) return ZL_WAKE_DETECT_OFF;
    if (s->capture) return ZL_WAKE_CAPTURE;
    if (s->mic) return ZL_WAKE_MIC;
    return ZL_WAKE_START;
}

zl_wake_t zl_sess_wake(zl_sess_t *s)
{
    const zl_wake_t w = zl_sess_wake_check(s);
    if (w != ZL_WAKE_START) return w;
    s->session++;
    s->open = true;
    s->mic = true;
    return ZL_WAKE_START;
}

void zl_sess_detect(zl_sess_t *s, bool on)
{
    s->detect_off = !on;
}

bool zl_sess_accepts(const zl_sess_t *s, const zl_msg_t *m)
{
    return !m->has_session || m->session == s->session;
}

bool zl_sess_state(zl_sess_t *s, zl_state_t state)
{
    if (state == ZL_STATE_LISTENING) {
        const bool reopen = s->open && !s->mic;
        if (reopen) s->mic = true;
        return reopen;
    }
    s->mic = false;
    if (state == ZL_STATE_IDLE || state == ZL_STATE_ERROR) s->open = false;
    return false;
}

bool zl_sess_capture(zl_sess_t *s)
{
    if (s->open) return false;
    s->capture = true;
    return true;
}

void zl_sess_lost(zl_sess_t *s, bool *was_open)
{
    *was_open = s->open;
    s->open = false;
    s->mic = false;
    s->detect_off = false;
}

// ---------------------------------------------------------------- the playback queue

void zl_playq_init(zl_playq_t *q, int16_t *buf, size_t cap, size_t prime)
{
    memset(q, 0, sizeof *q);
    q->buf = buf;
    q->cap = cap;
    q->prime = prime;
}

// The leading streams that are fully out and ended move to `done`.
static void collect(zl_playq_t *q)
{
    while (q->nstreams > 0 && q->streams[0].left == 0 && q->streams[0].ended) {
        const zl_stream_t *s = &q->streams[0];
        q->done[q->ndone++] = (zl_played_t){.id = s->id, .dry = s->dry, .gap = s->gap};
        q->nstreams--;
        memmove(&q->streams[0], &q->streams[1], q->nstreams * sizeof q->streams[0]);
    }
}

static void end_receiving(zl_playq_t *q)
{
    q->streams[q->nstreams - 1].ended = true;
    q->receiving = false;
    q->has_odd = false;
    collect(q);
}

bool zl_playq_start(zl_playq_t *q, uint32_t id)
{
    if (q->receiving) end_receiving(q);
    if (q->nstreams + q->ndone >= ZL_PLAYQ_STREAMS) return false;
    q->streams[q->nstreams++] = (zl_stream_t){.id = id};
    q->receiving = true;
    return true;
}

void zl_playq_write(zl_playq_t *q, const uint8_t *data, size_t len)
{
    if (!q->receiving) return;
    zl_stream_t *s = &q->streams[q->nstreams - 1];
    for (size_t i = 0; i < len; i++) {
        if (!q->has_odd) {
            q->odd = data[i];
            q->has_odd = true;
            continue;
        }
        q->has_odd = false;
        if (q->count == q->cap) {
            q->overflow++;
            continue;
        }
        q->buf[(q->head + q->count) % q->cap] = (int16_t)(uint16_t)(q->odd | (uint16_t)data[i] << 8);
        q->count++;
        s->left++;
    }
}

void zl_playq_end(zl_playq_t *q, uint32_t id)
{
    if (q->receiving && q->streams[q->nstreams - 1].id == id) end_receiving(q);
}

size_t zl_playq_read(zl_playq_t *q, int16_t *out, size_t max)
{
    size_t n = 0;
    while (n < max && q->nstreams > 0) {
        zl_stream_t *s = &q->streams[0];
        if (!s->playing && !s->ended && s->left < q->prime) break;  // not primed yet
        if (s->left == 0) {
            if (!s->ended) {
                // Ran dry while the rest is still arriving: prime again. Counted once, on the
                // read that finds a playing stream empty.
                if (s->playing) s->dry++;
                s->playing = false;
                break;
            }
            collect(q);
            continue;
        }
        s->playing = true;
        s->started = true;
        size_t take = s->left < max - n ? s->left : max - n;
        while (take > 0) {
            const size_t run = take < q->cap - q->head ? take : q->cap - q->head;
            memcpy(out + n, q->buf + q->head, run * sizeof *out);
            n += run;
            take -= run;
            s->left -= run;
            q->count -= run;
            q->head = (q->head + run) % q->cap;
        }
    }
    collect(q);
    // The caller plays silence for the samples it did not get. A head stream that has started
    // is waiting mid-stream, so they are its gap; one still waiting for its first prime is not.
    if (n < max && q->nstreams > 0 && q->streams[0].started) q->streams[0].gap += max - n;
    return n;
}

size_t zl_playq_flush(zl_playq_t *q, uint32_t *dropped)
{
    const size_t n = q->nstreams;
    for (size_t i = 0; i < n; i++) dropped[i] = q->streams[i].id;
    q->nstreams = 0;
    q->head = 0;
    q->count = 0;
    q->receiving = false;
    q->has_odd = false;
    return n;
}

size_t zl_playq_take_done(zl_playq_t *q, zl_played_t *out)
{
    const size_t n = q->ndone;
    memcpy(out, q->done, n * sizeof *out);
    q->ndone = 0;
    return n;
}

// ---------------------------------------------------------------- the model receiver

void zl_modelrx_start(zl_modelrx_t *r, size_t size)
{
    *r = (zl_modelrx_t){.active = true, .size = size};
}

size_t zl_modelrx_write(zl_modelrx_t *r, size_t n, size_t *at, bool *done)
{
    *at = r->got;
    *done = false;
    if (!r->active) return 0;
    const size_t take = n < r->size - r->got ? n : r->size - r->got;
    r->got += take;
    if (r->got == r->size) {
        r->active = false;
        *done = true;
    }
    return take;
}
