import React, { useState, useEffect, useRef } from "react";
import Z from "../data.js";
import { nav } from "../navStore.js";
import { Ic } from "../components/icons.jsx";
import { Spark, Field, Seg, Selector, Toggle, Slider, Stepper, Select, Pill, StatusPill, Waterfall, segsFor, total, PageHeader, SaveBar, Modal, Player, KV } from "../components/primitives.jsx";
  function StageNav({ active }) {
    return <div className="z-snav">
      {Z.stages.map((s, i) => <React.Fragment key={s.key}>
        <div className={"z-snav-i" + (s.key === active ? " on" : "") + (s.status === "off" ? " off" : "")} onClick={() => nav(s.key)}>
          <span className={"z-dot " + s.status} />{s.name}
        </div>
        {i < Z.stages.length - 1 && <span className="z-snav-sep">›</span>}
      </React.Fragment>)}
    </div>;
  }
  function Card({ title, sub, children, foot }) {
    return <div className="z-card">
      {title && <div className="z-card-h"><b>{title}</b>{sub && <span className="sub">{sub}</span>}</div>}
      <div className="z-card-b">{children}</div>
      {foot}
    </div>;
  }
  function KeyInput({ value }) {
    return <div className="z-inp mono"><input value={value} readOnly /><button className="z-mini">SHOW</button></div>;
  }

  // ── VAD ────────────────────────────────────────────────
  function VAD() {
    const [preset, setPreset] = useState("Balanced");
    const [agg, setAgg] = useState(2);
    return <div className="z-page">
      <PageHeader title="VAD · Voice capture" crumb="Pipeline / Stage 01"
        desc="The speaker streams audio continuously and never signals end-of-phrase — we detect it with WebRTC VAD. Tune sensitivity to pauses here." />
      <div>
          <Card title="Sensitivity preset" sub="quick tuning">
            <Field label="Pause sensitivity" hint="Presets adjust trailing-silence and min-speech together. Fine-tune below.">
              <Seg full options={["Fast", "Balanced", "Patient"]} value={preset} onChange={setPreset} />
            </Field>
          </Card>
          <div style={{ height: 16 }} />
          <Card title="Advanced parameters" foot={<SaveBar restart />}>
            <Field label="Filter aggressiveness" param="VAD_AGGRESSIVENESS" hint="Higher = cuts non-speech harder (0–3).">
              <div style={{ display: "flex", alignItems: "center", gap: 14 }}><div style={{ flex: 1 }}><Slider min={0} max={3} step={1} value={agg} onChange={setAgg} fmt={(v) => v} /></div></div>
            </Field>
            <Field label="Trailing silence" param="VAD_SILENCE_MS" hint="Silence after speech that finalizes the phrase." row>
              <Stepper value={800} step={50} unit="ms" onChange={() => {}} />
            </Field>
            <Field label="Min speech" param="VAD_MIN_SPEECH_MS" hint="Speech needed before arming end-of-utterance." row>
              <Stepper value={200} step={50} unit="ms" onChange={() => {}} />
            </Field>
            <Field label="Max utterance" param="VAD_MAX_UTTERANCE_MS" hint="Hard cap — finalize even without trailing silence." row>
              <Stepper value={15000} step={500} unit="ms" onChange={() => {}} />
            </Field>
            <Field label="No-speech timeout" param="VAD_NO_SPEECH_TIMEOUT_MS" hint="If no speech at all — end the turn." row>
              <Stepper value={8000} step={500} unit="ms" onChange={() => {}} />
            </Field>
          </Card>
      </div>
    </div>;
  }

  // ── STT ──────────────────────────────────────────────────────────────
  function STT() {
    const [prov, setProv] = useState(Z.stt.provider);
    return <div className="z-page">
      <PageHeader title="STT · Speech to text" crumb="Pipeline / Stage 02" desc="Recognize the captured phrase. Cloud Whisper via groq, or offline Vosk." />
      <Selector label="Backend" options={Z.stt.providers} value={prov} onChange={setProv} />
      <div className="z-cols">
        <div>
          <Card title={prov === "groq" ? "groq · cloud Whisper" : "vosk · offline"} foot={<SaveBar restart />}>
            {prov === "groq" ? <>
              <Field label="API key" param="STT_API_KEY" hint="groq key, stored encrypted."><KeyInput value={Z.stt.groq.STT_API_KEY} /></Field>
              <Field label="Model" param="STT_MODEL" hint="Whisper variant."><div className="z-inp mono"><input value={Z.stt.groq.STT_MODEL} readOnly /></div></Field>
              <div className="z-row2">
                <Field label="Language" param="(hardcoded)" hint="Recognition language."><Select value="ru" options={["ru", "en", "auto"]} /></Field>
                <Field label="Request timeout" param="(hardcoded)" hint="Seconds."><Stepper value={60} step={5} unit="s" onChange={() => {}} /></Field>
              </div>
            </> : <>
              <Field label="Model path" param="VOSK_MODEL_PATH" hint="Offline model directory on the server."><div className="z-inp mono"><input value={Z.stt.vosk.VOSK_MODEL_PATH} readOnly /></div></Field>
              <div className="z-banner warn" style={{ margin: "10px 0" }}><Ic n="system" w={14} /><span>Model must be downloaded on the server — run <b>make models</b>.</span></div>
            </>}
          </Card>
        </div>
        <div className="z-aside">
          <Card title="Test recognition">
            <div className="z-info" style={{ padding: "10px 0 12px" }}>Recognize a short sample to verify the provider and key.</div>
            <button className="z-btn g" style={{ width: "100%", justifyContent: "center" }}><Ic n="play" w={12} />Recognize sample</button>
            <div style={{ margin: "12px 0 10px" }}><div className="z-pv-line">→ «привет захар, как дела»</div></div>
          </Card>
        </div>
      </div>
    </div>;
  }

  // ── LLM ────────────────────────────────────────────────
  function LLM() {
    const [prov, setProv] = useState(Z.llm.provider);
    const [temp, setTemp] = useState(Z.llm.temperature);
    return <div className="z-page">
      <PageHeader title="LLM · Reasoning & tools" crumb="Pipeline / Stage 03" desc="Generates the reply and calls smart-home tools over MCP. OpenAI-compatible chat completions." />
      <div>
          <Selector label="Provider" options={Z.llm.providers} value={prov} onChange={setProv} />
          <Card title="Model & generation" foot={<SaveBar />}>
            <Field label="API key" param="INTENT_API_KEY" hint="Provider key, stored encrypted."><KeyInput value={Z.llm.INTENT_API_KEY} /></Field>
            <Field label="Model" param="INTENT_MODEL" hint="Provider model slug."><div className="z-inp mono"><input value={Z.llm.INTENT_MODEL} readOnly /></div></Field>
            <Field label="Temperature" param="temperature" hint="Creativity vs. determinism."><Slider min={0} max={2} step={0.1} value={temp} onChange={setTemp} fmt={(v) => v.toFixed(1)} /></Field>
            <div className="z-row2">
              <Field label="Max tokens" param="max_tokens" hint="Response length cap." row><Stepper value={4096} step={256} onChange={() => {}} /></Field>
              <Field label="Max tool rounds" param="MAX_TOOL_ROUNDS" hint="Model↔tools loop guard." row><Stepper value={5} step={1} min={1} onChange={() => {}} /></Field>
            </div>
            <Field label="Request timeout" param="(hardcoded)" hint="Seconds." row><Stepper value={300} step={30} unit="s" onChange={() => {}} /></Field>
          </Card>
          <div style={{ height: 16 }} />
          <Card title="System replies" sub="hardcoded fallbacks — editable" foot={<SaveBar />}>
            <Field label="On rate-limit (429)" hint="Spoken when the provider throttles."><div className="z-inp"><input value={Z.llm.fallbacks.rate_limit} readOnly /></div></Field>
            <Field label="Empty after tools" hint="When tools ran but the model returned no text."><div className="z-inp"><input value={Z.llm.fallbacks.empty_after_tools} readOnly /></div></Field>
            <Field label="Empty response" hint="When nothing came back at all."><div className="z-inp"><input value={Z.llm.fallbacks.empty} readOnly /></div></Field>
          </Card>
      </div>
    </div>;
  }

  // ── RUAccent ───────────────────────────────────────────
  function RUAccent() {
    const ru = Z.ruaccent;
    const [on, setOn] = useState(ru.enabled);
    const [model, setModel] = useState(ru.model);
    const [dict, setDict] = useState(ru.useDict);
    const [homo, setHomo] = useState(ru.homographs);
    return <div className="z-page">
      <PageHeader title="RUAccent · Stress marks" crumb="Pipeline / Stage 04 · optional"
        desc="Optional stage between LLM and TTS. Automatically places Russian stress marks (handles homographs and ё), so the LLM doesn't have to."
        actions={<div style={{ display: "flex", alignItems: "center", gap: 10 }}><span style={{ fontSize: 12.5, fontWeight: 600, color: on ? "var(--acc-ink)" : "var(--mut)" }}>{on ? "Enabled" : "Disabled"}</span><Toggle on={on} onChange={setOn} /></div>} />
      <div className="z-cols" style={{ opacity: on ? 1 : 0.6, transition: "opacity .15s" }}>
        <div>
          <Card title="Model & options" foot={<SaveBar />}>
            <Field label="Model" param="RUACCENT_MODEL" hint="Accuracy ↔ RAM/latency. Heavier = better.">
              <Seg options={ru.models} value={model} onChange={setModel} />
            </Field>
            <div style={{ display: "flex", gap: 8, margin: "2px 0 6px" }}>
              {ru.models.map((m) => <div key={m} style={{ flex: 1, fontSize: 10.5, color: m === model ? "var(--acc-ink)" : "var(--mut2)", fontFamily: "var(--mono)", textAlign: "center" }}>{ru.modelInfo[m]}</div>)}
            </div>
            <Field label="Use dictionary" param="USE_DICT" hint="More accurate, but more RAM." row><Toggle on={dict} onChange={setDict} /></Field>
            <Field label="Handle homographs" param="HOMOGRAPHS" hint="Disambiguate за́мок / замо́к by context." row><Toggle on={homo} onChange={setHomo} /></Field>
          </Card>
          <div style={{ height: 16 }} />
          <Card title="Custom stress dictionary" sub={ru.dict.length + " entries"} foot={<div className="z-foot"><button className="z-btn g sm"><Ic n="add" w={13} />Add word</button></div>}>
            <table className="z-dicttbl">
              <tbody>
                {ru.dict.map((d, i) => <tr key={i}><td style={{ color: "var(--mut)" }}>{d.word}</td><td>→</td><td className="accent" style={{ color: "var(--acc-ink)", fontWeight: 600 }}>{d.accented}</td><td style={{ textAlign: "right" }}><button className="z-dictdel">×</button></td></tr>)}
              </tbody>
            </table>
          </Card>
        </div>
        <div className="z-aside">
          <Card title="Preview">
            <div style={{ padding: "10px 0 12px" }} className="z-preview-io">
              <div className="z-inp"><input defaultValue={ru.previewIn} /></div>
              <div style={{ textAlign: "center", color: "var(--mut2)", fontSize: 12 }}>↓ accents placed</div>
              <div className="z-pv-line"><span className="accent">{ru.previewOut}</span></div>
            </div>
            <button className="z-btn g" style={{ width: "100%", justifyContent: "center", marginBottom: 12 }}>Re-run preview</button>
          </Card>
          <Card title="Impact"><div className="z-info" style={{ padding: "10px 0 12px" }}>Adds a stage to the pipeline — shows as its own segment in the log timeline and affects TTS pronunciation. On <b>{model}</b> ≈ +{model === "big" ? "180" : model === "turbo" ? "90" : "40"} ms.</div></Card>
        </div>
      </div>
    </div>;
  }

  // ── TTS ────────────────────────────────────────────────
  function TTS() {
    const [backend, setBackend] = useState(Z.tts.backend);
    const [emotion, setEmotion] = useState(Z.tts.yandex.YANDEX_TTS_EMOTION);
    const [speed, setSpeed] = useState(Z.tts.yandex.YANDEX_TTS_SPEED);
    const y = Z.tts.yandex;
    return <div className="z-page">
      <PageHeader title="TTS · Text to speech" crumb="Pipeline / Stage 05" desc="Synthesize the reply to audio served to the speakers." />
      <Selector label="Engine" options={Z.tts.backends} value={backend} onChange={setBackend} />
      <div className="z-cols wide">
        <div>
          <Card title={backend + " · synthesis"} foot={<SaveBar restart />}>
            {backend === "yandex" && <>
              <Field label="API key" param="YANDEX_TTS_API_KEY" hint="Cloud service key, format AQVN… · encrypted."><KeyInput value={y.YANDEX_TTS_API_KEY} /></Field>
              <div className="z-row2">
                <Field label="Voice" param="YANDEX_TTS_VOICE" hint="Male / female SpeechKit voices."><Select value={y.YANDEX_TTS_VOICE} options={y.voices} /></Field>
                <Field label="Emotion" param="YANDEX_TTS_EMOTION" hint="Depends on the voice."><Seg full options={y.emotions} value={emotion} onChange={setEmotion} /></Field>
              </div>
              <Field label="Speed" param="YANDEX_TTS_SPEED" hint="0.1–3.0 · playback rate."><Slider min={0.1} max={3} step={0.1} value={speed} onChange={setSpeed} fmt={(v) => v.toFixed(1) + "×"} /></Field>
              <div className="z-row2">
                <Field label="Folder ID" param="YANDEX_TTS_FOLDER_ID" hint="IAM auth only — empty for service key."><div className="z-inp mono"><input placeholder="— empty —" readOnly /></div></Field>
                <Field label="Timeout" param="TTS_TIMEOUT" hint="Seconds." row><Stepper value={30} step={5} unit="s" onChange={() => {}} /></Field>
              </div>
              <Field label="Endpoint" param="YANDEX_TTS_URL" hint="Advanced · usually unchanged."><div className="z-inp mono" style={{ fontSize: 11 }}><input value={y.YANDEX_TTS_URL} readOnly /></div></Field>
            </>}
            {backend === "teratts" && <Field label="Service base URL" param="TTS_BASE_URL" hint="HTTP TTS service endpoint."><div className="z-inp mono"><input value={Z.tts.teratts.TTS_BASE_URL} readOnly /></div></Field>}
            {backend === "piper" && <>
              <Field label="Voice path (.onnx)" param="PIPER_VOICE_PATH" hint="Offline voice model; <path>.json expected alongside."><div className="z-inp mono"><input value={Z.tts.piper.PIPER_VOICE_PATH} readOnly /></div></Field>
              <Field label="Sentence silence" param="TTS_SENTENCE_SILENCE" hint="Pause between sentences, seconds." row><Stepper value={0.4} step={0.1} unit="s" onChange={() => {}} /></Field>
            </>}
          </Card>
        </div>
        <div className="z-aside">
          <Card title="Preview · synthesize">
            <div style={{ padding: "10px 0 12px", display: "flex", flexDirection: "column", gap: 11 }}>
              <div className="z-inp"><input defaultValue="Привет, я Захар. Чем могу помочь?" /></div>
              <button className="z-btn p" style={{ justifyContent: "center" }}><Ic n="play" w={12} />Speak</button>
              <Player audio={{ ms: 680, bytes: 41984, fmt: "mp3" }} bars={40} />
            </div>
          </Card>
        </div>
      </div>
    </div>;
  }

export { VAD, STT, LLM, RUAccent, TTS };
