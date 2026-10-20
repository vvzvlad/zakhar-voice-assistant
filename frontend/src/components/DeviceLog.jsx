// One speaker's own log, which the server keeps on disk one file per day (src/device_logs.py).
// Pick a day to read it; the newest day may still be growing, so while it is shown it is polled
// and the new lines are appended. Mounted only while it is on screen, so it polls only then.
import React, { useState, useEffect, useLayoutEffect, useRef } from "react";
import { Field, Select, Loading, ErrorBox } from "./primitives.jsx";
import { getDeviceLogs, getDeviceLog } from "../api.js";

const POLL_MS = 2000;
// ESP-IDF colours its log lines with ANSI escapes; the view shows the plain text.
const ANSI_RE = /\x1b\[[0-9;]*m/g;

export default function DeviceLog({ device }) {
  const [days, setDays] = useState(null); // [YYYY-MM-DD, newest first] once loaded
  const [error, setError] = useState(null);
  const [day, setDay] = useState("");
  // The lines of each read, one block per line: appending blocks leaves the earlier ones, and a
  // selection in them, untouched, where a single growing string would be rewritten and the
  // selection lost; and a line too long for the box wraps under a hanging indent, so its tail
  // does not read as a line of its own.
  const [chunks, setChunks] = useState([]);
  const [readError, setReadError] = useState(null);
  const boxRef = useRef(null);
  // Whether the next render scrolls the text to its end: on the first read of a day, and after
  // a read that found the reader already at the end.
  const stickRef = useRef(true);

  const loadDays = () => {
    setError(null);
    getDeviceLogs()
      .then((r) => {
        const own = r.devices[device] || [];
        setDays(own);
        setDay(own[0] || "");
      })
      .catch(setError);
  };
  useEffect(() => { loadDays(); }, [device]);

  // Read the chosen day from its start; the newest day is then re-read every POLL_MS from where
  // the last read stopped, each read bringing whole lines only.
  useEffect(() => {
    if (!day) return undefined;
    const live = days[0] === day;
    let alive = true;
    let offset = 0;
    let timer = null;
    setChunks([]);
    setReadError(null);
    const read = () => getDeviceLog(device, day, offset)
      .then((r) => {
        if (!alive) return;
        const box = boxRef.current;
        stickRef.current = offset === 0 || !box || box.scrollHeight - box.scrollTop - box.clientHeight < 4;
        offset = r.offset;
        if (r.text) {
          const lines = r.text.replace(ANSI_RE, "").split("\n");
          if (lines[lines.length - 1] === "") lines.pop(); // a read ends on a whole line
          setChunks((c) => [...c, lines]);
        }
        setReadError(null);
      })
      .catch((e) => { if (alive) setReadError(e); })
      .finally(() => { if (alive && live) timer = setTimeout(read, POLL_MS); });
    read();
    return () => { alive = false; clearTimeout(timer); };
  }, [device, day, days]);

  useLayoutEffect(() => {
    const box = boxRef.current;
    if (box && stickRef.current) box.scrollTop = box.scrollHeight;
  }, [chunks]);

  if (error) return <ErrorBox error={error} onRetry={loadDays} />;
  if (!days) return <Loading />;
  if (days.length === 0) return <div className="z-empty"><b>No log yet</b>The server keeps this speaker's log once "Device log" above is on and saved.</div>;
  // The text box takes the height its container leaves it (.z-logview, .z-logbox).
  return <div className="z-logview">
    <Field row label="Day"><div className="z-rowctl"><Select value={day} options={days} onChange={setDay} /></div></Field>
    {readError && <div className="z-fh" style={{ color: "var(--bad)", marginBottom: 6 }}>{readError.message}</div>}
    <div ref={boxRef} className="z-code z-logbox" data-testid="device-log-text">
      {chunks.map((lines, i) => <React.Fragment key={i}>
        {lines.map((l, j) => <div key={j} className="z-logline">{l}</div>)}</React.Fragment>)}</div>
  </div>;
}
