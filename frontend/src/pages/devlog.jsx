// Device Log — the speakers' own logs the server keeps on disk, one file per device and day
// (src/device_logs.py). Pick a device and a day to read it; the device's newest day may still be
// growing, so while it is shown it is polled and the new lines are appended.
import React, { useState, useEffect, useLayoutEffect, useRef } from "react";
import { Ic } from "../components/icons.jsx";
import { PageHeader, Select, Loading, ErrorBox } from "../components/primitives.jsx";
import { getDeviceLogs, getDeviceLog } from "../api.js";
import { useDeviceList } from "../appData.jsx";
import { deviceLabel } from "../format.js";

const POLL_MS = 2000;
// ESP-IDF colours its log lines with ANSI escapes; the page shows the plain text.
const ANSI_RE = /\x1b\[[0-9;]*m/g;

function DeviceLog() {
  const devices = useDeviceList();
  const [days, setDays] = useState(null); // {device: [YYYY-MM-DD, newest first]} once loaded
  const [error, setError] = useState(null);
  const [device, setDevice] = useState("");
  const [day, setDay] = useState("");
  // One text node per read: appending a node leaves the earlier ones, and a selection in them,
  // untouched, where a single growing string would be rewritten and the selection lost.
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
        const first = Object.keys(r.devices)[0];
        setDays(r.devices);
        if (first) { setDevice(first); setDay(r.devices[first][0]); }
      })
      .catch(setError);
  };
  useEffect(() => { loadDays(); }, []);

  // Read the chosen day from its start; the device's newest day is then re-read every POLL_MS
  // from where the last read stopped, each read bringing whole lines only.
  useEffect(() => {
    if (!device || !day) return undefined;
    const live = days[device][0] === day;
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
        if (r.text) setChunks((c) => [...c, r.text.replace(ANSI_RE, "")]);
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

  const ids = days ? Object.keys(days) : [];
  const pickDevice = (id) => { setDevice(id); setDay(days[id][0]); };

  // The page fills the viewport under the 50px top bar, so the text block takes the height left.
  return <div className="z-page" style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 50px)" }}>
    <PageHeader title="Device Log"
      desc={'The speakers\' own logs, kept on the server while "Device log" is on in a device\'s dialog on the Devices page.'} />
    {error ? <div className="z-card"><ErrorBox error={error} onRetry={loadDays} /></div>
      : !days ? <div className="z-card"><Loading /></div>
        : ids.length === 0 ? <div className="z-card"><div className="z-empty"><div className="ic"><Ic n="log" w={20} /></div>
          <b>No device logs yet</b>A speaker's log is kept once "Device log" is switched on in its dialog on the Devices page.</div></div>
          : <>
            <div style={{ display: "flex", gap: 10, marginBottom: 10 }}>
              <Select w={240} value={device} options={ids.map((id) => ({ value: id, label: deviceLabel(devices, id) }))}
                onChange={pickDevice} />
              <Select w={150} value={day} options={days[device]} onChange={setDay} />
            </div>
            {readError && <div className="z-fh" style={{ color: "var(--bad)", marginBottom: 6 }}>{readError.message}</div>}
            <div ref={boxRef} className="z-code" data-testid="device-log-text"
              style={{ flex: 1, minHeight: 0, margin: 0, overflow: "auto" }}>
              {chunks.map((c, i) => <React.Fragment key={i}>{c}</React.Fragment>)}</div>
          </>}
  </div>;
}

export default DeviceLog;
