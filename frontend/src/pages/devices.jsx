import React, { useEffect, useState } from "react";
import { Ic } from "../components/icons.jsx";
import { Card, PageHeader, StatusPill, Pill } from "../components/primitives.jsx";
import DeviceModal from "../components/DeviceModal.jsx";
import { useAppData } from "../appData.jsx";
import { errorLines } from "../useStageForm.js";
import { getConfig } from "../api.js";
import { deviceLabel, deviceVersion, fmtUptime, fwVersion } from "../format.js";

// ── Devices ───────────────────────────────────────────────────────────────

export function Devices() {
  const { catalog, patch, system, refreshConfig } = useAppData();
  const devices = catalog.core.values.devices || [];

  // Statuses arrive live over the panel WS heartbeat (system.devices is refreshed
  // every second), so no one-shot fetch / polling is needed here.
  const live = system?.devices || [];
  const liveOf = (name) => live.find((d) => d.name === name) || null;
  const statusOf = (name) => {
    const m = liveOf(name);
    if (!m) return "off";
    return m.online ? "online" : "offline";
  };
  // A device connecting for the first time is added to the config on the server; the
  // heartbeat lists it before this page's config snapshot does, so re-read the config.
  const unlisted = live.filter((m) => !devices.some((d) => d.name === m.name)).map((m) => m.name).join(",");
  useEffect(() => {
    if (unlisted) refreshConfig().catch(() => {});
  }, [unlisted]);

  const [modal, setModal] = useState(null); // { mode:'add'|'edit', index }
  const [busyErr, setBusyErr] = useState(null);

  // Apply `change` to the server's CURRENT device list, fetched right before the
  // write, so a cutoff set from a slider or a device that connected (and was added)
  // while the page was open is not rolled back. Devices are matched by name.
  const saveList = async (change) => {
    setBusyErr(null);
    try {
      const fresh = await getConfig();
      await patch({ core: { devices: change(fresh.core?.devices || []) } });
      setModal(null);
    } catch (e) { setBusyErr(e); }
  };
  const onAdd = (d) => saveList((list) => [...list, d]);
  const onEdit = (name, edit) => saveList((list) => list.map((x) => (
    x.name === name ? { ...x, ...edit } : x
  )));
  const onDelete = (name) => {
    if (!window.confirm(`Delete device "${deviceLabel(devices, name)}"?`)) return;
    saveList((list) => list.filter((x) => x.name !== name));
  };

  return <div className="z-page">
    <PageHeader title="Devices" desc="An unknown device is added on its first connection. Each device id also keys its own dialog context."
      actions={<button className="z-btn g" onClick={() => setModal({ mode: "add" })}><Ic n="add" w={14} />Add device</button>} />
    {busyErr && <div className="z-banner warn" style={{ margin: "0 0 12px" }}><Ic n="restart" w={15} /><span>{errorLines(busyErr).join(" · ")}</span></div>}
    <Card>
      {/* On a narrow screen Firmware and Uptime hide (the dialog header has them) and what is
          still too wide scrolls inside the card, not the page. */}
      <div style={{ overflowX: "auto" }}>
      <table className="z-tbl z-devtbl">
        <thead><tr><th>Name</th><th>Firmware</th><th>Wake word model</th><th>Uptime</th><th>Status</th></tr></thead>
        <tbody>
          {devices.length === 0
            ? <tr><td colSpan={5} style={{ color: "var(--mut)", padding: "14px 0" }}>No devices configured.</td></tr>
            : devices.map((d, i) => {
              const enabled = d.enabled ?? true;
              const dim = enabled ? "" : "z-dim";
              const fw = deviceVersion(liveOf(d.name), "config_version");
              // Re-read the config first: the dialog starts from the device's saved label and
              // model, which may have changed elsewhere since this page loaded.
              return <tr key={i} data-testid="device-row" onClick={() => refreshConfig().catch(() => {})
                .finally(() => setModal({ mode: "edit", name: d.name }))}>
                <td className={dim} style={{ fontWeight: 600 }}>{d.label || d.name}</td>
                {/* Firmware/model versions come from the live WS snapshot; "—" while offline/disabled.
                    The firmware's build hash is in its tooltip. */}
                <td className={`mono ${dim}`} style={{ fontSize: 11.5 }} title={fw ?? undefined}>{fwVersion(fw) ?? "—"}</td>
                <td className={`mono ${dim}`} style={{ fontSize: 11.5 }}>{deviceVersion(liveOf(d.name), "model_version") ?? "—"}</td>
                <td className={`mono ${dim}`} style={{ fontSize: 11.5 }}>{fmtUptime(liveOf(d.name)?.uptime_seconds)}</td>
                <td className={dim}>{enabled ? <StatusPill status={statusOf(d.name)} /> : <Pill tone="muted">disabled</Pill>}</td>
              </tr>;
            })}
        </tbody>
      </table>
      </div>
    </Card>
    {modal?.mode === "add" && <DeviceModal title="Add device" onSave={onAdd} onClose={() => setModal(null)} />}
    {modal?.mode === "edit" && <DeviceModal key={modal.name} title="Edit device" onClose={() => setModal(null)}
      device={modal.name} label={devices.find((d) => d.name === modal.name)?.label || ""}
      enabled={devices.find((d) => d.name === modal.name)?.enabled ?? true}
      wakeModel={devices.find((d) => d.name === modal.name)?.wake_model || ""}
      log={!!devices.find((d) => d.name === modal.name)?.log} logDays={catalog.core.values.device_log_days}
      autoVolume={!!devices.find((d) => d.name === modal.name)?.auto_volume}
      compressReplies={!!devices.find((d) => d.name === modal.name)?.compress_replies}
      onSave={(edit) => onEdit(modal.name, edit)} onDelete={() => onDelete(modal.name)}
      online={statusOf(modal.name) === "online"} />}
  </div>;
}
