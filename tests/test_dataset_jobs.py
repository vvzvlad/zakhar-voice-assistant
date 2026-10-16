"""Wake-word dataset recording over the panel API: POST /api/datasets/{name} runs a
beep-then-capture session in the background, writing numbered clips under
<DATA_DIR>/datasets/<name>/; GET /api/datasets lists the sets, .../stop ends a session,
.../{name}.zip downloads a set."""

import asyncio
import io
import zipfile

import pytest

from tests.test_panel_api import _client, _wav_bytes


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config_store.DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


class FakeDevice:
    """Stands in for DeviceManager.play_chime/capture; logs the order of beeps and captures.

    Capture N returns a WAV whose PCM is byte N, so every clip is distinguishable. A capture
    raises `fail` once `fail_at` captures have succeeded; it waits for `gate` when set.
    """

    def __init__(self, fail_at=None, fail=None, gate=None):
        self.calls = []
        self.clips = []
        self.fail_at, self.fail, self.gate = fail_at, fail, gate

    async def play(self, sound_path, device):
        self.calls.append(("beep", sound_path, device))
        return {"played": [device], "offline": []}

    async def capture(self, device, seconds):
        self.calls.append(("capture", device, seconds))
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_at is not None and len(self.clips) == self.fail_at:
            raise self.fail
        self.clips.append(_wav_bytes(pcm=bytes([len(self.clips) + 1]) * 32))
        return self.clips[-1]


async def _wired(tmp_path, fake):
    return await _client(tmp_path, device_capture=fake.capture, device_play=fake.play)


async def _set(client, name):
    sets = (await (await client.get("/api/datasets")).json())["sets"]
    return next((s for s in sets if s["name"] == name), None)


async def _poll(client, name, want, *, tries=200):
    """Poll GET /api/datasets until the set's job reaches state `want`; return the set."""
    for _ in range(tries):
        s = await _set(client, name)
        if s and s["job"] and s["job"]["state"] == want:
            return s
        await asyncio.sleep(0)
    raise AssertionError(f"set {name!r} never reached {want!r}: {s}")


def _clips(data_dir, name):
    d = data_dir / "datasets" / name
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


async def test_session_records_numbered_clips_and_next_session_continues(tmp_path, _data_dir):
    fake = FakeDevice()
    client, _ = await _wired(tmp_path, fake)
    try:
        resp = await client.post("/api/datasets/захар", json={"device": "hall", "takes": 3})
        assert resp.status == 202
        assert await resp.json() == {"state": "recording", "device": "hall", "takes": 3,
                                     "seconds": 3, "done": 0, "error": None}
        await _poll(client, "захар", "done")
        # The device beeps (synthesized chime) before every capture.
        assert fake.calls == [("beep", "", "hall"), ("capture", "hall", 3)] * 3
        assert _clips(_data_dir, "захар") == {
            "0001.wav": fake.clips[0], "0002.wav": fake.clips[1], "0003.wav": fake.clips[2]}

        # A later session extends the set after its highest clip.
        resp = await client.post("/api/datasets/захар",
                                 json={"device": "hall", "takes": 2, "seconds": 5})
        assert resp.status == 202
        s = await _poll(client, "захар", "done")
        assert s == {"name": "захар", "clips": 5, "job": {
            "state": "done", "device": "hall", "takes": 2, "seconds": 5, "done": 2, "error": None}}
        assert sorted(_clips(_data_dir, "захар")) == [f"000{i}.wav" for i in range(1, 6)]
        assert _clips(_data_dir, "захар")["0005.wav"] == fake.clips[4]
    finally:
        await client.close()


async def test_list_counts_sets_on_disk_without_a_job(tmp_path, _data_dir):
    old = _data_dir / "datasets" / "old"
    old.mkdir(parents=True)
    for f in ("0001.wav", "0007.wav"):
        (old / f).write_bytes(_wav_bytes())
    client, _ = await _wired(tmp_path, FakeDevice())
    try:
        assert await (await client.get("/api/datasets")).json() == {
            "sets": [{"name": "old", "clips": 2, "job": None}]}
        await client.post("/api/datasets/old", json={"device": "hall", "takes": 1})
        await _poll(client, "old", "done")
        assert "0008.wav" in _clips(_data_dir, "old")
    finally:
        await client.close()


async def test_failing_take_ends_job_with_error_and_keeps_clips(tmp_path, _data_dir):
    fake = FakeDevice(fail_at=2, fail=RuntimeError("hall is offline"))
    client, _ = await _wired(tmp_path, fake)
    try:
        await client.post("/api/datasets/zakhar", json={"device": "hall", "takes": 5})
        s = await _poll(client, "zakhar", "error")
        assert s["clips"] == 2
        assert s["job"]["done"] == 2 and "offline" in s["job"]["error"]
        assert sorted(_clips(_data_dir, "zakhar")) == ["0001.wav", "0002.wav"]
        # No retries: the session stopped at the failing take.
        assert [c for c in fake.calls if c[0] == "capture"] == [("capture", "hall", 3)] * 3
    finally:
        await client.close()


async def test_stop_ends_after_current_take(tmp_path, _data_dir):
    fake = FakeDevice(gate=asyncio.Event())
    client, _ = await _wired(tmp_path, fake)
    try:
        await client.post("/api/datasets/zakhar", json={"device": "hall", "takes": 5})
        for _ in range(200):  # wait until the first capture is in flight
            if fake.calls[-1:] == [("capture", "hall", 3)]:
                break
            await asyncio.sleep(0)
        resp = await client.post("/api/datasets/zakhar/stop")
        assert resp.status == 200
        assert (await resp.json())["state"] == "recording"  # the current take still finishes
        fake.gate.set()
        s = await _poll(client, "zakhar", "stopped")
        assert s["clips"] == 1 and s["job"]["done"] == 1
        assert (await client.post("/api/datasets/zakhar/stop")).status == 404
    finally:
        fake.gate.set()
        await client.close()


async def test_zip_downloads_the_set(tmp_path, _data_dir):
    fake = FakeDevice()
    client, _ = await _wired(tmp_path, fake)
    try:
        await client.post("/api/datasets/захар", json={"device": "hall", "takes": 2})
        await _poll(client, "захар", "done")
        resp = await client.get("/api/datasets/захар.zip")
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "application/zip"
        # Non-ASCII set name is sanitized for the header.
        assert resp.headers["Content-Disposition"] == 'attachment; filename="_____.zip"'
        with zipfile.ZipFile(io.BytesIO(await resp.read())) as zf:
            assert {i.filename: zf.read(i) for i in zf.infolist()} == {
                "0001.wav": fake.clips[0], "0002.wav": fake.clips[1]}
            assert {i.compress_type for i in zf.infolist()} == {zipfile.ZIP_STORED}
        assert (await client.get("/api/datasets/missing.zip")).status == 404
    finally:
        await client.close()


async def test_bad_input_returns_400(tmp_path, _data_dir):
    fake = FakeDevice()
    client, _ = await _wired(tmp_path, fake)
    try:
        for name in ("bad.name", "a" * 65, "two%20words"):
            resp = await client.post(f"/api/datasets/{name}", json={"device": "h", "takes": 1})
            assert resp.status == 400, name
        for body in ({"takes": 1},
                     {"device": "h"},
                     {"device": "h", "takes": 0},
                     {"device": "h", "takes": 1001},
                     {"device": "h", "takes": True},
                     {"device": "h", "takes": "5"},
                     {"device": "h", "takes": 1, "seconds": 0},
                     {"device": "h", "takes": 1, "seconds": 11},
                     {"device": "h", "takes": 1, "seconds": True},
                     [1]):
            resp = await client.post("/api/datasets/zakhar", json=body)
            assert resp.status == 400, body
        assert fake.calls == []
    finally:
        await client.close()


async def test_second_session_on_same_device_returns_409(tmp_path, _data_dir):
    fake = FakeDevice(gate=asyncio.Event())
    client, _ = await _wired(tmp_path, fake)
    try:
        assert (await client.post("/api/datasets/a", json={"device": "hall", "takes": 2})).status == 202
        resp = await client.post("/api/datasets/b", json={"device": "hall", "takes": 2})
        assert resp.status == 409
        assert "in progress" in (await resp.json())["error"]
    finally:
        fake.gate.set()
        await client.close()


async def test_not_wired_returns_503(tmp_path):
    client, _ = await _client(tmp_path)
    try:
        assert (await client.get("/api/datasets")).status == 503
        assert (await client.post("/api/datasets/zakhar", json={"device": "h", "takes": 1})).status == 503
    finally:
        await client.close()
