from __future__ import annotations

import io
import json
import mimetypes
import os
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image

from .model_runner import ModelRunner

# ---------- Paths ----------
APP_ROOT = Path(__file__).resolve().parents[2]  # project root (.../drones-project)
RUNS_DIR = APP_ROOT / "data" / "runs"
MODELS_DIR = APP_ROOT / "data" / "models"
ACTIVE_MODEL_FILE = MODELS_DIR / "active_model.txt"

app = FastAPI(title="Drone Simulation Platform")

# Helpful now; tighten later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from natsort import natsorted  # add import

def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    """
    Safely extract a zip file to dest, preventing path traversal.
    """
    dest = dest.resolve()
    for member in zf.infolist():
        member_path = (dest / member.filename).resolve()
        if dest not in member_path.parents and dest != member_path:
            raise HTTPException(status_code=400, detail="Invalid zip entry path")
    zf.extractall(dest)

def _safe_resolve_under(root: Path, user_path: str) -> Path:
    """
    Resolve user-provided path safely under root.
    """
    p = Path(user_path)
    if not p.is_absolute():
        p = (APP_ROOT / p).resolve()
    else:
        p = p.resolve()

    root = root.resolve()
    if root == p or root in p.parents:
        return p
    raise HTTPException(status_code=400, detail="Invalid path (outside allowed root)")

def _safe_resolve_frame(user_path: str) -> Path:
    """
    Resolve a frame path under allowed root (runs only).
    """
    return _safe_resolve_under(RUNS_DIR, user_path)


def _to_timestamp(frame_idx: int) -> str:
    return str(float(f"{(frame_idx + 1) * 0.033333:.6f}"))


def _read_coordinates(coords_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """
    Tolerant read:
    timestamp: x1, y1, x2, y2[, idx][, class_name]
    Keeps absolute timestamps in milliseconds for nearest-neighbor lookup.
    """
    coords: Dict[int, List[Dict[str, Any]]] = {}
    if not coords_path.exists():
        return coords

    next_id = 1
    for line in coords_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ": " not in line:
            continue
        try:
            ts_raw_str, rest = line.split(": ", 1)
            ts_ms = int(round(float(ts_raw_str) * 1000))
            parts = [p.strip() for p in rest.split(",")]
            if len(parts) < 4:
                continue
            x1, y1, x2, y2 = map(float, parts[:4])
            if len(parts) >= 5 and parts[4]:
                track_id = int(float(parts[4]))
            else:
                track_id = next_id
                next_id += 1
            cls = parts[5] if len(parts) >= 6 and parts[5] else "drone"
        except Exception:
            continue
        coords.setdefault(ts_ms, []).append(
            {"id": track_id, "bbox_xyxy": [x1, y1, x2, y2], "cls": cls}
        )
    return coords


def _compute_letterbox_meta(padded_frame: Path, target_size: tuple[int, int]) -> Optional[Dict[str, float]]:
    """
    Detect padding on a padded frame and return offsets/scale to map padded coords -> target coords.
    """
    if not padded_frame.exists():
        return None
    with Image.open(padded_frame) as im:
        bbox = im.convert("RGB").getbbox()
        if not bbox:
            return None
        left, top, right, bottom = bbox
        content_w = right - left
        content_h = bottom - top
        target_w, target_h = target_size
        if content_w <= 0 or content_h <= 0:
            return None
        sx = content_w / target_w
        sy = content_h / target_h
        if sx == 0 or sy == 0:
            return None
        return {"left": float(left), "top": float(top), "sx": float(sx), "sy": float(sy)}


def _map_bbox_from_padded(bbox: List[float], meta: Dict[str, float]) -> List[float]:
    x1, y1, x2, y2 = bbox
    left = meta["left"]
    top = meta["top"]
    sx = meta["sx"]
    sy = meta["sy"]
    return [
        (x1 - left) / sx,
        (y1 - top) / sy,
        (x2 - left) / sx,
        (y2 - top) / sy,
    ]


def _generate_ticks_for_run(run_dir: Path) -> None:
    """
    Build sim_ticks.jsonl from RGB/Event/coordinates in run_dir.
    Requires run_dir/RGB, run_dir/Event/Frames, run_dir/coordinates.txt at root.
    """
    ticks_path = run_dir / "sim_ticks.jsonl"
    if ticks_path.exists():
        return

    rgb_dir = run_dir / "RGB"
    event_dir = run_dir / "Event" / "Frames"
    coords_path = run_dir / "coordinates.txt"
    padded_dir = run_dir / "PADDED_RGB"

    if not (rgb_dir.is_dir() and event_dir.is_dir() and coords_path.exists()):
        raise HTTPException(status_code=400, detail="sim_ticks.jsonl missing and could not be generated (RGB/Event/coordinates missing)")

    rgb_paths = natsorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
    event_paths = natsorted(list(event_dir.glob("*.png")) + list(event_dir.glob("*.jpg")))
    if not rgb_paths or not event_paths:
        raise HTTPException(status_code=400, detail="No RGB/Event frames found to generate sim_ticks.jsonl")

    n = min(len(rgb_paths), len(event_paths))
    coords = _read_coordinates(coords_path)
    coords_keys = sorted(coords.keys())

    with Image.open(rgb_paths[0]) as img0:
        target_w, target_h = img0.size

    letterbox_meta = None
    if padded_dir.is_dir():
        padded_sample = padded_dir / Path(rgb_paths[0]).name
        letterbox_meta = _compute_letterbox_meta(padded_sample, (target_w, target_h))

    def _nearest_coords(t_ms: int, tol_ms: int = 50) -> List[Dict[str, Any]]:
        best = None
        best_delta = None
        for k in coords_keys:
            delta = abs(k - t_ms)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = k
        if best_delta is not None and best_delta <= tol_ms:
            return coords.get(best, [])
        return []

    with ticks_path.open("w", encoding="utf-8") as out:
        for i in range(n):
            ts = float(_to_timestamp(i))
            t_ms = int(round(ts * 1000))
            gt_raw = _nearest_coords(t_ms)
            if letterbox_meta and gt_raw:
                gt = []
                for g in gt_raw:
                    bb = g.get("bbox_xyxy")
                    if not bb or len(bb) != 4:
                        continue
                    mapped = _map_bbox_from_padded(bb, letterbox_meta)
                    gt.append({**g, "bbox_xyxy": mapped})
            else:
                gt = gt_raw
            rec = {
                "tick": i,
                "t_capture": float(f"{ts:.6f}"),
                "frame_idx": i,
                "rgb_path": os.path.relpath(rgb_paths[i].resolve(), APP_ROOT),
                "event_path": os.path.relpath(event_paths[i].resolve(), APP_ROOT),
                "gt": gt,
            }
            out.write(json.dumps(rec) + "\n")

    # Optional: enforce GT presence; uncomment to fail uploads with zero GT
    # if sum(1 for _i in range(n) if coords.get(_to_timestamp(_i))) == 0:
    #     raise HTTPException(status_code=400, detail="GT missing: coordinates had no usable entries")


# ---------- API ----------
@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    out = []
    for p in sorted(RUNS_DIR.iterdir()):
        if p.is_dir():
            out.append(
                {
                    "run_id": p.name,
                    "has_ticks": (p / "sim_ticks.jsonl").exists(),
                    "has_metrics": (p / "metrics.json").exists(),
                }
            )
    return out


@app.get("/api/runs/{run_id}/metrics")
def get_metrics(run_id: str):
    mp = RUNS_DIR / run_id / "metrics.json"
    if not mp.exists():
        raise HTTPException(404, "metrics.json not found")
    return json.loads(mp.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/ticks")
def get_ticks(run_id: str, offset: int = 0, limit: int = 600):
    """
    Paginated ticks from sim_ticks.jsonl
    """
    ticks_path = RUNS_DIR / run_id / "sim_ticks.jsonl"
    if not ticks_path.exists():
        raise HTTPException(404, "sim_ticks.jsonl not found")

    lines = ticks_path.read_text(encoding="utf-8").splitlines()
    chunk = lines[offset : offset + limit]
    return {
        "run_id": run_id,
        "offset": offset,
        "limit": limit,
        "count": len(chunk),
        "total": len(lines),
        "items": [json.loads(x) for x in chunk if x.strip()],
    }


@app.get("/api/frame")
def frame(path: str = Query(..., description="Path to an RGB/Event frame file")):
    """
    Serves BOTH RGB and Event frame files.
    """
    p = _safe_resolve_frame(path)
    if not p.exists():
        raise HTTPException(404, "Frame not found")
    mime, _ = mimetypes.guess_type(str(p))
    return FileResponse(str(p), media_type=mime or "application/octet-stream")


@app.post("/api/models")
async def upload_model(model: UploadFile = File(...)):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_id = uuid.uuid4().hex[:8]
    model_dir = MODELS_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(model.filename or "").suffix or ".onnx"
    weights_path = model_dir / f"model{suffix}"
    data = await model.read()
    weights_path.write_bytes(data)
    return {"model_id": model_id, "filename": model.filename}


@app.get("/api/models")
def list_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = [p.name for p in MODELS_DIR.iterdir() if p.is_dir()]
    active = ACTIVE_MODEL_FILE.read_text().strip() if ACTIVE_MODEL_FILE.exists() else None
    return {"models": models, "active": active}


@app.post("/api/models/{model_id}/activate")
def activate_model(model_id: str):
    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        raise HTTPException(404, "model not found")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_MODEL_FILE.write_text(model_id)
    return {"active": model_id}


@app.post("/api/runs/{run_id}/run-model")
def run_model_on_run(run_id: str, source: str = Query("model", pattern="^(model|gt)$")):
    # Load active model or use ground truth
    use_gt = source == "gt"
    if use_gt:
        model_id = "ground_truth"
        runner = None
    else:
        if not ACTIVE_MODEL_FILE.exists():
            raise HTTPException(400, "no active model; upload and activate first")
        model_id = ACTIVE_MODEL_FILE.read_text().strip()
        runner = ModelRunner(MODELS_DIR)
        try:
            runner.load(model_id)
        except FileNotFoundError:
            raise HTTPException(400, f"weights for model {model_id} not found")

    ticks_path = RUNS_DIR / run_id / "sim_ticks.jsonl"
    if not ticks_path.exists():
        raise HTTPException(404, "sim_ticks.jsonl not found")

    out_path = RUNS_DIR / run_id / f"predictions_{model_id}.jsonl"
    count = 0

    with ticks_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            tick = json.loads(line)
            try:
                rgb_path = _safe_resolve_frame(tick.get("rgb_path", ""))
            except Exception:
                rgb_path = None
            try:
                event_path = _safe_resolve_frame(tick.get("event_path", ""))
            except Exception:
                event_path = None

            if rgb_path is None and event_path is None:
                continue  # skip if no inputs

            if use_gt:
                preds = {"model_id": "ground_truth", "detections": tick.get("gt", [])}
            else:
                preds = runner.predict(rgb_path, event_path)

            tick_out = {"tick": tick.get("tick"), "t_capture": tick.get("t_capture"), "predictions": preds}
            f_out.write(json.dumps(tick_out) + "\n")
            count += 1

    return {"run_id": run_id, "model_id": model_id, "predictions_file": str(out_path), "processed": count}


@app.post("/api/scenarios")
async def upload_scenario(scenario: UploadFile = File(...)):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    base_name = Path(scenario.filename or "scenario").stem or "scenario"
    scenario_id = f"{base_name}-{uuid.uuid4().hex[:6]}"
    scenario_dir = RUNS_DIR / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    data = await scenario.read()
    # Persist original zip for provenance/replay.
    (scenario_dir / "source.zip").write_bytes(data)
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            _safe_extract_zip(zf, scenario_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")

    # Require root layout if sim_ticks.jsonl is absent
    ticks_path = scenario_dir / "sim_ticks.jsonl"
    if not ticks_path.exists():
        if not ((scenario_dir / "RGB").is_dir() and (scenario_dir / "Event" / "Frames").is_dir() and (scenario_dir / "coordinates.txt").exists()):
            raise HTTPException(status_code=400, detail="sim_ticks.jsonl missing and RGB/Event/coordinates not found at zip root")
        _generate_ticks_for_run(scenario_dir)

    return {"scenario_id": scenario_id, "path": str(scenario_dir)}


# ---------- Plain HTML Player (Side-by-side RGB + Event) ----------
PLAYER_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Simulated Run Player</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    button, select, input { padding: 6px 10px; }
    button { cursor:pointer; }
    .small { color:#555; font-size:13px; }
    .pill { padding: 4px 10px; background:#f2f2f2; border-radius:999px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
    .panel { border: 1px solid #ccc; border-radius: 8px; padding: 10px; }
    .panel h3 { margin: 0 0 8px 0; font-size: 14px; color: #333; }
    canvas { width: 100%; height: auto; display:block; border: 1px solid #ddd; border-radius: 6px; }
    #json { white-space: pre-wrap; margin-top: 10px; }
    .hint { color:#777; font-size:12px; }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h2>Simulated Run Player (RGB + Event)</h2>

  <div class="row">
    <button id="refresh">Refresh runs</button>
    <label>Run:</label>
    <select id="runs"></select>
    <button id="load">Load</button>

    <span class="pill" id="status">idle</span>
    <span class="small" id="info"></span>
  </div>

  <div class="row" style="margin-top:10px;">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <button id="step">Step</button>
    <button id="nextGt">Next GT</button>

    <label>Speed:</label>
    <select id="speed">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>

    <label>Chunk:</label>
    <input id="chunk" type="number" min="100" max="3000" value="600" />

    <label class="hint">Show GT:</label>
    <input id="showGt" type="checkbox" checked />
  </div>

  <div class="row" style="margin-top:10px; flex-wrap:wrap;">
    <label>Scenario zip:</label>
    <input type="file" id="scenarioFile" />
    <button id="uploadScenario">Upload scenario</button>
  </div>

  <div class="row" style="margin-top:10px; flex-wrap:wrap;">
    <label>Model file:</label>
    <input type="file" id="modelFile" />
    <button id="uploadModel">Upload</button>
    <button id="refreshModels">Refresh models</button>

    <label>Model:</label>
    <select id="models"></select>
    <button id="activateModel">Activate</button>
    <button id="runModel">Run model on run</button>
    <button id="runGt">Use GT predictions</button>

    <span class="pill">Active: <span id="activeModel">none</span></span>
  </div>

  <div class="grid">
    <div class="panel">
      <h3>RGB Camera</h3>
      <canvas id="cvRgb"></canvas>
    </div>
    <div class="panel">
      <h3>Event Camera</h3>
      <canvas id="cvEvent"></canvas>
    </div>
  </div>

  <pre class="small" id="json"></pre>

<script>
  const runsEl = document.getElementById("runs");
  const statusEl = document.getElementById("status");
  const infoEl = document.getElementById("info");
  const jsonEl = document.getElementById("json");
  const speedEl = document.getElementById("speed");
  const chunkEl = document.getElementById("chunk");
  const showGtEl = document.getElementById("showGt");
  const scenarioFileEl = document.getElementById("scenarioFile");
  const modelFileEl = document.getElementById("modelFile");
  const modelsEl = document.getElementById("models");
  const activeModelEl = document.getElementById("activeModel");

  const cvRgb = document.getElementById("cvRgb");
  const ctxRgb = cvRgb.getContext("2d");
  const cvEvent = document.getElementById("cvEvent");
  const ctxEvent = cvEvent.getContext("2d");

  let runId = null;
  let ticks = [];
  let total = 0;
  let cursor = 0;
  let playing = false;
  let lastRenderedIdx = 0;

  function setStatus(s) { statusEl.textContent = s; }

  async function refreshRuns() {
    setStatus("loading runs...");
    const res = await fetch("/api/runs");
    const runs = await res.json();
    runsEl.innerHTML = "";
    runs.filter(r => r.has_ticks).forEach(r => {
      const opt = document.createElement("option");
      opt.value = r.run_id;
      opt.textContent = r.run_id;
      runsEl.appendChild(opt);
    });
    setStatus("runs loaded");
  }

  async function fetchChunk(offset, limit) {
    const res = await fetch(`/api/runs/${runId}/ticks?offset=${offset}&limit=${limit}`);
    return await res.json();
  }

  async function loadTicks(reset=true) {
    runId = runsEl.value;
    if (!runId) { alert("No runs found. Create or upload a scenario first."); return; }
    if (reset) { ticks = []; cursor = 0; total = 0; lastRenderedIdx = 0; }
    const chunk = parseInt(chunkEl.value || "600", 10);
    setStatus(`loading ticks (chunk=${chunk})...`);
    const first = await fetchChunk(0, chunk);
    ticks = first.items;
    total = first.total;
    setStatus(`loaded ${ticks.length}/${total}`);
    if (ticks.length > 0) {
      cursor = 0;
      lastRenderedIdx = 0;
      await renderTick(ticks[0], 0);
    }
  }

  async function ensureTicksAhead() {
    if (!runId) return;
    if (ticks.length >= total) return;
    if (cursor < ticks.length - 20) return;
    const chunk = parseInt(chunkEl.value || "600", 10);
    setStatus(`loading more ticks... (${ticks.length}/${total})`);
    const next = await fetchChunk(ticks.length, chunk);
    ticks = ticks.concat(next.items);
    total = next.total;
    setStatus(`loaded ${ticks.length}/${total}`);
  }

  async function loadImageFromTickPath(tickPath) {
    const url = `/api/frame?path=${encodeURIComponent(tickPath)}`;
    const img = new Image();
    img.src = url;
    try { await img.decode(); }
    catch (e) { await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = reject; }); }
    return img;
  }

  function drawGt(ctx, tick) {
    if (!showGtEl.checked) return;
    const gt = tick.gt || [];
    ctx.lineWidth = 3;
    ctx.strokeStyle = "lime";
    ctx.fillStyle = "lime";
    ctx.font = "16px Arial";
    for (const obj of gt) {
      const bb = obj.bbox_xyxy;
      if (!bb || bb.length !== 4) continue;
      const [x1, y1, x2, y2] = bb;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillText(`gt id=${obj.id}`, x1, Math.max(18, y1 - 6));
    }
  }

  async function drawRgb(tick) {
    if (!tick.rgb_path) {
      cvRgb.width = 640; cvRgb.height = 360;
      ctxRgb.clearRect(0, 0, cvRgb.width, cvRgb.height);
      ctxRgb.fillStyle = "#666";
      ctxRgb.font = "16px Arial";
      ctxRgb.fillText("No rgb_path in tick", 20, 30);
      return;
    }
    const img = await loadImageFromTickPath(tick.rgb_path);
    cvRgb.width = img.width; cvRgb.height = img.height;
    ctxRgb.clearRect(0, 0, cvRgb.width, cvRgb.height);
    ctxRgb.drawImage(img, 0, 0);
    drawGt(ctxRgb, tick);
  }

  async function drawEvent(tick) {
    if (!tick.event_path) {
      cvEvent.width = 640; cvEvent.height = 360;
      ctxEvent.clearRect(0, 0, cvEvent.width, cvEvent.height);
      ctxEvent.fillStyle = "#666";
      ctxEvent.font = "16px Arial";
      ctxEvent.fillText("No event_path in tick", 20, 30);
      return;
    }
    const img = await loadImageFromTickPath(tick.event_path);
    cvEvent.width = img.width; cvEvent.height = img.height;
    ctxEvent.clearRect(0, 0, cvEvent.width, cvEvent.height);
    ctxEvent.drawImage(img, 0, 0);
    drawGt(ctxEvent, tick);
  }

  async function renderTick(tick, idx) {
    if (!tick) return;
    try { await Promise.all([drawRgb(tick), drawEvent(tick)]); }
    catch (e) { setStatus("render error"); console.error(e); }
    const gtCount = (tick.gt || []).length;
    const hasE = tick.event_path ? "yes" : "no";
    const hasR = tick.rgb_path ? "yes" : "no";
    infoEl.textContent =
      `tick=${tick.tick} | t=${tick.t_capture.toFixed(3)}s | frame_idx=${tick.frame_idx} | gt=${gtCount} | rgb=${hasR} | event=${hasE} | loaded=${ticks.length}/${total}`;
    jsonEl.textContent = JSON.stringify(tick, null, 2);
    lastRenderedIdx = idx;
  }

  async function refreshModels() {
    setStatus("loading models...");
    const res = await fetch("/api/models");
    const data = await res.json();
    modelsEl.innerHTML = "";
    (data.models || []).forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m; opt.textContent = m;
      modelsEl.appendChild(opt);
    });
    activeModelEl.textContent = data.active || "none";
    setStatus("models loaded");
  }

  async function uploadModel() {
    const file = modelFileEl.files[0];
    if (!file) { alert("Select a model file first."); return; }
    const fd = new FormData();
    fd.append("model", file);
    setStatus("uploading model...");
    const res = await fetch("/api/models", { method: "POST", body: fd });
    const data = await res.json();
    setStatus(`uploaded model_id=${data.model_id}`);
    await refreshModels();
  }

  async function activateModel() {
    const id = modelsEl.value;
    if (!id) { alert("Select a model first."); return; }
    setStatus("activating...");
    await fetch(`/api/models/${id}/activate`, { method: "POST" });
    await refreshModels();
    setStatus(`active model=${id}`);
  }

  async function runModelOnRun() {
    if (!runId) { alert("Load a run first."); return; }
    setStatus("running model on run...");
    const res = await fetch(`/api/runs/${runId}/run-model`, { method: "POST" });
    const data = await res.json();
    setStatus(`predictions -> ${data.predictions_file || "done"}`);
  }

  async function runGtOnRun() {
    if (!runId) { alert("Load a run first."); return; }
    setStatus("copying GT as predictions...");
    const res = await fetch(`/api/runs/${runId}/run-model?source=gt`, { method: "POST" });
    const data = await res.json();
    setStatus(`gt predictions -> ${data.predictions_file || "done"}`);
  }

  async function uploadScenario() {
    const file = scenarioFileEl.files[0];
    if (!file) { alert("Select a scenario zip first."); return; }
    const fd = new FormData();
    fd.append("scenario", file);
    setStatus("uploading scenario...");
    const res = await fetch("/api/scenarios", { method: "POST", body: fd });
    if (!res.ok) {
      const msg = await res.text();
      setStatus("upload failed");
      alert(msg);
      return;
    }
    const data = await res.json();
    setStatus(`scenario uploaded: ${data.scenario_id}`);
    await refreshRuns();
  }

  async function loop() {
    if (!playing) return;
    await ensureTicksAhead();
    if (cursor >= ticks.length - 1) {
      playing = false;
      setStatus("end / waiting for more ticks");
      return;
    }
    const spd = parseFloat(speedEl.value || "1");
    const cur = ticks[cursor];
    const nxt = ticks[cursor + 1];
    setStatus("playing");
    await renderTick(cur, cursor);
    cursor += 1;
    const dt = Math.max(0.0, (nxt.t_capture - cur.t_capture) * 1000 / spd);
    setTimeout(loop, dt);
  }

  document.getElementById("refresh").onclick = refreshRuns;
  document.getElementById("load").onclick = () => loadTicks(true);
  document.getElementById("uploadScenario").onclick = uploadScenario;
  document.getElementById("refreshModels").onclick = refreshModels;
  document.getElementById("uploadModel").onclick = uploadModel;
  document.getElementById("activateModel").onclick = activateModel;
  document.getElementById("runModel").onclick = runModelOnRun;
  document.getElementById("runGt").onclick = runGtOnRun;

  document.getElementById("play").onclick = () => {
    if (!runId || ticks.length === 0) { alert("Load a run first."); return; }
    playing = true;
    loop();
  };
  document.getElementById("pause").onclick = () => { playing = false; setStatus("paused"); };
  document.getElementById("step").onclick = async () => {
    if (!runId || ticks.length === 0) { alert("Load a run first."); return; }
    await ensureTicksAhead();
    if (cursor >= ticks.length) return;
    playing = false;
    setStatus("step");
    await renderTick(ticks[cursor], cursor);
    cursor += 1;
  };
  document.getElementById("nextGt").onclick = async () => {
    if (!runId || ticks.length === 0) { alert("Load a run first."); return; }
    playing = false;
    setStatus("searching GT...");
    let i = cursor;
    while (true) {
      if (i >= ticks.length) {
        await ensureTicksAhead();
        if (i >= ticks.length) break;
      }
      const gt = ticks[i].gt || [];
      if (gt.length > 0) {
        cursor = i;
        setStatus("jumped to GT");
        await renderTick(ticks[cursor], cursor);
        return;
      }
      i += 1;
    }
    setStatus("no more GT found");
    alert("No more GT frames found (end reached).");
  };
  showGtEl.onchange = async () => {
    if (!ticks.length) return;
    const idx = Math.min(Math.max(0, lastRenderedIdx), ticks.length - 1);
    await renderTick(ticks[idx], idx);
  };

  refreshRuns();
  refreshModels();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(PLAYER_HTML)
