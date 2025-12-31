import argparse
import json
import os
import time
import uuid
from pathlib import Path


def load_index(index_path: str):
    records = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"Index file is empty: {index_path}")
    return records


def simulate_max_up_to_30(index_records, out_dir: str):
    """
    Adaptive replay: as fast as possible, capped to dataset time (30 FPS base).
    If processing is faster than real time, it sleeps to not exceed the schedule.
    If slower, it never sleeps and effective FPS drops naturally.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)  # fail if already exists (immutability)

    ticks_path = out_dir / "sim_ticks.jsonl"
    metrics_path = out_dir / "metrics.json"

    t0_wall = time.perf_counter()
    latencies = []
    processed = 0

    with open(ticks_path, "w", encoding="utf-8") as out:
        for rec in index_records:
            t_data = float(rec["t"])
            target_wall = t0_wall + t_data

            now = time.perf_counter()
            if now < target_wall:
                time.sleep(target_wall - now)

            t_emit = time.perf_counter()
            latency = max(0.0, t_emit - target_wall)
            latencies.append(latency)

            tick = {
                "tick": processed,
                "t_capture": t_data,
                "frame_idx": rec["frame_idx"],
                "rgb_path": rec["rgb_path"],
                "event_path": rec["event_path"],
                "gt": rec.get("gt", []),
            }
            out.write(json.dumps(tick) + "\n")
            processed += 1

    total_wall = time.perf_counter() - t0_wall
    achieved_fps = processed / total_wall if total_wall > 0 else 0.0
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000 if latencies else 0.0

    metrics = {
        "mode": "max_up_to_30",
        "frames": processed,
        "wall_seconds": total_wall,
        "achieved_fps": achieved_fps,
        "avg_latency_ms": avg_latency_ms,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] run_id={out_dir.name}")
    print(f"[OK] Run written: {out_dir}")
    print(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to index jsonl")
    ap.add_argument("--runs_root", default=os.path.join("data", "runs"), help="Runs output folder")
    ap.add_argument(
        "--run_id",
        default=None,
        help="Optional run id (folder name). If omitted, a random one is generated.",
    )
    args = ap.parse_args()

    records = load_index(args.index)

    run_id = args.run_id or str(uuid.uuid4())[:8]
    run_dir = Path(args.runs_root) / run_id

    # Best policy: never overwrite, never append
    if run_dir.exists():
        raise RuntimeError(
            f"Run '{run_id}' already exists at {run_dir}. "
            "Create a new run_id or delete the old run folder."
        )

    simulate_max_up_to_30(records, str(run_dir))


if __name__ == "__main__":
    main()
