import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from natsort import natsorted
from PIL import Image


def to_timestamp(frame_idx: int) -> str:
    # repo: str(float("{:.6f}".format((idx+1)*0.033333)))
    return str(float(f"{(frame_idx + 1) * 0.033333:.6f}"))


def read_coordinates(coords_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    FRED coordinates.txt format (tolerant):
    timestamp: x1, y1, x2, y2[, idx][, class_name]
    Returns dict[timestamp_ms] -> list[{id,bbox_xyxy,cls}] keyed by absolute ms.
    """
    coords: Dict[int, List[Dict[str, Any]]] = {}
    if not os.path.exists(coords_path):
        return coords

    next_id = 1
    with open(coords_path, "r", encoding="utf-8") as f:
        for line in f:
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


def build_index(seq_root: str, seq_id: str, out_path: str):
    seq_root_path = Path(seq_root)
    rgb_dir = seq_root_path / "RGB"
    event_dir = seq_root_path / "Event" / "Frames"
    coords_path = seq_root_path / "coordinates.txt"
    padded_dir = seq_root_path / "PADDED_RGB"

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB folder: {rgb_dir}")
    if not event_dir.is_dir():
        raise FileNotFoundError(f"Missing Event/Frames folder: {event_dir}")

    rgb_paths = natsorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
    event_paths = natsorted(list(event_dir.glob("*.png")) + list(event_dir.glob("*.jpg")))

    if len(rgb_paths) == 0:
        raise FileNotFoundError(f"No RGB .jpg frames found in {rgb_dir}")
    if len(event_paths) == 0:
        raise FileNotFoundError(f"No Event .png frames found in {event_dir}")

    # They assume 1:1 alignment; keep safe by clipping to min length
    n = min(len(rgb_paths), len(event_paths))

    coords = read_coordinates(str(coords_path))
    coords_keys = sorted(coords.keys())

    with Image.open(rgb_paths[0]) as img0:
        target_w, target_h = img0.size

    letterbox_meta = None
    if padded_dir.is_dir():
        padded_sample = padded_dir / rgb_paths[0].name
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

    gt_frames = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for i in range(n):
            ts = float(to_timestamp(i))
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
            if gt:
                gt_frames += 1
            record = {
                "seq_id": seq_id,
                "frame_idx": i,
                "t": float(f"{ts:.6f}"),
                "rgb_path": os.path.normpath(str(rgb_paths[i])),
                "event_path": os.path.normpath(str(event_paths[i])),
                "gt": gt,
            }
            out.write(json.dumps(record) + "\n")

    print(f"[OK] Wrote index: {out_path}")
    print(f"     frames indexed: {n}")
    print(f"     gt frames (non-empty): {gt_frames}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_root", required=True, help="Path to sequence folder, e.g. data/fred/store/train/0")
    ap.add_argument("--seq_id", required=True, help="Sequence ID, e.g. train/0")
    ap.add_argument("--out", required=True, help="Output jsonl, e.g. data/fred/indexes/train_0.jsonl")
    args = ap.parse_args()

    build_index(args.seq_root, args.seq_id, args.out)
