"""Load an exported golden dataset for one task, filtered to a split and optionally sampled."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FILES = {"seg": "seg.jsonl", "tax": "tax.jsonl"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stratified_sample(rows: List[Dict], n: int, seed: int, key) -> List[Dict]:
    """n rows keeping each key's share (e.g. the ~9% boundary base rate), in a seeded, stable order."""
    rng = random.Random(seed)
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(str(key(r)), []).append(r)
    picked = []
    for _, members in sorted(groups.items()):
        take = round(n * len(members) / len(rows))
        picked += rng.sample(members, min(take, len(members)))
    return sorted(picked, key=lambda r: r["id"])


def load(export_dir: Path, task: str, split: str = "eval", sample: Optional[int] = None,
         seed: int = 0) -> Tuple[List[Dict], Dict]:
    path = Path(export_dir) / FILES[task]
    if not path.exists():
        raise SystemExit(f"{path} not found: run `python -m golden.export --unreviewed` (or the reviewed export) first")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    if sample and sample < len(rows):
        key = (lambda r: r["gold"]) if task == "seg" else (lambda r: r["gold"]["industry"])
        rows = stratified_sample(rows, sample, seed, key)
    statuses = sorted({r.get("review_status", "unknown") for r in rows})
    info = {"path": str(path), "sha256": sha256(path), "split": split, "sample": sample, "seed": seed,
            "n": len(rows), "review_status": statuses}
    return rows, info
