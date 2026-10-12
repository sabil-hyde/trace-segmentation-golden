"""A seeded session joined with its generated text: the one view verify, review and export share."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from golden.vendor.harness import Turn

from golden.seeds import SegmentSeed, SessionSeed, read_seeds


@dataclass
class Session:
    seed: SessionSeed
    generation: Dict                     # one line of generations.jsonl

    @property
    def ok(self) -> bool:
        return all(g["status"] == "ok" for g in self.generation["segments"])

    def segment_turns(self) -> List[tuple]:
        """[(SegmentSeed, [{"user", "assistant"}, ...])] in session order."""
        return [(s, g["turns"]) for s, g in zip(self.seed.segments, self.generation["segments"])]

    def turns(self) -> List[Dict[str, str]]:
        return [t for _, turns in self.segment_turns() for t in turns]

    def gold_starts(self) -> List[int]:
        """Index of the first user turn of every segment (0 is always a start)."""
        starts, k = [], 0
        for segment in self.seed.segments:
            starts.append(k)
            k += segment.n_turns
        return starts

    def harness_turns(self) -> List[Turn]:
        """The turns in trace-mining's Turn shape (vendor/harness.py), so existing renderers apply unchanged."""
        return [Turn(i, t["user"], 0, replies=[t["assistant"]]) for i, t in enumerate(self.turns())]


def load_sessions(data_dir: Path, only_ok: bool = True) -> List[Session]:
    """Seeded sessions that have a generation, in seed order."""
    path = Path(data_dir) / "generations.jsonl"
    generations = {json.loads(l)["session_id"]: json.loads(l) for l in path.read_text().splitlines()} if path.exists() else {}
    sessions = [Session(s, generations[s.session_id]) for s in read_seeds(Path(data_dir) / "seeds.jsonl")
                if s.session_id in generations]
    return [s for s in sessions if s.ok] if only_ok else sessions


def read_jsonl_by_id(path: Path, key: str = "session_id") -> Dict[str, Dict]:
    """Last row per id wins, so re-running a step simply appends a newer answer."""
    if not Path(path).exists():
        return {}
    return {row[key]: row for row in map(json.loads, Path(path).read_text().splitlines())}
