"""Deterministic slot sampler: the gold labels of every session, fixed before any text exists.

    python -m golden.seeds --sessions 500 --seed 3 --min-per-length 40

Writes <data dir>/seeds.jsonl (one session per line) and prints coverage tables. Same arguments
-> byte-identical file. Later steps only add text; they never change a slot.

Sampling order per dataset:
  1. segments per session  balanced over 1..5
  2. segment lengths       every length 1..20 at least `min_per_length` times, the rest short;
                           packed into sessions under the MAX_SESSION_TURNS budget
  3. switch type per boundary   balanced pool (cross-industry is the smallest share)
  4. task                  least-used task that satisfies the switch constraint
  5. operation             the task's primary operation half the time, else any allowed one
  6. surface slots         per user turn: a length bucket (balanced within each segment, so first
                           and later turns share one length distribution) and a shape that depends
                           only on the length; per follow-up a kind (25% decoys); per boundary a
                           transition style. These keep boundaries from being visible on the surface.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from golden import DEFAULT_DATA_DIR
from golden.taxonomy import Task, load_compatibility, load_tasks

SEGMENTS_PER_SESSION = (1, 2, 3, 4, 5)
LENGTHS = range(1, 21)
SHORT_LENGTHS = range(1, 9)          # extra segments beyond the per-length minimum come from here
MAX_SESSION_TURNS = 40
SWITCH_SHARES = {"same_industry_other_task": 0.35, "same_task_new_object": 0.35, "cross_industry": 0.30}
PRIMARY_OPERATION_SHARE = 0.5

LENGTH_BUCKETS = (50, 100, 200, 400, 800)                         # target chars of a user turn
PASTED_SHARE = {50: 0.0, 100: 0.2, 200: 0.4, 400: 0.6, 800: 0.8}  # depends on length only, never position
FOLLOWUP_SHARES = {"decoy": 0.25, **{k: 0.75 / 7 for k in (
    "clarify", "provide_info", "correct", "refine", "confirm", "ask_detail", "paste_more")}}
TRANSITION_SHARES = {"explicit": 0.15, "no_marker": 0.35, "blended": 0.35, "misleading_cue": 0.15}

# Lead-in: the words a user turn starts with. Openers take the kind their transition style implies; follow-ups draw
# from the same shares and the same phrases, so the first words of a turn carry no signal about a boundary.
LEAD_IN_POOLS = {
    "none": ("",),
    "reaction": ("got it", "ok that works", "makes sense", "perfect, thanks", "looks good", "great", "ok", "thanks, that helps",
                 "understood", "alright", "nice", "fair enough", "cool, thanks", "right"),
    "cue": ("also,", "oh and", "and one more thing -", "one more thing:", "quick one -", "oh, also", "and actually",
            "another thing:", "plus,", "btw,", "while I'm at it,", "and,"),
    "marker": ("different question:", "unrelated, but", "separate thing:", "something else:", "random question:",
               "on another note,", "totally separate:", "other thing:", "quick separate question:", "different thing -"),
}
LEAD_IN_SHARES = {"none": 0.35, "reaction": 0.35, "cue": 0.15, "marker": 0.15}
LEAD_IN_OF_TRANSITION = {"no_marker": "none", "blended": "reaction", "misleading_cue": "cue", "explicit": "marker"}


@dataclass
class SegmentSeed:
    segment_id: str
    industry: str
    use_case: str
    workflow: str
    task: str
    operation: str
    n_turns: int
    switch_in: Optional[str]          # None for the first segment of a session
    followup_kinds: List[str]         # one per user turn after the opening request
    object_ref: str                   # unique id of the fresh object this segment works on
    user_lengths: List[int] = field(default_factory=list)    # target chars per user turn (LENGTH_BUCKETS)
    turn_shapes: List[str] = field(default_factory=list)     # "prose" | "pasted" per user turn
    transition_style: Optional[str] = None                   # how the first turn of a non-first segment switches
    lead_ins: List[str] = field(default_factory=list)        # words each user turn starts with ("" = none)


@dataclass
class SessionSeed:
    session_id: str
    persona: str
    segments: List[SegmentSeed]


def _balanced_pool(values: Sequence, size: int, rng: random.Random) -> List:
    """`size` items cycling through `values`, shuffled: every value appears size//len(values) times or one more.
    Pass `values` in random order when size < len(values), or the first values are favoured."""
    pool = [values[i % len(values)] for i in range(size)]
    rng.shuffle(pool)
    return pool


def _share_pool(shares: Dict[str, float], size: int, rng: random.Random) -> List[str]:
    """`size` items in the given shares (largest remainders fill rounding), shuffled."""
    pool = [k for k, share in shares.items() for _ in range(int(share * size))]
    by_remainder = sorted(shares, key=lambda k: -(shares[k] * size - int(shares[k] * size)))
    pool += by_remainder[: size - len(pool)]
    rng.shuffle(pool)
    return pool


def _segment_lengths(n_segments: int, min_per_length: int, rng: random.Random) -> List[int]:
    lengths = [n for n in LENGTHS for _ in range(min_per_length)]
    lengths += [rng.choice(SHORT_LENGTHS) for _ in range(n_segments - len(lengths))]
    if len(lengths) != n_segments:
        raise ValueError(f"{n_segments} segments cannot cover lengths 1-20 {min_per_length} times each")
    return lengths


def _pack(lengths: List[int], counts: List[int], rng: random.Random) -> List[List[int]]:
    """Assign lengths to sessions (session i gets counts[i] of them) so no session exceeds
    MAX_SESSION_TURNS: longest first, each into the open session with the most budget left."""
    sessions: List[List[int]] = [[] for _ in counts]
    order = list(range(len(counts)))
    rng.shuffle(order)                                  # random tie-break between equal sessions
    for length in sorted(lengths, reverse=True):
        open_ = [i for i in order if len(sessions[i]) < counts[i]]
        best = max(open_, key=lambda i: MAX_SESSION_TURNS - sum(sessions[i]))
        if sum(sessions[best]) + length > MAX_SESSION_TURNS:
            raise ValueError("lengths do not fit the session budget; use more sessions or a lower minimum")
        sessions[best].append(length)
    for s in sessions:
        rng.shuffle(s)
    return sessions


def _pick_task(candidates: List[Task], used: Counter, rng: random.Random) -> Task:
    """The least-used candidate (random among ties), so every task is covered evenly."""
    fewest = min(used[t.id] for t in candidates)
    return rng.choice([t for t in candidates if used[t.id] == fewest])


def _next_task(previous: Task, switch: str, tasks: List[Task], used: Counter, rng: random.Random) -> Task:
    if switch == "same_task_new_object":
        return previous
    if switch == "same_industry_other_task":
        return _pick_task([t for t in tasks if t.industry == previous.industry and t.id != previous.id], used, rng)
    return _pick_task([t for t in tasks if t.industry != previous.industry], used, rng)


def make_seeds(n_sessions: int, seed: int, min_per_length: int = 6) -> List[SessionSeed]:
    rng = random.Random(seed)
    tasks = sorted(load_tasks().values(), key=lambda t: t.id)
    compatibility = load_compatibility(require_reviewed=False)

    counts = _balanced_pool(SEGMENTS_PER_SESSION, n_sessions, rng)
    lengths = _pack(_segment_lengths(sum(counts), min_per_length, rng), counts, rng)
    n_boundaries = sum(counts) - n_sessions
    switches = _share_pool(SWITCH_SHARES, n_boundaries, rng)
    transitions = _share_pool(TRANSITION_SHARES, n_boundaries, rng)
    followups = _share_pool(FOLLOWUP_SHARES, sum(sum(s) for s in lengths) - sum(counts), rng)

    used: Counter = Counter()
    sessions: List[SessionSeed] = []
    for s_idx, session_lengths in enumerate(lengths):
        session_id = f"s{seed}-{s_idx:03d}"
        segments: List[SegmentSeed] = []
        task = None
        for g_idx, n_turns in enumerate(session_lengths):
            switch = None if g_idx == 0 else switches.pop()
            task = _pick_task(tasks, used, rng) if task is None else _next_task(task, switch, tasks, used, rng)
            used[task.id] += 1
            row = compatibility[task.id]
            operation = row["primary"] if rng.random() < PRIMARY_OPERATION_SHARE else rng.choice(row["allowed"])
            user_lengths = _balanced_pool(rng.sample(LENGTH_BUCKETS, len(LENGTH_BUCKETS)), n_turns, rng)
            segments.append(SegmentSeed(
                segment_id=f"{session_id}-g{g_idx}", industry=task.industry, use_case=task.use_case,
                workflow=task.workflow, task=task.id, operation=operation, n_turns=n_turns, switch_in=switch,
                followup_kinds=[followups.pop() for _ in range(n_turns - 1)],
                object_ref=f"REF-{rng.randrange(10_000, 100_000)}",
                user_lengths=user_lengths,
                turn_shapes=["pasted" if rng.random() < PASTED_SHARE[n] else "prose" for n in user_lengths],
                transition_style=None if switch is None else transitions.pop()))
        crosses = any(g.switch_in == "cross_industry" for g in segments)
        persona = ("an independent consultant who works with clients in several industries" if crosses
                   else f"a staff member at a {segments[0].industry.replace('_', ' ')} organization")
        sessions.append(SessionSeed(session_id, persona, segments))
    _add_lead_ins(sessions, seed)
    return sessions


def _add_lead_ins(sessions: List[SessionSeed], seed: int) -> None:
    """Lead-ins use their own random stream, so adding them leaves every other slot unchanged."""
    rng = random.Random(f"{seed}:lead-ins")
    kinds, weights = list(LEAD_IN_SHARES), list(LEAD_IN_SHARES.values())
    for session in sessions:
        for segment in session.segments:
            segment.lead_ins = []
            for pos in range(segment.n_turns):
                if pos == 0 and segment.transition_style:
                    kind = LEAD_IN_OF_TRANSITION[segment.transition_style]
                else:
                    kind = rng.choices(kinds, weights)[0]
                segment.lead_ins.append(rng.choice(LEAD_IN_POOLS[kind]))


def coverage(sessions: List[SessionSeed]) -> Dict[str, Counter]:
    segments = [g for s in sessions for g in s.segments]
    return {
        "segments_per_session": Counter(len(s.segments) for s in sessions),
        "length": Counter(g.n_turns for g in segments),
        "industry": Counter(g.industry for g in segments),
        "operation": Counter(g.operation for g in segments),
        "switch_in": Counter(g.switch_in for g in segments if g.switch_in),
        "transition_style": Counter(g.transition_style for g in segments if g.transition_style),
        "followup_kind": Counter(k for g in segments for k in g.followup_kinds),
        "first_turn_length": Counter(g.user_lengths[0] for g in segments if g.user_lengths),
        "later_turn_length": Counter(n for g in segments for n in g.user_lengths[1:]),
        "boundary_lead_in": Counter(_lead_kind(g.lead_ins[0]) for g in segments if g.switch_in and g.lead_ins),
        "follow_up_lead_in": Counter(_lead_kind(x) for g in segments for x in g.lead_ins[1:]),
        "task_count_histogram": Counter(Counter(g.task for g in segments).values()),
        "session_turns_max": Counter([max(sum(g.n_turns for g in s.segments) for s in sessions)]),
    }


def _lead_kind(phrase: str) -> str:
    return next(kind for kind, pool in LEAD_IN_POOLS.items() if phrase in pool)


def write_seeds(sessions: List[SessionSeed], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in sessions:
            f.write(json.dumps(asdict(s), ensure_ascii=False, sort_keys=True) + "\n")


def read_seeds(path: Path) -> List[SessionSeed]:
    out = []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        out.append(SessionSeed(d["session_id"], d["persona"], [SegmentSeed(**g) for g in d["segments"]]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-per-length", type=int, default=6)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()
    sessions = make_seeds(args.sessions, args.seed, args.min_per_length)
    write_seeds(sessions, args.data_dir / "seeds.jsonl")
    print(f"{len(sessions)} sessions, {sum(len(s.segments) for s in sessions)} segments -> {args.data_dir / 'seeds.jsonl'}")
    for name, counter in coverage(sessions).items():
        print(f"  {name}: {dict(sorted(counter.items(), key=lambda kv: str(kv[0])))}")


if __name__ == "__main__":
    main()
