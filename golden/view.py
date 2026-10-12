"""Render the generated sessions as readable chats in one HTML page.

    python -m golden.view                 # -> <data dir>/view.html
    open data/golden-v2/view.html

Each session is collapsible; each segment shows its gold slots, and boundaries are marked
between segments. Long messages are folded (click to expand). The verifier's disagreements and
your review decision are shown when those files exist.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from golden import DEFAULT_DATA_DIR
from golden.sessions import Session, load_sessions, read_jsonl_by_id
from golden.generate import LENGTH_RANGE
from golden.taxonomy import load_tasks

FOLD_CHARS = 350          # messages longer than this are folded behind a preview

STYLE = """
:root { --bg:#fafafa; --fg:#1d1d1f; --muted:#6b6b70; --card:#fff; --line:#e3e3e8; --user:#e8f0fe; --bot:#f1f1f4;
        --cut:#c2410c; --bad:#b91c1c; --good:#15803d; }
@media (prefers-color-scheme: dark) { :root { --bg:#151517; --fg:#ececf0; --muted:#9a9aa3; --card:#1e1e22;
        --line:#33333a; --user:#1f2b40; --bot:#26262b; --cut:#fb923c; --bad:#f87171; --good:#4ade80; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, system-ui, sans-serif; margin:0 auto;
       max-width:980px; padding:16px; }
details.session { background:var(--card); border:1px solid var(--line); border-radius:10px; margin:14px 0; padding:4px 14px; }
details.session > summary { cursor:pointer; padding:8px 0; font-weight:600; }
.meta { color:var(--muted); font-size:13px; }
.slots { border-left:4px solid var(--cut); padding:6px 10px; margin:14px 0 8px; font-size:13px; background:var(--bg); }
.boundary { color:var(--cut); font-weight:700; font-size:12px; letter-spacing:.06em; margin-top:22px; }
.msg { border-radius:8px; padding:8px 12px; margin:6px 0; white-space:pre-wrap; word-wrap:break-word; }
.user { background:var(--user); margin-right:60px; } .assistant { background:var(--bot); margin-left:60px; }
.who { font-size:11px; font-weight:700; color:var(--muted); text-transform:uppercase; }
details.fold > summary { cursor:pointer; list-style:none; } details.fold[open] > summary .preview { display:none; }
.more { color:var(--muted); font-size:12px; }
.bad { color:var(--bad); } .good { color:var(--good); }
button { margin-right:8px; }
"""

SCRIPT = """function setAll(open){document.querySelectorAll('details').forEach(d=>d.open=open)}"""


def message(role: str, k: int, text: str, label: str = "") -> str:
    body = escape(text)
    if len(text) > FOLD_CHARS:
        body = (f'<details class="fold"><summary><span class="preview">{escape(text[:FOLD_CHARS])}… </span>'
                f'<span class="more">[{len(text)} chars, click to toggle]</span></summary>{body}</details>')
    return f'<div class="msg {role}"><div class="who">[{k}] {role} {label}</div>{body}</div>'


def user_label(segment, pos: int, text: str) -> str:
    """The turn's seed slots, e.g. '· opener · ~400 chars (412) · pasted'; a missed length range is marked."""
    if not segment.user_lengths:                     # v1 seeds have no per-turn slots
        return ""
    target = segment.user_lengths[pos]
    low, high = LENGTH_RANGE[target]
    size = f"~{target} chars ({len(text)})" if low <= len(text) <= high else f"<span class='bad'>~{target} chars ({len(text)})</span>"
    kind = "opener" if pos == 0 else segment.followup_kinds[pos - 1]
    return f"· {kind} · {size} · {segment.turn_shapes[pos]}"


def render_session(session: Session, task_names: dict, verification: dict, review: dict) -> str:
    seed = session.seed
    n_turns = sum(g.n_turns for g in seed.segments)
    status = "<span class='bad'>generation failed</span>" if not session.ok else ""
    if verification:
        problems = verification["disagreements"]
        status += (" <span class='good'>verifier agrees</span>" if not problems
                   else f" <span class='bad'>verifier: {len(problems)} disagreement(s)</span>")
    if review:
        status += f" · reviewed: <b>{review['decision']}</b>"
    parts = [f"<details class='session' open><summary>{seed.session_id} · {len(seed.segments)} segment(s) · "
             f"{n_turns} turns {status}</summary><div class='meta'>persona: {escape(seed.persona)}</div>"]
    k = 0
    for segment, gen in zip(seed.segments, session.generation["segments"]):
        if segment.switch_in:
            style = f" · {segment.transition_style}" if segment.transition_style else ""
            parts.append(f"<div class='boundary'>── NEW TASK STARTS AT TURN {k} · {segment.switch_in}{style} ──</div>")
        parts.append(
            f"<div class='slots'><b>{escape(task_names[segment.task])}</b> <span class='meta'>({segment.task})</span><br>"
            f"operation <b>{segment.operation}</b> · {segment.n_turns} turn(s) · follow-ups: "
            f"{', '.join(segment.followup_kinds) or '—'}</div>")
        if gen["status"] != "ok":
            parts.append(f"<div class='bad'>generation failed after {gen['attempts']} attempts: "
                         f"{escape('; '.join(gen['problems']))}</div>")
            k += segment.n_turns
            continue
        for pos, turn in enumerate(gen["turns"]):
            parts.append(message("user", k, turn["user"], user_label(segment, pos, turn["user"]))
                         + message("assistant", k, turn["assistant"]))
            k += 1
    if verification and verification["disagreements"]:
        items = "".join(f"<li>{escape(p)}</li>" for p in verification["disagreements"])
        parts.append(f"<div class='bad'><b>Verifier disagreements</b><ul>{items}</ul></div>")
    return "\n".join(parts) + "</details>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    sessions = load_sessions(args.data_dir, only_ok=False)
    verification = read_jsonl_by_id(args.data_dir / "verification.jsonl")
    reviews = read_jsonl_by_id(args.data_dir / "review.jsonl")
    task_names = {t.id: t.name for t in load_tasks().values()}
    body = "\n".join(render_session(s, task_names, verification.get(s.seed.session_id), reviews.get(s.seed.session_id))
                     for s in sessions)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>Golden sessions</title><style>{STYLE}</style><script>{SCRIPT}</script></head><body>"
            f"<h2>{escape(args.data_dir.name)}: {len(sessions)} generated session(s)</h2>"
            f"<button onclick='setAll(true)'>expand all</button><button onclick='setAll(false)'>collapse all</button>"
            f"{body}</body></html>")
    out = args.data_dir / "view.html"
    out.write_text(page)
    print(f"{len(sessions)} sessions -> {out}")


if __name__ == "__main__":
    main()
