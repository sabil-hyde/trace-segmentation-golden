# Data versions

| folder | what | status |
|---|---|---|
| `golden-v2/` | **current dataset**: 500 sessions, 1,500 segments, seed 3; surface cues hidden by seed slots | generated + machine-verified; human review pending |
| `golden-v1/` | first full version: 500 sessions, seed 2 | **reference only**: boundaries are solvable from user-turn length (F1 0.977); see `docs/PROCESS.md` |

Rebuild the viewer for any folder with `python -m golden.view --data-dir data/<folder>`, and the check report with
`python -m golden.checks --data-dir data/<folder>`. `seeds.jsonl` is the gold; never edit it by hand. A change means a new seed
run in a new folder.
