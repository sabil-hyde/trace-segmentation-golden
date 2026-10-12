# trace-segmentation-golden

A golden dataset for measuring any model on two trace-mining tasks:

- **Segmentation:** at each user turn of a chat, does a new task start here? (yes/no)
- **Taxonomy classification:** which industry, use case, workflow and task does a segment belong to, and which operation does it need?

The gold labels are **exact by construction**. Code fixes every label before any text exists, and a model only writes the conversation. The current version is **golden-v2**: 500 sessions, 1,500 segments, about 1,000 task boundaries. Every surface shortcut we tested (length, line breaks, first words…) is too weak to find boundaries, so a model has to read the content.

> Status: generated and machine-verified. **Human review is still pending** (`python -m golden.review`), so treat the labels as unreviewed until then.

## Quick start

```bash
git clone https://github.com/sabil-hyde/trace-segmentation-golden && cd trace-segmentation-golden
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest golden/tests -q           # offline, no API key needed

.venv/bin/python -m golden.view && open data/golden-v2/view.html      # read the data as chats
.venv/bin/python -m golden.checks                                      # quality and shortcut report
```

Generating new data needs an [OpenRouter](https://openrouter.ai) key. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`.

## What's in the repo

```
golden/                  the pipeline (one module per step)
  taxonomy.py            label space: 64 tasks + 10 operations + the task->operation table
  seeds.py               gold slots, fixed from a random seed before any text exists
  generate.py            a cheap model writes text for the slots; code rejects bad text and repairs it
  verify.py              another model family re-labels everything blind (no gold shown)
  checks.py              quality and shortcut checks, run after every batch
  view.py                HTML viewer: sessions as chats, with gold slots and boundaries marked
  review.py              human accept/reject of every session
  export.py              accepted sessions -> seg.jsonl / tax.jsonl / manifest.json
  style_mining.py        one-off: mines how users in open traces write (feeds the generator's style)
  taxonomy/              frozen label files (see "Label space")
  vendor/                small, unchanged copies from Hyde-Inc/trace-mining (see vendor/__init__.py)
  tests/                 offline tests
data/
  golden-v2/             the dataset (current)
  golden-v1/             the first full version, kept as reference for what went wrong (see docs/PROCESS.md)
docs/PROCESS.md          how the dataset was built, what failed in v1, how v2 fixes it
```

Each data folder holds:

| file | content |
|---|---|
| `seeds.jsonl` | **the gold**: one session per line, every slot of every segment and turn |
| `generations.jsonl` | the generated text per segment, plus model, prompt hash, attempts, rejection reasons and cost |
| `verification.jsonl` | the blind verifier's boundaries, task and operation per segment, and its disagreements |
| `quality.jsonl` | the quality-check table after every generation batch |

## How the labels stay exact

```
seeds.py      code fixes every label slot from a seed                 -> seeds.jsonl        (the gold)
generate.py   google/gemini-3.8-flash writes text for those slots     -> generations.jsonl
              code checks each turn; bad turns are repaired or the segment is regenerated
verify.py     openai/gpt-6-luna re-labels everything blind             -> verification.jsonl
checks.py     quality + shortcut checks after every batch              -> quality.jsonl
review.py     a human accepts or rejects each session                  -> review.jsonl
export.py     accepted sessions only                                   -> seg.jsonl, tax.jsonl, manifest.json
```

## Label space

- **Tasks:** the starter-pack usage taxonomy `golden/taxonomy/lookups/usage-v0.3.0.json`.
  - 8 industries: banking & lending, healthcare, insurance, logistics & transport, manufacturing, professional services, retail & e-commerce, software & IT.
  - Each industry has 2 use cases → 4 workflows → 8 tasks: **64 tasks**.
  - The file has no parent ids. `taxonomy.py` derives them from its fixed order and fails loudly if that order changes.
- **Operations:** the cost taxonomy's `operation` dimension, `golden/taxonomy/lookups/cost-v0.3.0.json`.
  - 10 values: classify, extract, summarize, transform, generate, calculate, apply_rules, reconcile, plan, code.
  - `taxonomy.py` adds one tie-break rule per operation for pairs that are easy to confuse.
- **Which operations a task may use:** `golden/taxonomy/operations-v1.json`. It is hand-written and reviewed.

## Slots per segment (golden-v2)

A **segment** is one task: one goal on one item, 1–20 user turns, each with an assistant reply. A **session** has 1–5 segments and at most 40 user turns.

| slot | values | why |
|---|---|---|
| industry / use case / workflow / task | 64 tasks, least-used first (each 23–26×) | taxonomy gold |
| operation | the task's primary operation half the time, otherwise any allowed one | taxonomy gold |
| `n_turns` | 1–20, every length at least 40× | segment lengths of all sizes |
| `switch_in` | `same_task_new_object` 35%, `same_industry_other_task` 35%, `cross_industry` 30% | boundary types, from hardest to easiest |
| `transition_style` | `no_marker` 35%, `blended` 35% (reacts, then pivots in the same message), `explicit` 15%, `misleading_cue` 15% ("also…" but a new item) | vague, realistic boundaries |
| `user_lengths` | 50 / 100 / 200 / 400 / 800 chars per user turn, balanced within each segment | openers and follow-ups get the same length distribution |
| `turn_shapes` | `prose` or `pasted` (a list, table, log or excerpt); depends only on length | line breaks can't mark openers |
| `lead_ins` | none / reaction ("got it") / cue ("oh and") / marker ("different question:") at the same shares for openers and follow-ups | first words can't mark openers |
| `followup_kinds` | clarify, provide_info, correct, refine, confirm, ask_detail, paste_more, and 25% **decoys** (sound like a switch, same item and goal) | hard negatives |

Rules every segment follows (in the generator prompt, enforced by code where possible):
- each segment works on a new item that no earlier segment touched, and is never a step or variant of an earlier goal;
- no batch phrasing ("same for the next one");
- no taxonomy names and no meta talk ("new task");
- no tools: the assistant only analyses or drafts from what the user wrote.

The segment definition is the one trace-mining uses: `golden/vendor/segmentation.py`.

## golden-v2 at a glance

| | golden-v1 (reference) | **golden-v2** |
|---|---|---|
| Sessions / segments | 500 / 1,500 | 500 / 1,500 |
| Failed segments | 6 | 1 (`s3-451-g1`, excluded at export) |
| Boundaries found by user-turn length alone (F1) | **0.977** | **0.21** |
| … by line breaks / first two words | 0.94 / 0.55 | 0.16 / 0.27 |
| Opener vs follow-up median length | 5.6× | 1.16× |
| Verifier boundary recall / precision | — | 98.8% / 54% |
| Verifier task / operation agreement | 93.8% / 89% | 90.3% / 84% |
| Cost (generate + verify) | $25.5 | $43.7 |

Reading the verifier numbers:
- **High recall:** real boundaries stay visible in the text.
- **Low precision:** decoys make models split where the task actually continues. That's the intended difficulty. On a 28-session trial, Claude Sonnet 5.5 also reached only boundary F1 0.74.
- **Gold quality:** reading every case where two models disagreed with the gold suggested about 0.5% of follow-ups drifted into adjacent work. Human review should catch those.

## Run the pipeline

```bash
source .venv/bin/activate
python -m golden.seeds --sessions 500 --seed 3 --min-per-length 40      # writes data/golden-v2/seeds.jsonl
python -m golden.generate --concurrency 14     # text + blind verify + checks per batch; resumable
python -m golden.view                          # data/golden-v2/view.html, works on partial runs too
python -m golden.review                        # human: accept/reject every session
python -m golden.export                        # seg.jsonl, tax.jsonl, manifest.json from accepted sessions
```

- **New version or trial:** pass `--data-dir data/<name>` to every step, for example `--data-dir data/trial --limit 20` for a trial.
- **Stop and resume:** Ctrl-C at any time, then re-run `generate` to resume. Sessions lost to a network outage are retried automatically.
- **Cost and time:** about $0.03 per segment, so roughly $40 and 3–3.5 hours for 500 sessions at concurrency 14. Keep the machine awake (`caffeinate -is` on macOS).
- **Checks:** they warn and never stop the run. The ones to watch: shortcut F1 ≥ 0.5, verifier boundary recall < 85%, failed segments > 2%.

## Using the exported data

- **`seg.jsonl`:** one row per gap before user turn k ≥ 1.
  - `conversation_so_far` is the full prefix from turn 0, never truncated.
  - `new_user_request` is the k-th user turn.
  - `gold` is `"true"` where a segment starts.
  - Slicing fields: `switch_in`, `transition_style`, `followup_kind`, `user_len_bucket`, `segment_len_before` and `segment_len_after`.
- **`tax.jsonl`:** one row per segment, with its text and gold `industry`, `use_case`, `workflow`, `task` and `operation`.
- **Splits:** by whole session, 30% `tune` / 70% `eval`. Tune prompts and thresholds on `tune` only.
- **Do not rank** Gemini- or GPT-family models on this data: Gemini wrote it and GPT filtered it.

## Known limits

- **Synthetic chat, not real agent traffic:** no tools and no system prompts. The 8 starter-pack industries don't match the open traces' domains (telecom and travel are missing).
- **Boundaries are rare:** about 9% of gaps. Report precision/recall/F1, not accuracy.
- **Operation is effectively multi-label:** score it as a secondary label.
- **Some tasks overlap:** retail order status vs logistics shipment status, code review vs release check. Verifier task agreement is about 90%.
- **Long contexts:** assistant replies are not capped, so long sessions can exceed small-context models.
