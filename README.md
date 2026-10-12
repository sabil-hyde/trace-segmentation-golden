# trace-segmentation-golden

A golden dataset for two trace-mining tasks, plus a playground that measures any classifier on it:

- **Segmentation:** at each user turn of a chat, does a new task start? (yes/no)
- **Taxonomy:** which of 64 industry tasks (8 industries) does a segment belong to?

The labels are **exact by construction**. Code fixes every label from a seed, a model only writes the conversation, and
a second model family checks it blind.

> Status: generated and machine-verified. Human review is pending, so results are provisional.

## The dataset: `data/golden-v2`

| | |
|---|---|
| Sessions / segments / boundaries | 500 / 1,500 / 999 (9% of the 11,062 gaps) |
| Label space | starter-pack `usage-v0.3.0`: 8 industries × 8 tasks = 64 tasks, plus the cost taxonomy's `operation` |
| Segment lengths | 1–20 user turns, each length at least 40 times; 1–5 segments per session |
| Boundary types | `same_task_new_object` 35% (hardest), `same_industry_other_task` 35%, `cross_industry` 30% |
| Hard negatives | 25% of follow-ups are **decoys**: they sound like a switch ("also…", "different question:") but continue the same task |
| Surface cues | Hidden. User-turn length, line breaks and opening words are seed slots with the same distribution for openers and follow-ups. The best single-cue rule finds boundaries with F1 ≤ 0.26. |
| Blind verifier (gpt-6-luna) | boundary recall 99%, task agreement 93% |
| Splits | by session: 150 tune / 349 eval |
| Generator / cost | `google/gemini-3.8-flash`, about $45 including verification |

**Files:**
- `seeds.jsonl` is **the gold**: every slot of every segment and turn.
- `generations.jsonl` holds the text.
- `verification.jsonl` holds the blind verifier's labels.
- `quality.jsonl` holds the checks from each generation batch.

## Results: segmentation (eval split, 7,746 gaps)

| classifier | boundary F1 | precision / recall | p50 latency | cost per 1k | failures |
|---|---|---|---|---|---|
| word-overlap baseline (`similarity`) | 0.375 | 0.39 / 0.36 | free | $0 | 0 |
| **Jev** (`typesafe/jev-1.13`), full history | **0.931** | 0.95 / 0.91 | 0.55 s | **$0.20** | 0 |
| Jev, last 2 turns only | 0.844 | 0.84 / 0.85 | 0.49 s | $0.05 | 0 |
| Claude Haiku 5.5, full history ¹ | 0.915 | 0.94 / 0.89 | 2.10 s | $0.74 | 95 (content filter) |

Recall by boundary type, and how often decoys are wrongly split:

| | cross-industry | same industry, other task | same task, new item | false splits on decoys |
|---|---|---|---|---|
| baseline | 0.56 | 0.43 | 0.13 | 8.7% |
| Jev, full | 1.00 | 0.98 | 0.77 | 1.1% |
| Jev, last 2 | 0.98 | 0.94 | 0.65 | 3.3% |
| Haiku 5.5 ¹ | 0.99 | 0.90 | 0.79 | 1.3% |

**Findings:**
- **Surface overlap doesn't work:** the baseline reaches F1 0.375.
- **Jev is the best trade-off:** it's more accurate than Haiku 5.5, 4× cheaper, 4× faster and never failed.
- **History matters:** with only the last 2 turns, Jev loses 9 points of F1, mostly on same-task-new-item switches and decoys.
- **The open problem is `same_task_new_object`:** the same kind of work on a different item. Every model misses about a fifth to a third of these.

Thresholds were chosen on the tune split, never on eval. ¹ Run on the export from before the 49-session taxonomy fix (below), which barely changes segmentation.

**Taxonomy:** not benchmarked yet. The playground supports it (`--task tax`) and reports accuracy two ways: strictly, and
with the known-overlap pairs forgiven.

## Quick start

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                                  # set OPENROUTER_API_KEY (only needed for API classifiers / generation)
.venv/bin/python -m pytest -q                         # offline tests

.venv/bin/python -m golden.view && open data/golden-v2/view.html      # read the data as chats
.venv/bin/python -m golden.export --unreviewed                         # build seg.jsonl / tax.jsonl
.venv/bin/python -m playground.run --task seg --classifier jev --fit-threshold
.venv/bin/python -m playground.compare                                 # table over all runs -> runs/COMPARISON.md
./experiments.sh                                                       # reproduce the results above
```

**Each run** writes `runs/<time>_<task>_<classifier>_<variant>/`:
- `manifest.json`: dataset sha256, the exact question sent, config, threshold, hardware;
- `metrics.json`: quality, slices, latency, cost and failures;
- `predictions.jsonl` and `report.md`: per-item outputs and every mistake. These two are git-ignored, and re-running rebuilds them.

**Options:**
- `--task seg`: `--context full|last2`.
- `--task tax`: `--level task|industry` and `--view segment|first_turn`.
- Both: `--sample N` takes a stratified sample.

**Adding a classifier:** subclass `playground.classifiers.base.Classifier`, implement
`answer(spec, state) -> Answer(probs, cost_usd, cost_basis, ...)`, and add one branch to `playground.classifiers.make`.

## Repo

```
golden/        dataset pipeline: taxonomy → seeds → generate (+ blind verify, checks) → view → review → export
  taxonomy/    label files: usage/cost lookups, task→operation table, real-user style examples
  vendor/      small unchanged copies from Hyde-Inc/trace-mining (LLM client, turn parser, segmentation prompt)
playground/    classifiers (similarity, tfidf, jev, llm), run loop, metrics, comparison
data/golden-v2 the dataset
runs/          the four runs above (manifest + metrics)
```

**Regenerating the dataset:** `python -m golden.seeds --sessions 500 --seed 3 --min-per-length 40`, then
`python -m golden.generate`. That takes about 4 hours and costs about $45. Checks print after every batch, and the run
can be resumed if stopped.

## Decisions

- **Taxonomy:** `usage-v0.3.0`, without the 12 shared cross-industry tasks or v0.4.0's telecom and travel additions.
- **Operation table:** `access_action` defaults to `code`. One tie-break rule per operation; `apply_rules` covers one item or a list.
- **The 49 drifting sessions:** these contained `product_search`, `option_analysis` and `code_review`. They were regenerated with task hints, which cut verifier disagreement on those tasks from 44/69 to 4/69.
- **Overlapping tasks:** retail `order_status` and logistics `shipment_status` / `delay_investigation` genuinely overlap, so their gold labels are kept and taxonomy accuracy is reported two ways (`golden.taxonomy.KNOWN_OVERLAPS`).

## Limits

- **Synthetic chat, not real agent traffic:** no tools and no system prompts.
- **Don't rank Gemini or GPT models on this data:** Gemini wrote it and GPT verified it.
- **Boundaries are rare** (9% of gaps): report P/R/F1, not accuracy.
- **Operation is effectively multi-label:** treat it as a secondary label.
- **No gold "unknown" inputs,** so rejection of unknown inputs can't be measured yet.
