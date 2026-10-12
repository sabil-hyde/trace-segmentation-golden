# trace-segmentation-golden

A test set for two questions we ask about agent traces, and a small tool for scoring any model against it.

1. **Segmentation:** when a user sends a new message, are they starting a new task or continuing the current one?
2. **Taxonomy:** which kind of work is a task? There are 64 types across 8 industries.

> Status: the data is generated and machine-checked. A human review is still to do, so treat the results as provisional.

## How the data is built

We don't label conversations after the fact. We decide the answers first and then write conversations that fit them.

1. **Pick the answers.** Code chooses, from a fixed random seed, everything about each conversation: which tasks it contains, how many messages each task lasts, where the task changes, what kind of work each task is, and how each message should look. That file, `seeds.jsonl`, *is* the answer key.
2. **Write the text.** A cheap model (Gemini 3.8 Flash) writes a realistic chat that follows those choices. Code rejects anything that breaks the rules and asks for a fix.
3. **Check it blind.** A different model (GPT-6 Luna) labels every conversation without seeing the answers. Its disagreements show us where to look.
4. **Review.** A person reads the conversations and accepts or rejects them. This step is still to do.

**What we learned from the first version.** In v1, you could find almost every task change just by checking whether a message was long: new tasks opened with a long pasted document, and follow-ups were short. A simple length rule scored 0.98, so the test measured message length, not understanding.

**How v2 fixes it.** Message length, line breaks and opening words are now chosen by code, and they're spread the same way across new-task messages and follow-ups. A quarter of the follow-ups are deliberate traps: they sound like a switch ("also…", "different question:") but carry on the same task. Real user messages from public agent traces were used as style examples. Now no single surface feature does better than 0.26 F1.

## The dataset (`data/golden-v2`)

| | |
|---|---|
| Size | 500 conversations, 1,500 tasks, 999 task changes (about 9% of messages) |
| Task length | 1 to 20 user messages; each length appears at least 40 times |
| Kinds of task change | same kind of work on a new item (35%, the hardest), different work in the same industry (35%), different industry (30%) |
| Label set | 8 industries × 8 tasks = 64 tasks (starter-pack taxonomy `usage-v0.3.0`), plus the kind of operation (summarize, extract, code, …) |
| Blind check | the checker finds 99% of the real task changes and agrees on the task 93% of the time |
| Split | by conversation: 150 for tuning, 349 for evaluation |
| Cost to build | about $45 |

The files:
- `seeds.jsonl`: the answers.
- `generations.jsonl`: the conversations.
- `verification.jsonl`: the blind checker's labels.
- `quality.jsonl`: the quality checks recorded during generation.

## Results: spotting task changes

Evaluation split, 7,746 messages:

| model | F1 | precision / recall | typical latency | cost per 1,000 | failed |
|---|---|---|---|---|---|
| word-overlap baseline | 0.375 | 0.39 / 0.36 | instant | free | 0 |
| **Jev**, full conversation | **0.931** | 0.95 / 0.91 | 0.55 s | **$0.20** | 0 |
| Jev, last 2 messages only | 0.844 | 0.84 / 0.85 | 0.49 s | $0.05 | 0 |
| Claude Haiku 5.5, full conversation | 0.915 | 0.94 / 0.89 | 2.10 s | $0.74 | 95 |

Where the models go wrong:

| | new industry | other work, same industry | same work, new item | traps wrongly split |
|---|---|---|---|---|
| baseline | 56% found | 43% | 13% | 8.7% |
| Jev, full | 100% | 98% | 77% | 1.1% |
| Jev, last 2 | 98% | 94% | 65% | 3.3% |
| Haiku 5.5 | 99% | 90% | 79% | 1.3% |

**What this tells us:**
- **Word overlap isn't enough:** the baseline gets F1 0.375.
- **Jev is the best option:** it's more accurate than Haiku, about 4× cheaper and 4× faster, and it never failed. Haiku had 95 messages blocked by its content filter.
- **Jev needs the whole conversation:** with only the last two messages it loses 9 points.
- **The hard case is the same kind of work on a new item.** Every model misses a quarter or more of these.

The decision threshold for each model was tuned on the tuning split, never on the evaluation split.

**Taxonomy results:** not run yet. The tool supports them and reports accuracy two ways: strictly, and with known
look-alike task pairs forgiven.

## Try it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                       # add OPENROUTER_API_KEY (needed for Jev, Haiku and generation)
.venv/bin/python -m pytest -q              # tests, no API key needed

.venv/bin/python -m golden.view && open data/golden-v2/view.html    # read the conversations
.venv/bin/python -m golden.export --unreviewed                       # build the files the models are scored on
.venv/bin/python -m playground.run --task seg --classifier jev --fit-threshold
.venv/bin/python -m playground.compare                               # one table of all runs
./experiments.sh                                                     # rerun everything above (~$10)
```

**What each run saves:** a folder in `runs/` with:
- `manifest.json`: exactly what was asked and with which settings;
- `metrics.json`: accuracy, speed, cost and failures;
- predictions and a readable list of every mistake. These two are kept out of git; re-running rebuilds them.

**Adding a model:** write a small class with one method, `answer(question, input)`, which returns a probability for each
answer and what the call cost. Then register it in `playground/classifiers/__init__.py`. Everything else works unchanged.

## What's in the repo

```
golden/        builds the dataset: answers → text → blind check → viewer → review → export
playground/    scores models on it: baseline, Jev, any OpenRouter chat model; metrics and comparison
data/golden-v2 the dataset
runs/          the four runs above
```

## Decisions made

- **Taxonomy:** we use `usage-v0.3.0` (64 industry tasks). The cross-industry tasks and the newer telecom and travel categories are left out for now.
- **Code operations:** setting up access permissions counts as writing code, since the output is a policy or a command.
- **Three tasks that drifted:** product search, option comparison and code review were being written as single-item rule checks. We clarified what each request must contain and regenerated those 49 conversations. The checker's disagreement on them dropped from 44 of 69 to 4 of 69.
- **Look-alike tasks:** retail "order status" and logistics "shipment status" genuinely describe the same thing, so we keep the labels and also report accuracy with that pair forgiven.

## Limits

- **These are synthetic chats,** without tool calls or system prompts, so they're not real agent traffic.
- **Don't rank Gemini or GPT models on this data:** Gemini wrote it and GPT checked it.
- **Task changes are rare** (9%): look at precision, recall and F1, not accuracy.
- **A task can need more than one kind of operation,** so treat the operation label as secondary.
- **There are no "none of these" examples yet,** so we can't test how models handle unknown work.
