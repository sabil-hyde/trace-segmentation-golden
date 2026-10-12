# How the golden dataset was built

The goal was a playground that compares classifiers (Jev, Kev, Laya, frontier models, simple baselines) on segmentation and taxonomy classification of traces, reporting quality, speed and cost. A comparison is only as good as its labels, so we first built a dataset whose gold is exact.

## 1. Why synthetic, seed-first data

Labelling real traces by hand is slow, and labels written by a frontier model are exactly what we want to test models against. So we reversed the order:

1. **Code picks the labels first.** It decides which task, operation, boundary type and segment length come where, from a random seed.
2. **A model writes the text to fit those labels.**
3. **Code rejects text that breaks a rule.**
4. **A model from a different family re-labels it blind,** and a human reviews it.

The gold can't drift, because no model ever chooses a label.

## 2. Pilot (60 sessions)

**What we checked:** whether a cheap model can follow the slots. Generation cost $2.82, and every segment passed.

**What reading the verifier's disagreements taught us:**
- **Two notions of "operation" were being compared.** The verifier judged the operation over the whole segment, while the gold uses the opening request. We aligned the verifier with the gold.
- **`generate` and `code` blur together.** IAM policies and CLI commands are code. We added one tie-break rule per operation, and `access_action` now defaults to `code`.
- **Some starter-pack tasks overlap,** for example retail order status vs logistics shipment status. This is a taxonomy limit, not a data bug.

## 3. golden-v1 (500 sessions): good labels, trivially solvable

v1 looked healthy. The verifier agreed on 95.5% of boundaries and 93.8% of tasks, there were no near-duplicates and no refusals. Then we asked a different question: **could a rule with no understanding find the boundaries?** It could, almost perfectly:

| Rule (threshold fit on half the sessions, scored on the other half) | Boundary F1 |
|---|---|
| new task if the user turn is ≥ 499 characters | **0.977** |
| new task if the user turn has ≥ 6 line breaks | 0.939 |
| new task if the turn is ≥ 3× the previous user turn | 0.79 |

**Cause:** our own rule that "the opening request is self-contained and pastes its material". It made every opener long (median 824 chars, 12 line breaks) and every follow-up short (median 147 chars, 0 line breaks). On top of that:
- openers started with words follow-ups never used ("I am…", "Please evaluate…");
- every switch was worded clearly.

The open traces (tau2 customer-service sessions) look nothing like this. First and later user turns have similar lengths (median 194 vs 153 chars), later turns are *more* often multi-line, and real switches can be buried mid-message.

`data/golden-v1/` is kept so this stays reproducible: `python -m golden.checks --data-dir data/golden-v1`.

## 4. golden-v2: hide every surface cue

Each cue became a seed slot, so it is controlled by code, not left to the model:

| Cue in v1 | Fix in v2 |
|---|---|
| Opener length | Every user turn gets a length bucket (50/100/200/400/800 chars), balanced within each segment, so openers and follow-ups share one distribution. Code enforces each turn's range. |
| Line breaks | The turn shape (prose or pasted block) depends only on the length bucket, never on position. |
| Clear switch wording | A transition style per boundary: no marker, blended (react, then pivot in the same message), explicit, or misleading cue. Plus 25% decoy follow-ups that sound like a switch but continue. |
| First words | Lead-in phrases come from shared pools in the same mix for openers and follow-ups. A trial showed the model copying example phrases from the prompt verbatim (first-words F1 rose to 0.52), so the phrases are now picked in code. |
| Unrealistic register | Real user turns mined from the open traces (`style_mining.py` → `taxonomy/user_style-v1.json`) are shown to the generator as style references. |

Four 20–28-session trials tuned the generator before the full run:
- **Strict JSON parsing** threw away valid replies that contained raw line breaks. Fixed with `strict=False`.
- **A single bad turn made a 20-turn segment regenerate from scratch.** Fixed with a repair pass that rewrites only the failing turns. Failures went from 6–9% to 0%, and cost fell from $0.040 to $0.027 per segment.
- **The internal item id leaked into the text.** Fixed by removing it from the prompt.

The checks now run after every batch, so a leak like v1's shows up while the data is still being generated.

## 5. Is v2 hard but still correct?

**On the 28-session trial:**
- **Recall:** gpt-6-luna and Claude Sonnet 5.5 found 98% and 94% of real boundaries. A missed boundary would mean an invisible switch; that almost never happens.
- **Precision:** they over-split at decoys (precision 56% and 62%), so boundary F1 is about 0.71–0.74. That is the intended difficulty.
- **Gold quality:** of the 15 gaps where both models split but the gold continues, about 12 are real decoys on the same item, and about 3 are generator drift into adjacent work. That's roughly 0.5% of follow-ups.

**On the full golden-v2:**
- every shortcut F1 is between 0.16 and 0.27;
- opener vs follow-up median length is 1.16×;
- verifier recall is 98.8%;
- 1 of 1,500 segments failed;
- cost was $43.70.

## 6. What's next

1. **Human review** of every session, starting with the gaps where the verifier split but the gold continues. Then export.
2. **The playground:** a classifier interface, a run loop, saved predictions, and metrics covering quality, latency, cost and failures. Run Jev and a simple baseline on the eval split.
3. **v3 candidates:**
   - the shared cross-industry tasks;
   - telecom and travel from usage-v0.4.0;
   - operation as a multi-label gold;
   - a real-trace eval set alongside the synthetic one.
