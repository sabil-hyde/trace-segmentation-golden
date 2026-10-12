# Taxonomy review: decisions needed before sharing golden-v2

Three decisions for the dataset owner. The first needs the team; the other two need the owner only. Each section gives the
evidence, the options and a recommendation. Record each decision in its **Decision** line.

---

## 1. Is usage-v0.3.0 the taxonomy to benchmark?

**What golden-v2 uses:** the starter-pack usage taxonomy `usage-v0.3.0`.
- 8 industries: banking & lending, healthcare, insurance, logistics & transport, manufacturing, professional services, retail & e-commerce, software & IT.
- Each has 2 use cases → 4 workflows → 8 tasks, giving **64 task labels**.
- The second axis is the cost taxonomy's `operation` (10 values).

**What it leaves out:**
- the 12 cross-industry "shared" tasks (e.g. debug software, reconcile financial records);
- `usage-v0.4.0`'s additions: telecom and travel & hospitality, plus 67 categories found by clustering the open traces. These are not yet checked on held-out data.

**Why it matters:** every gold label lives in this label space. Changing it later means a new dataset version (seeds plus
generation, about $40 and 4 hours).

**Draft message to the team:**

> We've built a golden benchmark for trace segmentation + taxonomy classification (500 synthetic sessions, exact
> labels, hardened so surface cues don't give away task boundaries). It labels against **usage-v0.3.0**: 8 industries
> × 8 tasks = 64 task labels, plus the cost taxonomy's `operation`. It does **not** include the 12 shared cross-industry
> tasks or v0.4.0's telecom / travel additions. Is v0.3.0 (industry tasks only) the label space we want models compared
> on, or should v3 add the shared tasks and/or v0.4.0? Repo: github.com/sabil-hyde/trace-segmentation-golden (private;
> happy to add you).

**Recommendation:** keep v0.3.0 for v2 and ask about adding the shared tasks in v3. The open traces are mostly
customer-service, coding and research work, which maps to the shared tasks and v0.4.0 rather than the industry tasks.

**Decision (2026-10-13):** keep **usage-v0.3.0** for golden-v2.

---

## 2. Changes made after the operations table was reviewed

The table (`golden/taxonomy/operations-v1.json`) was reviewed before the pilot. Since then:

### 2a. `software_and_it.task.access_action`: primary `generate` → `code`

The pilot showed that every access-change segment produced an IAM policy or CLI command, which is `code`. Allowed is now
`["code", "generate"]`. The owner approved this in chat, but it wasn't re-reviewed in the file.

**Decision (2026-10-13):** keep. `code` is the primary operation.

### 2b. Operation tie-break rules (`OPERATION_TIE_BREAKS` in `golden/taxonomy.py`)

Each rule is appended to the cost lookup's definition, and both the generator and the verifier read it. Nobody has reviewed them yet.

| operation | rule as written | concern |
|---|---|---|
| classify | The answer is one option from a fixed, named set (a path, tier, category). Checking whether criteria are met is apply_rules. | ok |
| extract | Fields present in the evidence are returned as they are, with no comparison or judgement. | ok |
| summarize | One body of supplied information is condensed. If the point is to find differences between sources, it is reconcile. | ok |
| transform | The same content in another form or language; nothing new is added and nothing is judged. | ok |
| generate | New non-executable content: a message, report, recommendation or listing. Executable code or configuration is code. | ok |
| calculate | Getting the answer requires arithmetic on numbers. | ok |
| **apply_rules** | Stated criteria, a policy or a checklist are applied to the facts of **one item**. Comparing two or more sources against each other is reconcile. | **Bug.** "one item" turns a catalog search and an option comparison into a single-item eligibility check (see section 3). Proposed: "…applied to the facts of the supplied item(s), e.g. checking one item or filtering a list. Comparing two sources *for inconsistencies* is reconcile." |
| reconcile | Two or more supplied sources (records, documents, logs) are compared against each other. | ok |
| plan | An ordered set of dependent steps is chosen under constraints. | ok |
| code | The output is meant to run or be applied: source code, tests, a CLI command, a policy or configuration file, even when the user says 'draft' or 'generate'. | ok |

**Decision (2026-10-13):** approved, with the apply_rules fix (now in `golden/taxonomy.py`).

---

## 3. Known overlap pairs: drift, overlap or merge?

The blind verifier disagreed with the gold task on **146 of 1,498 segments (9.7%)**. The disagreements are concentrated
in a few tasks (rate = flagged / all segments of that task × operation):

| gold task (operation) | verifier says | flagged | diagnosis |
|---|---|---|---|
| retail `order_status` (summarize / extract) | logistics `shipment_status` / `delay_investigation` | **20 / 23 (87%)** | **Genuine overlap.** The starter pack's own order_status example, "Explain a delayed delivery using tracking events", is effectively logistics shipment_status. The text fits both labels. |
| retail `product_search` (apply_rules) | other | **20 / 23 (87%)** | **Generator drift, caused by rule 2b.** Text: "does SKU KRM-60 qualify for the rebate?" is one item checked, not a catalog searched. |
| software `code_review` (apply_rules) | `release_check` | 14 / 23 (61%) | **Mostly drift (2b):** "compliance check on RFC-8841 against Rules 1–4" reviews change-request paperwork against a policy, so it reads as a release gate rather than a code diff. |
| prof. services `option_analysis` (apply_rules) | other | 10 / 18 (56%) | **Drift (2b):** "check if Kestrel meets Policy Sec-7" is one vendor checked, not options compared. |
| prof. services `deliverable_review` | `scope_compare` / other | 10 / 23 | Smaller overlap; review a few. |
| manufacturing `supplier_quality` | `inspection_check` | 6 / 18 | Sibling overlap (certificate vs measurements); acceptable. |

How each example was checked: `data/golden-v2/view.html`, searching for the segment ids, for example `s3-001-g0`,
`s3-027-g4`, `s3-072-g0` (product_search), `s3-014-g3`, `s3-051-g1` (order_status), `s3-006-g0`, `s3-156-g0`
(code_review) and `s3-025-g2`, `s3-180-g1` (option_analysis).

### Options

| | product_search, option_analysis, code_review (drift) | order_status (overlap) |
|---|---|---|
| **A. Fix and regenerate** (recommended for drift) | Fix the apply_rules rule (2b), add a one-line generator hint for these three tasks ("several catalog items / several options / a code diff"), then regenerate only the sessions that contain these tasks (about 60 sessions, about $5, about 30 min) and re-verify. | — |
| **B. Reject in review** | Drop the flagged sessions. The data shrinks and the task coverage becomes uneven. | — |
| **C. Keep and document** (recommended for overlap) | — | Keep the gold labels. Document the pair as a known overlap. In the playground, report task accuracy both strictly and with known-overlap pairs counted as correct, and rely on industry-level accuracy for this pair. Flag it to the taxonomy owners for v0.4. |
| **D. Merge in v3** | — | Merge retail order_status into a cross-industry "explain order/shipment status" task. This needs the team. |

**Recommendation:** use **A** for the three drift tasks and **C** for order_status. Raise D with the team in the message
from section 1.

**Decision (2026-10-13):**
- **A** for the drift tasks. The apply_rules rule is fixed and `TASK_HINTS` in `golden/generate.py` states what each of the three tasks' requests must contain. The 49 golden-v2 sessions containing product_search, option_analysis or code_review were regenerated and re-verified; the others are unchanged.
- **C** for order_status. The gold labels are kept, and the pairs are listed in `golden.taxonomy.KNOWN_OVERLAPS`. The playground reports taxonomy accuracy strictly *and* with these swaps forgiven (`accuracy_overlap_forgiven`).
- Merging these tasks (D) is raised with the team for v3.
