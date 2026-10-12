"""The frozen label space: 64 industry tasks (usage lookup v0.3.0), 10 operations (cost lookup
v0.3.0), and the reviewed table of which operations each task allows.

The usage lookup has no parent ids. Inside each industry its categories are listed in the fixed
order use_case, workflow, task, task, workflow, task, task (twice), so parents follow from
position. That pattern is asserted; any change to the file fails loudly instead of mislabelling.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from golden import PACKAGE_DIR

LOOKUPS = PACKAGE_DIR / "taxonomy" / "lookups"      # starter-pack lookups, copied from trace-mining

USAGE_LOOKUP = "usage-v0.3.0"
COST_LOOKUP = "cost-v0.3.0"
OPERATIONS_FILE = PACKAGE_DIR / "taxonomy" / "operations-v1.json"
EXPECTED_ORDER = "uwttwttuwttwtt"   # use_case, workflow, task ... per industry


@dataclass(frozen=True)
class Task:
    id: str
    industry: str
    use_case: str
    workflow: str
    name: str
    definition: str
    examples: Tuple[str, ...]
    excludes: Tuple[str, ...]

    def card(self) -> str:
        """The task as a generator or verifier reads it. The name is left out on purpose so it
        cannot leak into generated text."""
        text = self.definition
        if self.examples:
            text += " For example: " + "; ".join(self.examples) + "."
        if self.excludes:
            text += " Not this: " + "; ".join(self.excludes) + "."
        return text


def load_tasks() -> Dict[str, Task]:
    usage = load(USAGE_LOOKUP)
    by_industry: Dict[str, List[dict]] = {}
    for category in usage["categories"]:
        if category["industry"] != "all":
            by_industry.setdefault(category["industry"], []).append(category)

    tasks: Dict[str, Task] = {}
    for industry, categories in by_industry.items():
        order = "".join(c["dimension"][0] for c in categories)
        if order != EXPECTED_ORDER:
            raise ValueError(f"{USAGE_LOOKUP}: {industry} has order {order!r}, expected {EXPECTED_ORDER!r}")
        use_case = workflow = None
        for c in categories:
            if c["dimension"] == "use_case":
                use_case = c["id"]
            elif c["dimension"] == "workflow":
                workflow = c["id"]
            else:
                tasks[c["id"]] = Task(c["id"], industry, use_case, workflow, c["name"], c["definition"],
                                      tuple(c.get("examples") or ()), tuple(c.get("excludes") or ()))
    return tasks


# Tie-break rules added to the cost lookup's definitions (the ids and definitions stay as published).
# The pilot's verifier disagreed most on these pairs; generator and verifier both read them.
OPERATION_TIE_BREAKS = {
    "classify": "The answer is one option from a fixed, named set (a path, tier, category). Checking whether "
                "criteria are met (pass/fail, eligible or not) is apply_rules.",
    "extract": "Fields present in the evidence are returned as they are, with no comparison or judgement.",
    "summarize": "One body of supplied information is condensed. If the point is to find differences between "
                 "two or more sources, it is reconcile.",
    "transform": "The same content in another form or language; nothing new is added and nothing is judged.",
    "generate": "New non-executable content: a message, report, recommendation or listing. Executable code or "
                "configuration is code, not generate.",
    "calculate": "Getting the answer requires arithmetic on numbers.",
    "apply_rules": "Stated criteria, a policy or a checklist are applied to the facts of the supplied item(s), e.g. "
                   "checking one item or filtering a list. Comparing two sources for inconsistencies is reconcile.",
    "reconcile": "Two or more supplied sources (records, documents, logs) are compared against each other.",
    "plan": "An ordered set of dependent steps is chosen under constraints.",
    "code": "The output is meant to run or be applied: source code, tests, a CLI command, a policy or "
            "configuration file, even when the user says 'draft' or 'generate'.",
}


# Task pairs whose starter-pack definitions genuinely overlap (the same text fits both; docs/TAXONOMY_REVIEW.md).
# Gold labels are kept; the playground reports task accuracy both strictly and with these pairs forgiven.
KNOWN_OVERLAPS = (
    ("retail_and_ecommerce.task.order_status", "logistics_and_transport.task.shipment_status"),
    ("retail_and_ecommerce.task.order_status", "logistics_and_transport.task.delay_investigation"),
)


def load_operations() -> Dict[str, str]:
    """Operation id -> the cost lookup's definition plus its tie-break rule."""
    (dimension,) = [d for d in load(COST_LOOKUP)["dimensions"] if d["id"] == "operation"]
    return {v["id"]: f"{v['definition']} {OPERATION_TIE_BREAKS[v['id']]}" for v in dimension["values"]}


def category_names() -> List[str]:
    """Every use-case, workflow and task name of the 8 industries (used to catch label leaks)."""
    return [c["name"] for c in load(USAGE_LOOKUP)["categories"]
            if c["industry"] != "all" and c["dimension"] != "role"]


def load_compatibility(path: Path = OPERATIONS_FILE, require_reviewed: bool = True) -> Dict[str, dict]:
    """task id -> {"primary": op, "allowed": [ops]}, validated against the frozen label space."""
    table = json.loads(Path(path).read_text())
    if require_reviewed and table.get("reviewed") is not True:
        raise SystemExit(f"{path} is not reviewed yet: check every row, then set \"reviewed\": true")
    tasks, operations = load_tasks(), load_operations()
    rows = table["tasks"]
    problems = [f"missing task {t}" for t in tasks if t not in rows]
    problems += [f"unknown task {t}" for t in rows if t not in tasks]
    for task_id, row in rows.items():
        problems += [f"{task_id}: unknown operation {op}" for op in row["allowed"] if op not in operations]
        if row["primary"] not in row["allowed"]:
            problems.append(f"{task_id}: primary {row['primary']} not in allowed")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return rows


def load(name: str) -> dict:
    return json.loads((LOOKUPS / f"{name}.json").read_text())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lookup_path(name: str) -> Path:
    return LOOKUPS / f"{name}.json"
