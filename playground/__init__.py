"""Classification playground: golden export -> classifier -> saved predictions -> metrics.

    python -m playground.run --task seg --classifier jev          (see README.md, "Playground")
"""

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_DIR / "runs"
DEFAULT_EXPORT = REPO_DIR / "data" / "golden-v2" / "export-unreviewed"
