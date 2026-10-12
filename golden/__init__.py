"""Golden dataset: seed-based synthetic chat sessions with exact segmentation and taxonomy gold.

    seeds -> generate (+ blind verify + checks) -> view -> review -> export      (see README.md)
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
DEFAULT_DATA_DIR = REPO_DIR / "data" / "golden-v2"
