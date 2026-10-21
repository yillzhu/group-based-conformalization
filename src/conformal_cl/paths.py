"""Repository-local paths; relocation only, not a scientific change.

Use an editable install from a source checkout. Outputs are separate from the
immutable historical artifacts. The algorithms retain their original defaults.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
