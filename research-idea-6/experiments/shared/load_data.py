"""
Shared data loader for gender and ethnicity experiments.
Loads questions, name lists, and model registry from JSON files.
"""

import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent


def load_questions():
    """Load the 200-question dataset."""
    with open(_DATA_DIR / "questions.json") as f:
        data = json.load(f)
    return data["questions"]


def load_gender_names():
    """Load gender name lists: male, female, ambiguous."""
    with open(_DATA_DIR / "gender_names.json") as f:
        return json.load(f)


def load_ethnicity_names():
    """Load ethnicity name lists with gender annotations.

    Categories aligned with EEOC federal standards (OMB race/ethnicity).
    Returns dict with keys: white, black, hispanic, asian,
    native_american, pacific_islander, ambiguous.
    Each group (except ambiguous) has 'names' and 'gender' lists.
    """
    with open(_DATA_DIR / "ethnicity_names.json") as f:
        data = json.load(f)
    # Strip metadata keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_model_registry():
    """Load model configuration registry."""
    with open(_DATA_DIR / "model_registry.json") as f:
        return json.load(f)


def get_ethnicity_comparison(comparison_str):
    """Parse 'white_vs_black' into (ref_group, cmp_group) tuple.

    Returns (reference_key, comparison_key).
    """
    parts = comparison_str.split("_vs_")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid comparison '{comparison_str}'. "
            f"Expected format: 'white_vs_black'"
        )
    return parts[0], parts[1]


VALID_COMPARISONS = [
    "white_vs_black",
    "white_vs_hispanic",
    "white_vs_asian",
    "white_vs_native_american",
    "white_vs_pacific_islander",
]
