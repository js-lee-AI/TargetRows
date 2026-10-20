"""Target-aware tree selection and exact-address probability reuse."""

from .core import (
    CandidatePool,
    ExactRowCache,
    VerificationResult,
    distribution,
    expected_acceptance,
    expected_root_draw_acceptance,
    select_draft,
    select_oracle,
    select_preselected,
    select_root,
    select_root_draw,
    verify_tree,
)

__all__ = [
    "CandidatePool", "ExactRowCache", "VerificationResult", "distribution",
    "expected_acceptance", "expected_root_draw_acceptance", "select_draft",
    "select_oracle", "select_preselected", "select_root", "select_root_draw",
    "verify_tree",
]
