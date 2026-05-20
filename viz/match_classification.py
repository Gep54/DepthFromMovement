from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.map import TwoViewResult


@dataclass(frozen=True)
class MatchClassification:
    """Mutually exclusive boolean masks over ``len(tw.pts1)`` correspondences."""

    epipolar: np.ndarray
    cheiral: np.ndarray
    inlier: np.ndarray


def classify_match_rejections(
    tw: TwoViewResult,
    *,
    check_cheiral: bool = True,
) -> MatchClassification:
    """
    Classify each match into exactly one category (priority: epipolar → cheiral → inlier).

    Cheiral applies only to motion-epipolar inliers; triangulation columns align with those.
    """
    n = len(tw.pts1)
    epi = np.zeros(n, dtype=bool)
    che = np.zeros(n, dtype=bool)
    inl = np.zeros(n, dtype=bool)

    if n == 0:
        return MatchClassification(epi, che, inl)

    epi_in = tw.inlier_mask.reshape(-1).astype(bool)
    epi[~epi_in] = True

    n_inl = int(epi_in.sum())
    if n_inl == 0 or tw.X_world_h.shape[1] == 0:
        return MatchClassification(epi, che, inl)

    cheiral = tw.cheiral_mask.reshape(-1).astype(bool)
    if cheiral.shape[0] != n_inl:
        cheiral = np.zeros(n_inl, dtype=bool)

    inlier_idx = 0
    for k in range(n):
        if not epi_in[k]:
            continue
        if check_cheiral and not cheiral[inlier_idx]:
            che[k] = True
        else:
            inl[k] = True
        inlier_idx += 1

    return MatchClassification(epi, che, inl)


def classification_counts(cls: MatchClassification) -> dict[str, int]:
    return {
        "epipolar": int(cls.epipolar.sum()),
        "cheiral": int(cls.cheiral.sum()),
        "inlier": int(cls.inlier.sum()),
    }


def audit_record(
    i: int,
    j: int,
    cls: MatchClassification,
) -> dict[str, Any]:
    counts = classification_counts(cls)
    return {
        "i": i,
        "j": j,
        "counts": counts,
    }


def append_rejection_audit(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _has_all_rejection_types(counts: dict[str, int]) -> bool:
    return counts.get("epipolar", 0) > 0 and counts.get("cheiral", 0) > 0


def write_pairs_all_rejection_types(
    path: str | Path,
    records: list[dict[str, Any]],
) -> None:
    pairs = [
        {"i": r["i"], "j": r["j"], "counts": r["counts"]}
        for r in records
        if _has_all_rejection_types(r["counts"])
    ]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pairs": pairs}, indent=2), encoding="utf-8")
