"""
Manifest splitting utilities for dividing JSONL manifests into train/val/test splits.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _allocate_counts(total: int, fractions: list[float]) -> list[int]:
    """Allocate integer item counts proportionally according to fractional weights.

    Uses the largest remainder method to ensure the allocated integers sum exactly
    to ``total``.
    """
    if total <= 0:
        return [0] * len(fractions)
    exact = [total * f for f in fractions]
    counts = [int(x) for x in exact]
    remainder = total - sum(counts)
    if remainder > 0:
        remainders = sorted([(exact[i] - counts[i], i) for i in range(len(fractions))], reverse=True)
        for i in range(remainder):
            counts[remainders[i][1]] += 1
    return counts


def _allocate_stratified_counts(strata_sizes: list[int], fractions: list[float]) -> list[list[int]]:
    """Allocate counts across multiple strata such that each stratum is split proportionally

    and the global sum across all strata matches the global target split counts exactly.
    """
    total = sum(strata_sizes)
    global_target = _allocate_counts(total, fractions)
    num_strata = len(strata_sizes)
    num_splits = len(fractions)

    base_alloc: list[list[int]] = []
    rem_s: list[int] = []
    for size in strata_sizes:
        base = [int(size * f) for f in fractions]
        base_alloc.append(base)
        rem_s.append(size - sum(base))

    needed = [
        global_target[j] - sum(base_alloc[s][j] for s in range(num_strata))
        for j in range(num_splits)
    ]

    # Rank candidates by largest fractional deficit
    candidates: list[tuple[float, int, int]] = []
    for s, size in enumerate(strata_sizes):
        for j, f in enumerate(fractions):
            deficit = (size * f) - base_alloc[s][j]
            candidates.append((deficit, s, j))
    candidates.sort(key=lambda x: x[0], reverse=True)

    alloc = [list(b) for b in base_alloc]
    for _deficit, s, j in candidates:
        if rem_s[s] > 0 and needed[j] > 0:
            alloc[s][j] += 1
            rem_s[s] -= 1
            needed[j] -= 1

    # Fallback in case of rare ties
    for s in range(num_strata):
        while rem_s[s] > 0:
            for j in range(num_splits):
                if needed[j] > 0:
                    alloc[s][j] += 1
                    rem_s[s] -= 1
                    needed[j] -= 1
                    break

    return alloc


def _get_stratify_value(record: dict[str, Any], field: str) -> str:
    """Extract stratification key from metadata or top-level fields."""
    if "metadata" in record and isinstance(record["metadata"], dict) and field in record["metadata"]:
        val = record["metadata"][field]
        return str(val) if val is not None else "__null__"
    if field in record:
        val = record[field]
        return str(val) if val is not None else "__null__"
    return "__missing__"


def split_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    stratify_field: str | None = None,
    seed: int | None = 42,
) -> dict[str, int]:
    """Split a JSONL manifest into train, validation, and test subsets.

    Args:
        manifest_path: Path to the input JSONL manifest file.
        output_dir: Directory where train.jsonl, val.jsonl, and test.jsonl will be saved.
        train_fraction: Proportion of samples allocated to training split.
        val_fraction: Proportion of samples allocated to validation split.
        test_fraction: Proportion of samples allocated to test split.
        stratify_field: Optional metadata key used to perform stratified sampling.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dictionary containing counts of records for each split and the total.
    """
    fractions = [train_fraction, val_fraction, test_fraction]
    total_frac = sum(fractions)
    if abs(total_frac - 1.0) > 1e-5:
        raise ValueError(
            f"Fractions must sum to 1.0, got train={train_fraction}, val={val_fraction}, "
            f"test={test_fraction} (sum={total_frac:.4f})"
        )
    if any(f < 0.0 for f in fractions):
        raise ValueError("Split fractions must be non-negative.")

    manifest_p = Path(manifest_path)
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    records: list[tuple[dict[str, Any], str]] = []
    with manifest_p.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num} in {manifest_p}: {exc}") from exc
            records.append((parsed, stripped))

    rng = random.Random(seed)

    train_lines: list[str] = []
    val_lines: list[str] = []
    test_lines: list[str] = []

    if stratify_field is not None and records:
        strata: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
        for parsed, raw in records:
            key = _get_stratify_value(parsed, stratify_field)
            strata[key].append((parsed, raw))

        sorted_keys = sorted(strata.keys())
        strata_sizes = [len(strata[k]) for k in sorted_keys]
        allocations = _allocate_stratified_counts(strata_sizes, fractions)

        for s_idx, key in enumerate(sorted_keys):
            stratum_records = list(strata[key])
            if seed is not None:
                rng.shuffle(stratum_records)
            n_train, n_val, n_test = allocations[s_idx]

            train_lines.extend(raw for _, raw in stratum_records[:n_train])
            val_lines.extend(raw for _, raw in stratum_records[n_train : n_train + n_val])
            test_lines.extend(
                raw for _, raw in stratum_records[n_train + n_val : n_train + n_val + n_test]
            )

        if seed is not None:
            rng.shuffle(train_lines)
            rng.shuffle(val_lines)
            rng.shuffle(test_lines)
    else:
        all_records = list(records)
        if seed is not None:
            rng.shuffle(all_records)
        counts = _allocate_counts(len(all_records), fractions)
        n_train, n_val, n_test = counts[0], counts[1], counts[2]

        train_lines = [raw for _, raw in all_records[:n_train]]
        val_lines = [raw for _, raw in all_records[n_train : n_train + n_val]]
        test_lines = [raw for _, raw in all_records[n_train + n_val : n_train + n_val + n_test]]

    train_file = output_p / "train.jsonl"
    val_file = output_p / "val.jsonl"
    test_file = output_p / "test.jsonl"

    with train_file.open("w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    with val_file.open("w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")

    with test_file.open("w", encoding="utf-8") as f:
        for line in test_lines:
            f.write(line + "\n")

    return {
        "train": len(train_lines),
        "val": len(val_lines),
        "test": len(test_lines),
        "total": len(records),
    }
