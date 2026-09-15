"""Evidence-masking ensemble: re-run imputation with random subsets of each building's known inputs hidden.

Pass 0 keeps full evidence. Every other pass hides each known *imputable* input with probability ``drop_prob``
(conditioning inputs such as State are always kept, and at least one droppable input survives). All passes for a
chunk of buildings run as one ``N × P`` encoded batch through ``impute_encoded``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from usg.schema import EncodedBatch


@dataclass
class EnsembleChunk:
    start: int
    stop: int
    cat_codes: np.ndarray  # (n, P, Cc) predicted codes (known inputs of the pass passed through)
    num_values: np.ndarray  # (n, P, Cn) data units


def _pass_masks(known: np.ndarray, droppable: np.ndarray, n_passes: int, drop_prob: float, rng) -> np.ndarray:
    """Known masks per pass, shape (n, P, C). ``known``/``droppable``: (n, C)."""
    n, c = known.shape
    keep = rng.random((n, n_passes, c)) >= drop_prob
    keep[:, 0, :] = True
    keep &= droppable[:, None, :]
    starved = droppable.any(axis=1)[:, None] & ~keep.any(axis=2)  # (n, P): every droppable input was dropped
    for i, p in zip(*np.nonzero(starved)):
        keep[i, p, rng.choice(np.flatnonzero(droppable[i]))] = True
    return (known & ~droppable)[:, None, :] | keep


def iter_evidence_masking(
    imputer,
    batch: EncodedBatch,
    *,
    n_passes: int = 24,
    drop_prob: float = 0.35,
    seed: int = 0,
    n_samples: int = 1,
    chunk_rows: int = 4096,
) -> Iterator[EnsembleChunk]:
    schema = imputer.schema
    n_passes = max(1, int(n_passes))
    rng = np.random.default_rng(seed)
    step = max(1, chunk_rows // n_passes)
    cat_drop = schema.cat_imputable[None, :]
    num_drop = schema.num_imputable[None, :]
    for start in range(0, len(batch), step):
        part = batch.take(slice(start, start + step))
        n = len(part)
        n_cat = part.cat_known.shape[1]
        known_all = np.concatenate([part.cat_known, part.num_known], axis=1)
        droppable = np.concatenate([part.cat_known & cat_drop, part.num_known & num_drop], axis=1)
        masks = _pass_masks(known_all, droppable, n_passes, drop_prob, rng).reshape(n * n_passes, -1)
        cat_known = masks[:, :n_cat]
        num_known = masks[:, n_cat:]
        cat = np.where(cat_known, np.repeat(part.cat, n_passes, axis=0), -1)
        num = np.where(num_known, np.repeat(part.num, n_passes, axis=0), 0.0).astype(np.float32)
        raw = imputer.impute_encoded(
            EncodedBatch(cat, cat_known, num, num_known), n_samples=n_samples, seed=seed, batch_size=chunk_rows
        )
        num_values = np.empty_like(raw.num_values, dtype=np.float64)
        for j, spec in enumerate(schema.num_columns):
            num_values[:, j] = schema.inverse_numeric(spec.name, raw.num_values[:, j])
        yield EnsembleChunk(
            start,
            start + n,
            raw.cat_codes.reshape(n, n_passes, -1),
            num_values.reshape(n, n_passes, -1),
        )
