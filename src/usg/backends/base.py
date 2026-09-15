"""Common imputer interface for all model types."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from usg.constants import ID_COLUMN
from usg.schema import EncodedBatch, InputIssue

if TYPE_CHECKING:
    from usg.bundle.reader import Bundle


@dataclass
class RawOutput:
    """Model output in encoded space for one batch."""

    cat_codes: np.ndarray  # (N, Cc) int64, known passthrough + filled hidden imputable
    cat_probs: np.ndarray  # (N, Cc, K_max) float32
    num_values: np.ndarray  # (N, Cn) transformed
    num_mean: np.ndarray  # (N, Cn) transformed
    num_lo: np.ndarray  # (N, Cn) transformed 5% quantile
    num_hi: np.ndarray  # (N, Cn) transformed 95% quantile
    cat_samples: np.ndarray | None = None  # (N, K, Cc)
    num_samples: np.ndarray | None = None  # (N, K, Cn)

    @staticmethod
    def concat(parts: list[RawOutput]) -> RawOutput:
        def cat(name):
            arrays = [getattr(p, name) for p in parts]
            return None if arrays[0] is None else np.concatenate(arrays, axis=0)

        return RawOutput(**{f: cat(f) for f in RawOutput.__dataclass_fields__})


@dataclass
class ImputationResult:
    values: pd.DataFrame  # schema columns; known inputs passed through, hidden imputable columns filled
    imputed: pd.DataFrame  # bool, True where the model filled the cell
    probabilities: dict[str, np.ndarray]  # imputable categorical column -> (N, |vocab|)
    numeric_stats: dict[str, dict[str, np.ndarray]]  # numeric column -> {mean, q05, q95}
    issues: list[InputIssue]
    model_id: str
    seed: int | None
    n_samples: int
    decode: str
    vocab: dict[str, tuple[str, ...]] = field(default_factory=dict)
    samples: dict[str, np.ndarray] | None = None  # column -> (N, K) labels / values

    def top_k(self, column: str, k: int = 3) -> list[list[tuple[str, float]]]:
        probs = self.probabilities[column]
        vocab = self.vocab[column]
        order = np.argsort(-probs, axis=1)[:, :k]
        return [[(vocab[j], float(probs[i, j])) for j in row] for i, row in enumerate(order)]


class TorchImputer:
    """Encode -> chunked ``predict_encoded`` -> decode. Subclasses implement ``predict_encoded``."""

    capabilities: frozenset[str] = frozenset({"probabilities"})

    def __init__(self, bundle: Bundle, net):
        import torch

        self.bundle = bundle
        self.schema = bundle.schema
        self.config = bundle.config
        self.net = net
        self.device = next(net.parameters()).device
        self._torch = torch
        self._cat_imputable = torch.as_tensor(self.schema.cat_imputable, device=self.device)
        self._num_imputable = torch.as_tensor(self.schema.num_imputable, device=self.device)
        temps = self.schema.temperatures
        self._temperature = None if np.allclose(temps, 1.0) else torch.as_tensor(temps, device=self.device)

    @property
    def model_id(self) -> str:
        return self.bundle.model_id

    @property
    def inference_config(self) -> dict:
        return self.config.get("inference", {})

    def default_samples(self) -> int:
        return int(self.inference_config.get("default_samples", 1))

    def generator(self, seed: int | None):
        if seed is None:
            return None
        return self._torch.Generator(device=self.device).manual_seed(int(seed))

    def to_tensors(self, batch: EncodedBatch):
        t = self._torch
        return (
            t.as_tensor(batch.cat, device=self.device),
            t.as_tensor(batch.cat_known, device=self.device),
            t.as_tensor(batch.num, device=self.device),
            t.as_tensor(batch.num_known, device=self.device),
        )

    def predict_encoded(self, batch: EncodedBatch, *, n_samples: int, generator, decode: str | None = None) -> RawOutput:
        raise NotImplementedError

    def impute(
        self,
        df_known: pd.DataFrame,
        *,
        n_samples: int | None = None,
        seed: int | None = 0,
        batch_size: int = 4096,
        unknown: str = "missing",
        return_samples: bool = False,
        decode: str | None = None,
        ignore_columns: Iterable[str] = (ID_COLUMN,),
    ) -> ImputationResult:
        batch, issues = self.schema.encode(df_known, unknown=unknown, ignore_columns=ignore_columns)
        k = self.resolve_samples(n_samples)
        raw = self.impute_encoded(batch, n_samples=k, seed=seed, batch_size=batch_size, decode=decode)
        return self._build_result(df_known, batch, raw, issues, k, seed, return_samples, decode)

    def resolve_samples(self, n_samples: int | None) -> int:
        k = int(n_samples) if n_samples else self.default_samples()
        return max(1, min(k, int(self.inference_config.get("max_samples", 1024))))

    def impute_encoded(
        self,
        batch: EncodedBatch,
        *,
        n_samples: int | None = None,
        seed: int | None = 0,
        batch_size: int = 4096,
        decode: str | None = None,
    ) -> RawOutput:
        """Chunked prediction on an already-encoded batch (used by evaluation and confidence estimation)."""
        k = self.resolve_samples(n_samples)
        if len(batch) == 0:
            return self._empty_raw()
        gen = self.generator(seed)
        step = max(1, batch_size // k)
        with self._torch.inference_mode():
            parts = [
                self.predict_encoded(batch.take(slice(i, i + step)), n_samples=k, generator=gen, decode=decode)
                for i in range(0, len(batch), step)
            ]
        return RawOutput.concat(parts)

    def _empty_raw(self) -> RawOutput:
        cc, cn, km = len(self.schema.cat_columns), len(self.schema.num_columns), self.schema.max_cardinality
        z = np.zeros((0, cn), dtype=np.float32)
        return RawOutput(np.zeros((0, cc), np.int64), np.zeros((0, cc, km), np.float32), z, z, z, z)

    def _build_result(self, df_known, batch, raw, issues, k, seed, return_samples, decode) -> ImputationResult:
        schema = self.schema
        index = df_known.index
        values = schema.decode(raw.cat_codes, raw.num_values)
        values.index = index
        for issue in issues:  # the model conditioned on the nearest option; report the user's own value
            if issue.kind == "coerced" and issue.row >= 0:
                values.iloc[issue.row, values.columns.get_loc(issue.column)] = issue.value
        imputed = pd.DataFrame(False, index=index, columns=schema.names)
        cat_hidden = schema.cat_imputable[None, :] & ~batch.cat_known
        num_hidden = schema.num_imputable[None, :] & ~batch.num_known
        for j, spec in enumerate(schema.cat_columns):
            imputed[spec.name] = cat_hidden[:, j]
        for j, spec in enumerate(schema.num_columns):
            imputed[spec.name] = num_hidden[:, j]
            if spec.name in df_known.columns:  # keep the user's exact numeric input
                original = pd.to_numeric(df_known[spec.name], errors="coerce").to_numpy(dtype=float)
                values[spec.name] = np.where(batch.num_known[:, j], original, values[spec.name].to_numpy())

        probabilities = {
            spec.name: raw.cat_probs[:, j, : spec.cardinality].astype(np.float64)
            for j, spec in enumerate(schema.cat_columns)
            if spec.role == "imputable"
        }
        numeric_stats = {
            spec.name: {
                "mean": schema.inverse_numeric(spec.name, raw.num_mean[:, j]),
                "q05": schema.inverse_numeric(spec.name, raw.num_lo[:, j]),
                "q95": schema.inverse_numeric(spec.name, raw.num_hi[:, j]),
            }
            for j, spec in enumerate(schema.num_columns)
        }
        samples = None
        if return_samples and raw.cat_samples is not None:
            samples = {}
            for j, spec in enumerate(schema.cat_columns):
                vocab = np.asarray(spec.vocab, dtype=object)
                codes = raw.cat_samples[:, :, j]
                samples[spec.name] = np.where(codes >= 0, vocab[np.clip(codes, 0, len(vocab) - 1)], None)
            if raw.num_samples is not None:
                for j, spec in enumerate(schema.num_columns):
                    samples[spec.name] = schema.inverse_numeric(spec.name, raw.num_samples[:, :, j])
        return ImputationResult(
            values=values,
            imputed=imputed,
            probabilities=probabilities,
            numeric_stats=numeric_stats,
            issues=issues,
            model_id=self.model_id,
            seed=seed,
            n_samples=k,
            decode=decode or self.inference_config.get("decode", "default"),
            vocab={c.name: c.vocab for c in schema.cat_columns},
            samples=samples,
        )
