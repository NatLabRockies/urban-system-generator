"""Feature schema: ordered columns, roles, vocabularies and numeric transforms, plus vectorized encode/decode.

The schema is the contract between training and inference. Categorical codes are vocabulary indices
(``0..K-1``); ``-1`` means missing. Models reserve index ``K`` as their ``[MASK]`` embedding.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd

from usg.constants import SCHEMA_FORMAT
from usg.errors import InputValidationError, SchemaError

ROLES = ("imputable", "conditioning")
TYPES = ("categorical", "numeric")

_INTEGRAL_FLOAT_RE = re.compile(r"^[+-]?\d+\.0+$")
_INTEGER_RE = re.compile(r"^\d+$")


def canonical_label(value: Any) -> str | None:
    """Canonical string for a raw cell value; ``None`` means missing.

    ``"None"`` is a real ResStock option and is kept. Integral floats collapse (``3.0``/``"3.0"`` -> ``"3"``).
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else str(float(value))
    text = unicodedata.normalize("NFC", str(value)).strip()
    if not text:
        return None
    if _INTEGRAL_FLOAT_RE.match(text):
        return str(int(float(text)))
    return text


@dataclass(frozen=True)
class QuantileNormal:
    """Monotone map from data values to standard-normal scores (sklearn-style, stored as JSON arrays)."""

    quantiles: tuple[float, ...]
    references: tuple[float, ...]

    @classmethod
    def fit(cls, values: Iterable[float], n_quantiles: int = 1000) -> QuantileNormal:
        x = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise SchemaError("cannot fit a quantile transform on no finite values")
        n = int(min(n_quantiles, max(x.size, 2)))
        levels = np.linspace(0.0, 1.0, n)
        q = np.quantile(x, levels)
        dist = NormalDist()
        eps = 1e-7
        refs = [dist.inv_cdf(min(max(p, eps), 1 - eps)) for p in levels]
        return cls(tuple(float(v) for v in q), tuple(float(v) for v in refs))

    def forward(self, x: np.ndarray) -> np.ndarray:
        q = np.asarray(self.quantiles)
        r = np.asarray(self.references)
        x = np.clip(np.asarray(x, dtype=np.float64), q[0], q[-1])
        # Average the forward and reversed interpolation so tied quantiles map to their midpoint.
        return 0.5 * (np.interp(x, q, r) - np.interp(-x, -q[::-1], -r[::-1]))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        q = np.asarray(self.quantiles)
        r = np.asarray(self.references)
        z = np.clip(np.asarray(z, dtype=np.float64), r[0], r[-1])
        return np.interp(z, r, q)

    def to_json(self) -> dict:
        return {"kind": "quantile_normal", "quantiles": list(self.quantiles), "references": list(self.references)}

    @classmethod
    def from_json(cls, obj: dict) -> QuantileNormal:
        if obj.get("kind") != "quantile_normal":
            raise SchemaError(f"unsupported numeric transform {obj.get('kind')!r}")
        return cls(tuple(obj["quantiles"]), tuple(obj["references"]))


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    role: Literal["imputable", "conditioning"]
    type: Literal["categorical", "numeric"]
    vocab: tuple[str, ...] | None = None
    ordinal: bool = False
    transform: QuantileNormal | None = None
    clip: tuple[float, float] | None = None
    unit: str | None = None
    # Metadata below is excluded from the schema hash.
    counts: tuple[int, ...] | None = None
    temperature: float = 1.0
    description: str | None = None

    def __post_init__(self):
        if self.role not in ROLES:
            raise SchemaError(f"column {self.name!r}: role must be one of {ROLES}, got {self.role!r}")
        if self.type not in TYPES:
            raise SchemaError(f"column {self.name!r}: type must be one of {TYPES}, got {self.type!r}")
        if self.type == "categorical":
            if not self.vocab:
                raise SchemaError(f"categorical column {self.name!r} needs a non-empty vocab")
            if len(set(self.vocab)) != len(self.vocab):
                raise SchemaError(f"categorical column {self.name!r} has duplicate vocab entries")
        elif self.transform is None:
            raise SchemaError(f"numeric column {self.name!r} needs a transform")

    @property
    def cardinality(self) -> int:
        return len(self.vocab) if self.vocab else 0

    def to_json(self) -> dict:
        obj: dict[str, Any] = {"name": self.name, "role": self.role, "type": self.type}
        if self.type == "categorical":
            obj["vocab"] = list(self.vocab)
            if self.ordinal:
                obj["ordinal"] = True
        else:
            obj["transform"] = self.transform.to_json()
            if self.clip is not None:
                obj["clip"] = list(self.clip)
            if self.unit:
                obj["unit"] = self.unit
        if self.counts is not None:
            obj["counts"] = list(self.counts)
        if self.temperature != 1.0:
            obj["temperature"] = self.temperature
        if self.description:
            obj["description"] = self.description
        return obj

    @classmethod
    def from_json(cls, obj: dict) -> ColumnSpec:
        try:
            return cls(
                name=obj["name"],
                role=obj["role"],
                type=obj["type"],
                vocab=tuple(str(v) for v in obj["vocab"]) if obj.get("vocab") is not None else None,
                ordinal=bool(obj.get("ordinal", False)),
                transform=QuantileNormal.from_json(obj["transform"]) if obj.get("transform") else None,
                clip=tuple(obj["clip"]) if obj.get("clip") is not None else None,
                unit=obj.get("unit"),
                counts=tuple(int(c) for c in obj["counts"]) if obj.get("counts") is not None else None,
                temperature=float(obj.get("temperature", 1.0)),
                description=obj.get("description"),
            )
        except KeyError as exc:
            raise SchemaError(f"schema column is missing field {exc}") from None

    def _hash_view(self, include_transform: bool = True) -> list:
        view: list[Any] = [self.name, self.type]
        if include_transform:
            view.insert(1, self.role)
            view += [list(self.vocab) if self.vocab else None]
            view += [self.transform.to_json() if self.transform else None, list(self.clip) if self.clip else None]
        else:
            view += [list(self.vocab) if self.vocab else None]
        return view


@dataclass(frozen=True)
class InputIssue:
    row: int  # -1 for column-level issues
    column: str
    value: str
    kind: Literal["unknown_label", "unknown_column", "not_numeric", "coerced"]
    action: str


@dataclass
class EncodedBatch:
    cat: np.ndarray  # int64 (N, Cc); -1 = missing
    cat_known: np.ndarray  # bool (N, Cc)
    num: np.ndarray  # float32 (N, Cn); transformed, 0 where missing
    num_known: np.ndarray  # bool (N, Cn)

    def __len__(self) -> int:
        return self.cat.shape[0]

    def take(self, index: np.ndarray | slice) -> EncodedBatch:
        return EncodedBatch(self.cat[index], self.cat_known[index], self.num[index], self.num_known[index])


def _sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Schema:
    """Ordered list of columns. Categorical and numeric columns are indexed separately, in list order."""

    def __init__(
        self,
        columns: Iterable[ColumnSpec],
        *,
        data_release: str | None = None,
        groups: dict[str, list[str]] | None = None,
        metadata: dict | None = None,
    ):
        self.columns: list[ColumnSpec] = list(columns)
        names = [c.name for c in self.columns]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise SchemaError(f"duplicate column names in schema: {dupes}")
        self.data_release = data_release
        self.groups = {k: list(v) for k, v in (groups or {}).items()}
        for group, members in self.groups.items():
            unknown = [m for m in members if m not in names]
            if unknown:
                raise SchemaError(f"group {group!r} references unknown columns {unknown}")
        self.metadata = dict(metadata or {})
        self.cat_columns = [c for c in self.columns if c.type == "categorical"]
        self.num_columns = [c for c in self.columns if c.type == "numeric"]
        self._by_name = {c.name: c for c in self.columns}
        self._cat_pos = {c.name: i for i, c in enumerate(self.cat_columns)}
        self._num_pos = {c.name: i for i, c in enumerate(self.num_columns)}
        # Lookups accept the vocab spelling and its canonical form ("2.0" in options_lookup matches input "2").
        self._exact = []
        self._lower = []
        for c in self.cat_columns:
            exact: dict[str, int] = {}
            lower: dict[str, int] = {}
            for i, v in enumerate(c.vocab):
                exact[v] = i
            for i, v in enumerate(c.vocab):
                canon = canonical_label(v)
                if canon is not None:
                    exact.setdefault(canon, i)
                lower.setdefault(v.lower(), i)
                if canon is not None:
                    lower.setdefault(canon.lower(), i)
            self._exact.append(exact)
            self._lower.append(lower)

    # ---- introspection -------------------------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def cardinalities(self) -> list[int]:
        return [c.cardinality for c in self.cat_columns]

    @property
    def max_cardinality(self) -> int:
        return max(self.cardinalities, default=1)

    @property
    def cat_imputable(self) -> np.ndarray:
        return np.array([c.role == "imputable" for c in self.cat_columns], dtype=bool)

    @property
    def num_imputable(self) -> np.ndarray:
        return np.array([c.role == "imputable" for c in self.num_columns], dtype=bool)

    @property
    def imputable_names(self) -> list[str]:
        return [c.name for c in self.columns if c.role == "imputable"]

    @property
    def conditioning_names(self) -> list[str]:
        return [c.name for c in self.columns if c.role == "conditioning"]

    @property
    def temperatures(self) -> np.ndarray:
        return np.array([c.temperature for c in self.cat_columns], dtype=np.float32)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self.columns)

    def column(self, name: str) -> ColumnSpec:
        try:
            return self._by_name[name]
        except KeyError:
            raise SchemaError(f"unknown column {name!r}") from None

    def cat_index(self, name: str) -> int:
        return self._cat_pos[name]

    def num_index(self, name: str) -> int:
        return self._num_pos[name]

    def with_temperatures(self, temperatures: dict[str, float]) -> Schema:
        from dataclasses import replace

        cols = [replace(c, temperature=float(temperatures.get(c.name, c.temperature))) for c in self.columns]
        return Schema(cols, data_release=self.data_release, groups=self.groups, metadata=self.metadata)

    # ---- serialization -------------------------------------------------------------------------------
    def to_json(self) -> dict:
        obj: dict[str, Any] = {
            "schema_format": SCHEMA_FORMAT,
            "data_release": self.data_release,
            "missing": {"na_tokens": [""], "none_is_value": True},
            "columns": [c.to_json() for c in self.columns],
        }
        if self.groups:
            obj["groups"] = self.groups
        if self.metadata:
            obj["metadata"] = self.metadata
        return obj

    @classmethod
    def from_json(cls, obj: dict) -> Schema:
        fmt = obj.get("schema_format")
        if fmt != SCHEMA_FORMAT:
            raise SchemaError(f"unsupported schema_format {fmt!r} (expected {SCHEMA_FORMAT!r})")
        return cls(
            [ColumnSpec.from_json(c) for c in obj.get("columns", [])],
            data_release=obj.get("data_release"),
            groups=obj.get("groups"),
            metadata=obj.get("metadata"),
        )

    def schema_hash(self) -> str:
        """Hash of everything that affects model inputs/outputs (names, roles, types, vocab, transforms)."""
        return _sha256_json([c._hash_view() for c in self.columns])

    def vocab_hash(self) -> str:
        """Hash of names, types and vocabularies only — identical across all bundles of one data release."""
        return _sha256_json([c._hash_view(include_transform=False) for c in self.columns])

    # ---- encode / decode -----------------------------------------------------------------------------
    def encode(
        self,
        df: pd.DataFrame,
        *,
        unknown: Literal["missing", "error"] = "missing",
        ignore_columns: Iterable[str] = (),
    ) -> tuple[EncodedBatch, list[InputIssue]]:
        """Encode known values. Missing cells, absent columns and unrecognized labels become unknown."""
        n = len(df)
        issues: list[InputIssue] = []
        ignore = set(ignore_columns)
        for col in df.columns:
            if col not in self._by_name and col not in ignore:
                issues.append(InputIssue(-1, str(col), "", "unknown_column", "ignored"))

        cat = np.full((n, len(self.cat_columns)), -1, dtype=np.int64)
        cat_known = np.zeros((n, len(self.cat_columns)), dtype=bool)
        for j, spec in enumerate(self.cat_columns):
            if spec.name not in df.columns:
                continue
            raw = df[spec.name]
            canon = raw.map(canonical_label)
            present = canon.notna().to_numpy()
            codes = canon.map(self._exact[j])
            unmatched = present & codes.isna().to_numpy()
            if unmatched.any():
                retry = canon[unmatched].str.lower().map(self._lower[j])
                codes.loc[retry.index] = retry
                still = present & codes.isna().to_numpy()
                for i in np.flatnonzero(still):
                    nearest = self._nearest_integer_option(j, canon.iloc[i])
                    if nearest is not None:
                        # e.g. 3 attached units when the model only saw 5, 6, 8, ...: condition on the closest option
                        codes.iloc[i] = nearest
                        action = f"used_nearest_model_option_{spec.vocab[nearest]}"
                        issues.append(InputIssue(int(i), spec.name, str(canon.iloc[i]), "coerced", action))
                    else:
                        issues.append(
                            InputIssue(int(i), spec.name, str(raw.iloc[i]), "unknown_label", "treated_as_missing")
                        )
            ok = codes.notna().to_numpy()
            cat[ok, j] = codes.to_numpy()[ok].astype(np.int64)
            cat_known[:, j] = ok

        num = np.zeros((n, len(self.num_columns)), dtype=np.float32)
        num_known = np.zeros((n, len(self.num_columns)), dtype=bool)
        for j, spec in enumerate(self.num_columns):
            if spec.name not in df.columns:
                continue
            raw = df[spec.name]
            present = raw.map(canonical_label).notna().to_numpy()
            values = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64)
            ok = present & np.isfinite(values)
            for i in np.flatnonzero(present & ~ok):
                issues.append(InputIssue(int(i), spec.name, str(raw.iloc[i]), "not_numeric", "treated_as_missing"))
            if spec.clip is not None:
                values = np.clip(values, spec.clip[0], spec.clip[1])
            transformed = np.zeros(n, dtype=np.float64)
            if ok.any():
                transformed[ok] = spec.transform.forward(values[ok])
            num[:, j] = transformed.astype(np.float32)
            num_known[:, j] = ok

        if unknown == "error":
            bad = [i for i in issues if i.kind not in ("unknown_column", "coerced")]
            if bad:
                preview = "; ".join(f"row {i.row} {i.column}={i.value!r}" for i in bad[:5])
                raise InputValidationError(f"{len(bad)} input value(s) are not valid options: {preview}")
        return EncodedBatch(cat, cat_known, num, num_known), issues

    def _nearest_integer_option(self, j: int, label: str) -> int | None:
        """Vocab index of the closest option for an unseen integer label in an integer-valued column, else None.

        Applies only when (nearly) every non-"None" option of the column is an integer (unit counts, stories, ...).
        """
        if not _INTEGER_RE.match(label):
            return None
        cache = self.__dict__.setdefault("_integer_options", {})
        if j not in cache:
            vocab = self.cat_columns[j].vocab
            pairs = [(int(v), i) for i, v in enumerate(vocab) if _INTEGER_RE.match(v)]
            n_real = sum(v != "None" for v in vocab)
            cache[j] = pairs if len(pairs) >= 3 and len(pairs) >= 0.8 * n_real else []
        if not cache[j]:
            return None
        value = int(label)
        return min(cache[j], key=lambda p: (abs(p[0] - value), p[0]))[1]

    def decode(self, cat_codes: np.ndarray, num_values: np.ndarray | None = None) -> pd.DataFrame:
        """Decode codes (``-1`` -> missing) and transformed numeric values back to labels / data units."""
        out: dict[str, Any] = {}
        for j, spec in enumerate(self.cat_columns):
            vocab = np.asarray(spec.vocab, dtype=object)
            codes = np.asarray(cat_codes[:, j])
            vals = vocab[np.clip(codes, 0, len(vocab) - 1)]
            out[spec.name] = np.where(codes >= 0, vals, None)
        for j, spec in enumerate(self.num_columns):
            if num_values is None:
                out[spec.name] = np.full(len(cat_codes), np.nan)
                continue
            vals = spec.transform.inverse(np.asarray(num_values[:, j], dtype=np.float64))
            if spec.clip is not None:
                vals = np.clip(vals, spec.clip[0], spec.clip[1])
            out[spec.name] = vals
        n = len(cat_codes) if len(self.cat_columns) else (0 if num_values is None else len(num_values))
        return pd.DataFrame(out, columns=self.names, index=pd.RangeIndex(n))

    def inverse_numeric(self, name: str, z: np.ndarray) -> np.ndarray:
        spec = self.column(name)
        vals = spec.transform.inverse(z)
        return np.clip(vals, *spec.clip) if spec.clip is not None else vals


@dataclass
class SchemaSummary:
    n_categorical: int
    n_numeric: int
    imputable: list[str] = field(default_factory=list)
    conditioning: list[str] = field(default_factory=list)


def summarize(schema: Schema) -> SchemaSummary:
    return SchemaSummary(
        len(schema.cat_columns), len(schema.num_columns), schema.imputable_names, schema.conditioning_names
    )
