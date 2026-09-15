from __future__ import annotations

from usg.backends.base import RawOutput, TorchImputer
from usg.models.gnn import gnn_impute
from usg.schema import EncodedBatch

Z95 = 1.6448536


class GNNImputer(TorchImputer):
    """Single pass (or refinement) point prediction with softmax probabilities; ``n_samples > 1`` adds draws."""

    capabilities = frozenset({"probabilities", "sampling"})

    def predict_encoded(self, batch: EncodedBatch, *, n_samples: int, generator, decode: str | None = None) -> RawOutput:
        torch = self._torch
        cat, known, num, num_known = self.to_tensors(batch)
        mode = decode or self.inference_config.get("decode", "argmax")
        rounds = int(self.inference_config.get("rounds", 8))
        common = {
            "cat_imputable": self._cat_imputable,
            "num_imputable": self._num_imputable,
            "temperature": self._temperature,
            "rounds": rounds,
            "generator": generator,
        }
        out = gnn_impute(self.net, cat, known, num, num_known, mode=mode, **common)
        num_mean = out.get("num_mean", num)
        num_std = out.get("num_std", torch.zeros_like(num))
        raw = RawOutput(
            cat_codes=out["cat_codes"].cpu().numpy(),
            cat_probs=out["cat_probs"].float().cpu().numpy(),
            num_values=out.get("num_value", num).float().cpu().numpy(),
            num_mean=num_mean.float().cpu().numpy(),
            num_lo=(num_mean - Z95 * num_std).float().cpu().numpy(),
            num_hi=(num_mean + Z95 * num_std).float().cpu().numpy(),
        )
        if n_samples > 1:
            rep = lambda x: x.repeat_interleave(n_samples, dim=0)
            draws = gnn_impute(self.net, rep(cat), rep(known), rep(num), rep(num_known), mode="sample", **common)
            b = cat.shape[0]
            raw.cat_samples = draws["cat_codes"].view(b, n_samples, -1).cpu().numpy()
            if "num_value" in draws:
                raw.num_samples = draws["num_value"].view(b, n_samples, -1).float().cpu().numpy()
        return raw
