from __future__ import annotations

from usg.backends.base import RawOutput, TorchImputer
from usg.models import build_schedule
from usg.models.tabddpm import aggregate_samples, tabddpm_sample
from usg.schema import EncodedBatch


class TabDDPMImputer(TorchImputer):
    """Draws ``n_samples`` completions per building; probabilities are smoothed sample frequencies."""

    capabilities = frozenset({"probabilities", "sampling"})

    def __init__(self, bundle, net):
        super().__init__(bundle, net)
        self.schedule = build_schedule(self.config)

    def default_samples(self) -> int:
        return int(self.inference_config.get("default_samples", 32))

    def predict_encoded(self, batch: EncodedBatch, *, n_samples: int, generator, decode: str | None = None) -> RawOutput:
        torch = self._torch
        cat, known, num, num_known = self.to_tensors(batch)
        steps = int(self.inference_config.get("sample_steps", 100))
        draws = tabddpm_sample(
            self.net,
            self.schedule,
            cat,
            known,
            num,
            num_known,
            cat_imputable=self._cat_imputable,
            num_imputable=self._num_imputable,
            n_samples=n_samples,
            steps=steps,
            generator=generator,
        )
        hidden = self._cat_imputable.unsqueeze(0) & ~known
        agg = aggregate_samples(draws["cat"], draws["num"], self.net.cat_heads.valid, hidden, temperature=self._temperature)
        aggregate = decode or self.inference_config.get("aggregate", "medoid")
        codes = agg["mode"] if aggregate == "mode" else agg["medoid"]
        codes = torch.where(known, cat, codes)
        num_samples = draws["num"]
        if num_samples.shape[-1]:
            num_values = agg["num_mean"] if aggregate == "mode" else agg["num_medoid"]
            num_mean = agg["num_mean"]
            if n_samples > 1:
                num_lo = torch.quantile(num_samples.float(), 0.05, dim=1)
                num_hi = torch.quantile(num_samples.float(), 0.95, dim=1)
            else:
                num_lo = num_hi = num_mean
        else:
            num_values = num_mean = num_lo = num_hi = num
        return RawOutput(
            cat_codes=codes.cpu().numpy(),
            cat_probs=agg["probs"].float().cpu().numpy(),
            num_values=num_values.float().cpu().numpy(),
            num_mean=num_mean.float().cpu().numpy(),
            num_lo=num_lo.float().cpu().numpy(),
            num_hi=num_hi.float().cpu().numpy(),
            cat_samples=draws["cat"].cpu().numpy(),
            num_samples=num_samples.float().cpu().numpy(),
        )
