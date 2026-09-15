import pytest
import torch

from usg.models import build_model, build_schedule
from usg.models.gnn import gnn_impute
from usg.models.layers import PAD_LOGIT
from usg.models.tabddpm import aggregate_samples, tabddpm_sample
from usg.testing.fixtures import tiny_config, tiny_dataframe, tiny_schema


def _encoded(n=16, drop=("Vintage", "HVAC Heating Type", "Geometry Garage", "sqft")):
    schema = tiny_schema()
    df = tiny_dataframe(n, seed=1).drop(columns=list(drop))
    batch, _ = schema.encode(df)
    t = lambda a: torch.as_tensor(a)
    return schema, t(batch.cat), t(batch.cat_known), t(batch.num), t(batch.num_known)


@pytest.mark.parametrize("mode", ["argmax", "refine", "sample"])
def test_gnn_impute_fills_only_hidden_imputable(mode):
    torch.manual_seed(0)
    schema, cat, known, num, num_known = _encoded()
    net = build_model(schema, tiny_config("gnn")).eval()
    out = net(cat, known, num, num_known)
    assert out["cat_logits"].shape == (16, len(schema.cat_columns), schema.max_cardinality)
    state = schema.cat_index("State")
    assert torch.all(out["cat_logits"][:, state] == PAD_LOGIT)  # conditioning column has no head
    res = gnn_impute(
        net, cat, known, num, num_known,
        cat_imputable=torch.as_tensor(schema.cat_imputable),
        num_imputable=torch.as_tensor(schema.num_imputable),
        mode=mode, rounds=3, generator=torch.Generator().manual_seed(0),
    )
    codes = res["cat_codes"]
    assert torch.equal(codes[known], cat[known])
    imputable = torch.as_tensor(schema.cat_imputable)
    hidden = imputable & ~known
    assert torch.all(codes[hidden] >= 0)
    cards = torch.tensor(schema.cardinalities)
    assert torch.all(codes[hidden] < cards.expand_as(codes)[hidden])
    probs = res["cat_probs"]
    torch.testing.assert_close(probs[hidden].sum(-1), torch.ones(int(hidden.sum())), atol=1e-5, rtol=0)
    assert res["num_value"].shape == num.shape


def test_tabddpm_sample_keeps_known_and_aggregates():
    torch.manual_seed(0)
    schema, cat, known, num, num_known = _encoded(n=6)
    config = tiny_config("tabddpm")
    net = build_model(schema, config).eval()
    sched = build_schedule(config)
    imputable = torch.as_tensor(schema.cat_imputable)
    out = tabddpm_sample(
        net, sched, cat, known, num, num_known,
        cat_imputable=imputable, num_imputable=torch.as_tensor(schema.num_imputable),
        n_samples=5, steps=7, generator=torch.Generator().manual_seed(0),
    )
    samples = out["cat"]
    assert samples.shape == (6, 5, len(schema.cat_columns))
    assert torch.equal(samples[:, :, :][known.unsqueeze(1).expand_as(samples)], cat.unsqueeze(1).expand_as(samples)[known.unsqueeze(1).expand_as(samples)])
    hidden = imputable & ~known
    agg = aggregate_samples(samples, out["num"], net.cat_heads.valid, hidden)
    torch.testing.assert_close(agg["probs"][hidden].sum(-1), torch.ones(int(hidden.sum())), atol=1e-5, rtol=0)
    for row in range(6):
        assert any(torch.equal(agg["medoid"][row], samples[row, k]) for k in range(5))


def test_same_seed_same_samples():
    schema, cat, known, num, num_known = _encoded(n=4)
    config = tiny_config("tabddpm")
    torch.manual_seed(0)
    net = build_model(schema, config).eval()
    sched = build_schedule(config)
    kwargs = {"cat_imputable": torch.as_tensor(schema.cat_imputable),
              "num_imputable": torch.as_tensor(schema.num_imputable), "n_samples": 3, "steps": 5}
    a = tabddpm_sample(net, sched, cat, known, num, num_known, generator=torch.Generator().manual_seed(7), **kwargs)
    b = tabddpm_sample(net, sched, cat, known, num, num_known, generator=torch.Generator().manual_seed(7), **kwargs)
    assert torch.equal(a["cat"], b["cat"]) and torch.allclose(a["num"], b["num"])
