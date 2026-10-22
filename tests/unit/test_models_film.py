"""G-NN arch v2: FiLM conditioning on a few categorical columns.

Conditioning entered through ``ColumnEmbedder`` is one slice of a wide concatenation, so the backbone can
only use it additively. Arch v2 also routes chosen columns through a :class:`ConditionEncoder` whose output
scales and shifts every residual block, which is how location behaves in the stock: the same floor area
implies different equipment in different places.
"""

from __future__ import annotations

import pytest
import torch

from usg.errors import CompatibilityError
from usg.models import SUPPORTED_ARCH, build_model
from usg.testing.fixtures import tiny_config, tiny_schema

FILM = ["State", "Geometry Building Type RECS", "Vintage"]


def _v2(cfg: dict, columns: list[str] | None = None) -> dict:
    return {**cfg, "arch_version": 2, "arch": {**cfg["arch"], "film_columns": FILM if columns is None else columns}}


def _batch(schema, n: int = 7, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    cat = torch.stack([torch.randint(0, k, (n,), generator=g) for k in schema.cardinalities], dim=1)
    known = torch.rand(n, len(schema.cardinalities), generator=g) < 0.5
    num = torch.randn(n, len(schema.num_columns), generator=g)
    num_known = torch.rand(n, len(schema.num_columns), generator=g) < 0.5
    return cat, known, num, num_known


def test_arch_2_is_released_for_gnn_only():
    assert 2 in SUPPORTED_ARCH["gnn"] and 1 in SUPPORTED_ARCH["gnn"]
    assert SUPPORTED_ARCH["tabddpm"] == (1,)


def test_v1_checkpoint_still_loads_strict():
    """The shipped bundles are arch v1; adding v2 must not change their module tree."""
    schema, cfg = tiny_schema(), tiny_config("gnn")
    state = build_model(schema, cfg).state_dict()
    build_model(schema, cfg).load_state_dict(state, strict=True)


def test_film_starts_as_an_exact_no_op():
    """FiLM is zero-initialised, so a v2 run begins exactly where v1 begins."""
    torch.manual_seed(0)
    schema, cfg = tiny_schema(), tiny_config("gnn")
    v1 = build_model(schema, cfg).eval()
    v2 = build_model(schema, _v2(cfg)).eval()
    missing, unexpected = v2.load_state_dict(v1.state_dict(), strict=False)
    assert unexpected == [] and missing, "v2 adds parameters and reuses every v1 one"
    with torch.no_grad():
        batch = _batch(schema)
        assert torch.allclose(v1(*batch)["cat_logits"], v2(*batch)["cat_logits"], atol=1e-6)


def test_conditioning_modulates_every_block():
    schema = tiny_schema()
    net = build_model(schema, _v2(tiny_config("gnn")))
    assert net.cond is not None
    assert all(block.film is not None for block in net.backbone.blocks)
    assert net.cond.out_dim == net.backbone.blocks[0].film.in_features


def test_unknown_conditioning_degrades_to_a_learned_default():
    """A caller may leave any column blank, so FiLM must run on a masked conditioner, not crash."""
    torch.manual_seed(0)
    schema = tiny_schema()
    net = build_model(schema, _v2(tiny_config("gnn"))).eval()
    net.cond.mlp[0].weight.data.normal_(0, 0.5)  # lift it off the zero init so the two paths can differ
    cat, known, num, num_known = _batch(schema)
    blank = known.clone()
    for name in FILM:
        blank[:, schema.cat_index(name)] = False
    with torch.no_grad():
        given = net(cat, torch.ones_like(known), num, num_known)["cat_logits"]
        absent = net(cat, blank, num, num_known)["cat_logits"]
    assert torch.isfinite(absent).all()
    assert not torch.allclose(given, absent), "the conditioner must actually read its columns"


def test_film_columns_are_rejected_on_arch_1():
    """Silently ignoring them would serve a model missing the conditioning it was configured for."""
    cfg = tiny_config("gnn")
    with pytest.raises(CompatibilityError, match="arch_version >= 2"):
        build_model(tiny_schema(), {**cfg, "arch": {**cfg["arch"], "film_columns": FILM}})


@pytest.mark.parametrize(
    ("columns", "match"),
    [(["No Such Column"], "absent from the schema"), (["sqft"], "must be categorical")],
)
def test_bad_film_columns_are_rejected(columns, match):
    with pytest.raises(CompatibilityError, match=match):
        build_model(tiny_schema(), _v2(tiny_config("gnn"), columns))


def test_a_masked_conditioning_column_cannot_leak_its_own_label():
    """FiLM is useful on columns the model also predicts, which is only sound if it reads the masked input.

    The raw ``cat`` tensor carries true values for every column; what makes a hidden column hidden is the
    ``cat_known`` gate. If the conditioner read around that gate it would see the answer while predicting
    it, and validation accuracy would rise for a reason that does not exist at inference time.
    """
    torch.manual_seed(0)
    schema = tiny_schema()
    net = build_model(schema, _v2(tiny_config("gnn"))).eval()
    for module in net.cond.modules():  # lift every FiLM weight off its zero init
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(0, 0.5)

    cat, known, num, num_known = _batch(schema)
    hidden = schema.cat_index("Vintage")
    known[:, hidden] = False
    tampered = cat.clone()
    tampered[:, hidden] = (cat[:, hidden] + 1) % schema.cardinalities[hidden]

    with torch.no_grad():
        before = net(cat, known, num, num_known)["cat_logits"]
        after = net(tampered, known, num, num_known)["cat_logits"]
    assert torch.equal(before, after), "a hidden column's true value changed the output: label leakage"
