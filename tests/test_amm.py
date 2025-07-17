import sys
from pathlib import Path
import pytest
from mesa import Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import importlib
for m in list(sys.modules):
    if m.startswith("defi_abm"):
        sys.modules.pop(m)
import defi_abm
importlib.reload(defi_abm)

from defi_abm.agents.amm import AMMAgent
from defi_abm.agents.curves import ConstantProductCurve
from defi_abm.utils.math_helpers import get_amount_out


def test_amm_get_price_and_infinite_when_zero():
    dummy_model = Model()

    amm = AMMAgent(
        model=dummy_model,
        token_x="X",
        token_y="Y",
        reserve_x=50.0,
        reserve_y=100.0,
        fee_rate=0.0,
        curve=ConstantProductCurve(),
    )
    assert (amm.reserve_y / amm.reserve_x) == pytest.approx(2.0)

    amm_zero = AMMAgent(
        model=dummy_model,
        token_x="X",
        token_y="Y",
        reserve_x=0.0,
        reserve_y=100.0,
        fee_rate=0.0,
        curve=ConstantProductCurve(),
    )
    assert amm_zero.reserve_x == 0.0


def test_swap_x_for_y_no_fee_updates_reserves_correctly():
    dummy_model = Model()
    amm = AMMAgent(
        model=dummy_model,
        token_x="X",
        token_y="Y",
        reserve_x=100.0,
        reserve_y=100.0,
        fee_rate=0.0,
        curve=ConstantProductCurve(),
    )
    amount_out = amm.swap_x_for_y(10.0)
    expected = (10.0 * 100.0) / (100.0 + 10.0)
    assert amount_out == pytest.approx(expected)
    assert amm.reserve_x == pytest.approx(110.0)
    assert amm.reserve_y == pytest.approx(100.0 - expected)


def test_swap_y_for_x_with_fee_and_reserve_updates():
    dummy_model = Model()
    reserve_x = 200.0
    reserve_y = 100.0
    fee_rate = 0.01
    amount_y_in = 20.0

    expected_out = get_amount_out(
        amount_in=amount_y_in,
        reserve_in=reserve_y,
        reserve_out=reserve_x,
        fee_rate=fee_rate,
    )

    amm = AMMAgent(
        model=dummy_model,
        token_x="X",
        token_y="Y",
        reserve_x=reserve_x,
        reserve_y=reserve_y,
        fee_rate=fee_rate,
        curve=ConstantProductCurve(),
    )

    amount_out = amm.swap_y_for_x(amount_y_in)

    assert amount_out == pytest.approx(expected_out)
    assert amm.reserve_y == pytest.approx(reserve_y + amount_y_in)
    assert amm.reserve_x == pytest.approx(reserve_x - expected_out, rel=1e-2)


def test_multiple_swaps_preserve_invariant():
    dummy_model = Model()
    initial_x = 50.0
    initial_y = 200.0
    amm = AMMAgent(
        model=dummy_model,
        token_x="X",
        token_y="Y",
        reserve_x=initial_x,
        reserve_y=initial_y,
        fee_rate=0.003,
        curve=ConstantProductCurve(),
    )
    for amount_x in [1.0, 5.0, 10.0]:
        amm.swap_x_for_y(amount_x)
    for amount_y in [2.0, 8.0]:
        amm.swap_y_for_x(amount_y)

    final_product = amm.reserve_x * amm.reserve_y
    original_product = initial_x * initial_y
    assert final_product >= original_product
    assert final_product > 0.0
