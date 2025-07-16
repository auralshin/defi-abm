import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import importlib
for m in list(sys.modules):
    if m.startswith("defi_abm"):
        sys.modules.pop(m)
import defi_abm
importlib.reload(defi_abm)

from defi_abm.agents.oracle import OracleAgent


class DummyOracleModel:
    def __init__(self, seed=None):
        self.steps = 1
        self.current_price = None
        self.seed = seed
        self.random = np.random.default_rng(seed)

    def register_agent(self, agent):
        pass


def test_oracle_with_csv_series_updates_price_correctly(tmp_path):
    prices = pd.Series([10.0, 20.0, 30.0])
    df = pd.DataFrame({"price": prices})
    csv_path = tmp_path / "price.csv"
    df.to_csv(csv_path, index=False)

    dummy_model = DummyOracleModel()
    csv_series = pd.read_csv(csv_path)["price"]
    agent = OracleAgent(model=dummy_model, price_series=csv_series, mode="csv")

    dummy_model.steps = 1
    agent.step()
    assert dummy_model.current_price == pytest.approx(10.0)

    dummy_model.steps = 2
    agent.step()
    assert dummy_model.current_price == pytest.approx(20.0)

    dummy_model.steps = 6
    agent.step()
    assert dummy_model.current_price == pytest.approx(30.0)


def test_oracle_static_mode_always_returns_same_price():
    dummy_model = DummyOracleModel()
    agent = OracleAgent(model=dummy_model, price_series=None, mode="static", static_price=5.0)

    for i in range(3):
        dummy_model.steps = i
        agent.step()
        assert dummy_model.current_price == pytest.approx(5.0)


def test_oracle_gbm_mode_generates_random_walk():
    dummy_model = DummyOracleModel(seed=42)
    initial_price = 50.0
    prices = pd.Series([initial_price])
    gbm_params = {"mu": 0.1, "sigma": 0.2, "dt": 1.0}

    agent = OracleAgent(
        model=dummy_model,
        price_series=prices,
        mode="gbm",
        gbm_params=gbm_params,
        static_price=initial_price,
    )

    dummy_model.steps = 1
    agent.step()
    first_price = dummy_model.current_price
    assert first_price > 0.0
    assert isinstance(first_price, float)
