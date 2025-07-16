import sys
from pathlib import Path
import copy
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import importlib
for m in list(sys.modules):
    if m.startswith("defi_abm"):
        sys.modules.pop(m)
import defi_abm
importlib.reload(defi_abm)

from defi_abm.models.defi_model import DeFiModel


def make_temp_price_csv(tmp_path, prices):
    df = pd.DataFrame({"price": prices})
    csv_path = tmp_path / "price.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def simple_config(tmp_path):
    price_csv_path = make_temp_price_csv(tmp_path, [100.0, 90.0, 80.0, 70.0])
    config = {
        "simulation": {"steps": 2, "seed": 123},
        "protocols": {
            "oracle": {"price_csv": price_csv_path, "initial_price": 100.0, "mode": "csv"},
            "amm": [{"token_x": "ETH", "token_y": "DAI", "reserve_x": 5.0, "reserve_y": 1000.0, "fee_rate": 0.0}],
            "lending": {
                "collateral_factor": 0.75,
                "interest_rate_apr": 0.0,
                "blocks_per_year": 1,
                "liquidation_penalty": 0.0,
                "agents": [
                    {"collateral_token": "ETH", "borrow_token": "DAI", "collateral_amount": 1.0, "desired_ltv": 0.5}
                ],
            },
        },
    }
    return config


def test_model_initialization_creates_expected_agents(simple_config):
    model = DeFiModel(simple_config)
    all_agents = list(model.agents)
    assert len(all_agents) == 4
    names = {a.__class__.__name__ for a in all_agents}
    assert "OracleAgent" in names
    assert "AMMAgent" in names
    assert "DeFiLendingAgent" in names
    assert "LiquidatorAgent" in names


def test_model_step_updates_price_and_metrics(simple_config):
    model = DeFiModel(simple_config)
    model.step()
    assert model.current_price == pytest.approx(100.0)
    df1 = model.datacollector.get_model_vars_dataframe()
    assert df1.shape[0] == 1
    assert df1["TVL"].iloc[0] == pytest.approx(100.0)
    assert df1["Num_Loans"].iloc[0] == 1

    model.step()
    assert model.current_price == pytest.approx(90.0)
    df2 = model.datacollector.get_model_vars_dataframe()
    assert df2.shape[0] == 2
    assert df2["TVL"].iloc[1] == pytest.approx(90.0)
    assert df2["Num_Loans"].iloc[1] == 1


def test_model_liquidation_on_price_drop(simple_config):
    cfg = copy.deepcopy(simple_config)
    cfg["protocols"]["lending"]["collateral_factor"] = 0.4

    model = DeFiModel(cfg)
    model.step()

    assert len(model.loans) == 0
    assert "liquidations" in model.metrics
    assert len(model.metrics["liquidations"]) == 1

    df = model.datacollector.get_model_vars_dataframe()
    assert df["TVL"].iloc[0] < 100.0
