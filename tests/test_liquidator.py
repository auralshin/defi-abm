import sys
from pathlib import Path
import pytest
from mesa import Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import importlib
import defi_abm
importlib.reload(defi_abm)

from defi_abm.agents.amm import AMMAgent, ConstantProductCurve
from defi_abm.agents.liquidator import LiquidatorAgent


class DummyLendingAgent:
    def __init__(self, unique_id, collateral_amount, borrow_amount, collateral_token, borrow_token):
        self.unique_id = unique_id
        self.collateral_amount = float(collateral_amount)
        self.borrow_amount = float(borrow_amount)
        self.is_marked_for_liquidation = True
        self.collateral_token = collateral_token
        self.borrow_token = borrow_token

    def remove(self):
        pass


class DummyLiquidationModel:
    def __init__(self, current_price: float):
        self.current_price = current_price
        self.loans = []
        self.metrics = {}
        self._mesa_model = Model()
        self.amm_pool = AMMAgent(
            model=self._mesa_model,
            token_x="ETH",
            token_y="DAI",
            reserve_x=10.0,
            reserve_y=2000.0,
            fee_rate=0.0,
            curve=ConstantProductCurve(),
        )

    def find_amm_pool(self, token_x: str, token_y: str):
        if self.amm_pool.token_x == token_x and self.amm_pool.token_y == token_y:
            return self.amm_pool
        return None


def test_liquidator_clears_debt_and_reduces_collateral():
    model = DummyLiquidationModel(current_price=100.0)
    model.steps = 1
    lending_agent = DummyLendingAgent(
        unique_id=1,
        collateral_amount=1.0,
        borrow_amount=100.0,
        collateral_token="ETH",
        borrow_token="DAI",
    )
    model.loans.append(lending_agent)

    mesa_model = Model()
    liquidator = LiquidatorAgent(model=mesa_model, liquidation_penalty=0.05)
    liquidator.model = model

    assert lending_agent.borrow_amount == pytest.approx(100.0)
    assert lending_agent.collateral_amount == pytest.approx(1.0)

    liquidator.step()

    assert lending_agent.borrow_amount == pytest.approx(0.0)
    assert lending_agent.collateral_amount == pytest.approx(0.0)
    assert lending_agent not in model.loans

    assert "liquidations" in model.metrics
    event = model.metrics["liquidations"][0]
    expected_recovered = 2000.0 / 11.0
    assert event["agent_id"] == lending_agent.unique_id
    assert event["collateral_seized"] == pytest.approx(1.0)
    assert event["recovered_amount"] == pytest.approx(expected_recovered)
    assert event["debt_repaid"] == pytest.approx(100.0)
    assert event["penalty_paid"] == pytest.approx(5.0)
    assert event["remaining_debt"] == pytest.approx(0.0)
    assert event["timestamp"] > 0


def test_liquidator_handles_no_amm_pool():
    class NoAMMModel(DummyLiquidationModel):
        def find_amm_pool(self, token_x: str, token_y: str):
            return []

    model = NoAMMModel(current_price=50.0)
    model.steps = 1
    lending_agent = DummyLendingAgent(
        unique_id=2,
        collateral_amount=2.0,
        borrow_amount=50.0,
        collateral_token="ETH",
        borrow_token="DAI",
    )
    model.loans.append(lending_agent)

    mesa_model = Model()
    liquidator = LiquidatorAgent(model=mesa_model, liquidation_penalty=0.10)
    liquidator.model = model

    liquidator.step()
    assert lending_agent.borrow_amount == pytest.approx(50.0)
    assert lending_agent.collateral_amount == pytest.approx(0.9)
    assert "liquidations" in model.metrics
    event = model.metrics["liquidations"][0]
    assert event["recovered_amount"] == pytest.approx(0.0)
