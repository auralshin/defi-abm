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

from defi_abm.agents.lending import DeFiLendingAgent


class DummyLendingModel(Model):
    def __init__(self, price: float = 100.0, collateral_factor: float = 0.75, rate_per_block: float = 0.0):
        super().__init__()
        self._mesa_model = Model()
        self.current_price = price
        self.collateral_factor = collateral_factor
        self._rate = rate_per_block
        self.loans = []
        self.registered = []

    def get_lending_rate(self, token: str) -> float:
        return self._rate

    def register_loan(self, agent):
        self.registered.append(agent)
        self.loans.append(agent)


def test_lending_initial_borrow_and_registration():
    dummy_model = DummyLendingModel(price=100.0, collateral_factor=0.75, rate_per_block=0.0)
    agent = DeFiLendingAgent(
        model=dummy_model,
        collateral_token="ETH",
        borrow_token="DAI",
        collateral_amount=2.0,
        desired_ltv=0.5,
    )
    agent.step()
    assert agent.borrow_amount == pytest.approx(100.0)
    assert agent in dummy_model.loans


def test_lending_health_and_flag_for_liquidation():
    dummy = DummyLendingModel(price=100.0, collateral_factor=0.5, rate_per_block=0.0)
    agent = DeFiLendingAgent(
        model=dummy,
        collateral_token="ETH",
        borrow_token="DAI",
        collateral_amount=1.0,
        desired_ltv=0.75,
    )
    agent.step()
    assert agent.borrow_amount == pytest.approx(75.0)
    assert agent.is_marked_for_liquidation is True


def test_accrue_interest_over_steps():
    dummy = DummyLendingModel(price=100.0, collateral_factor=0.75, rate_per_block=0.10)
    agent = DeFiLendingAgent(
        model=dummy,
        collateral_token="ETH",
        borrow_token="DAI",
        collateral_amount=1.0,
        desired_ltv=0.5,
    )
    agent.borrow()
    assert agent.borrow_amount == pytest.approx(50.0)
    agent.step()
    assert agent.borrow_amount == pytest.approx(55.0)
    agent.step()
    assert agent.borrow_amount == pytest.approx(60.5)


def test_no_liquidation_when_within_collateral_factor():
    dummy = DummyLendingModel(price=100.0, collateral_factor=0.75, rate_per_block=0.0)
    agent = DeFiLendingAgent(
        model=dummy,
        collateral_token="ETH",
        borrow_token="DAI",
        collateral_amount=2.0,
        desired_ltv=0.5,
    )
    agent.step()
    assert agent.borrow_amount == pytest.approx(100.0)
    assert agent.is_marked_for_liquidation is False


class DummyIRMModel(Model):
    def __init__(self, price=100.0, collateral_factor=0.75):
        super().__init__()
        self.current_price = price
        self.collateral_factor = collateral_factor
        self.total_cash = 1000.0
        self.total_borrowed = 0.0

    def register_loan(self, agent):
        self.total_borrowed += agent.borrow_amount

    def get_lending_rate(self, token: str) -> float:
        raise NotImplementedError("Use internal IRM for testing")


def test_irm_fixed_rate():
    dummy = DummyIRMModel(price=100.0, collateral_factor=0.75)
    util_fn = lambda: 0.2
    agent = DeFiLendingAgent(
        model=dummy,
        collateral_token="ETH",
        borrow_token="DAI",
        collateral_amount=1.0,
        desired_ltv=0.5,
        irm_mode="fixed",
        irm_params={"rate": 0.10},
        utilization_model=util_fn,
    )
    agent.borrow()
    assert agent.borrow_amount == pytest.approx(50.0)
    agent.step()
    assert agent.borrow_amount == pytest.approx(55.0)


def test_irm_linear_rate_with_utilization():
    dummy = DummyIRMModel(price=100.0, collateral_factor=0.75)
    util_fn = lambda: 0.5
    agent = DeFiLendingAgent(
        model=dummy,
        collateral_token="ETH",
        borrow_token="DAI",
        collateral_amount=1.0,
        desired_ltv=0.5,
        irm_mode="linear",
        irm_params={"base": 0.02, "slope": 0.10},
        utilization_model=util_fn,
    )
    agent.borrow()
    assert agent.borrow_amount == pytest.approx(50.0)
    agent.step()
    assert agent.borrow_amount == pytest.approx(50.0 * 1.07)


def test_irm_kinked_rate_behavior():
    dummy = DummyIRMModel(price=100.0, collateral_factor=0.75)
    util_fn = lambda: 0.9
    agent = DeFiLendingAgent(
        model=dummy,
        collateral_token="ETH",
        borrow_token="DAI",
        collateral_amount=1.0,
        desired_ltv=0.5,
        irm_mode="kinked",
        irm_params={"base": 0.01, "slope1": 0.05, "slope2": 0.20, "kink": 0.8},
        utilization_model=util_fn,
    )
    agent.borrow()
    assert agent.borrow_amount == pytest.approx(50.0)
    agent.step()
    assert agent.borrow_amount == pytest.approx(50.0 * 1.07)
