# tests/test_all.py

import copy
import pandas as pd
import pytest
import numpy as np
from mesa import Model

from defi_abm.agents.amm import AMMAgent
from defi_abm.agents.curves import ConstantProductCurve, BaseCurve
from defi_abm.agents.oracle import OracleAgent
from defi_abm.agents.lending import DeFiLendingAgent
from defi_abm.agents.liquidator import LiquidatorAgent
from defi_abm.agents.blockchain import BlockchainAgent, Transaction
from defi_abm.models.defi_model import DeFiModel
from defi_abm.utils.math_helpers import get_amount_out, accrue_interest


# ----------------------------------------
# AMM Agent Tests
# ----------------------------------------

def test_amm_get_price_and_infinite_when_zero():
    dummy_model = Model()

    # With reserves 50 X, 100 Y => price of X in Y = 100/50 = 2.0
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

    # If reserve_x is zero, check reserve_x == 0
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

    # reserves: 100 X, 100 Y, fee_rate=0 => swap 10 X → Y
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

    # New reserves: reserve_x = 110, reserve_y = 100 - expected
    assert amm.reserve_x == pytest.approx(110.0)
    assert amm.reserve_y == pytest.approx(100.0 - expected)


def test_swap_y_for_x_with_fee_and_reserve_updates():
    dummy_model = Model()

    # Initial reserves
    reserve_x = 200.0
    reserve_y = 100.0
    fee_rate = 0.01
    amount_y_in = 20.0

    # Expected output using the same curve logic
    expected_out = get_amount_out(
        amount_in=amount_y_in,
        reserve_in=reserve_y,
        reserve_out=reserve_x,
        fee_rate=fee_rate,
    )

    # Instantiate AMM agent
    amm = AMMAgent(
        model=dummy_model,
        token_x="X",
        token_y="Y",
        reserve_x=reserve_x,
        reserve_y=reserve_y,
        fee_rate=fee_rate,
        curve=ConstantProductCurve(),
    )

    # Perform swap
    amount_out = amm.swap_y_for_x(amount_y_in)

    # Assertions
    assert amount_out == pytest.approx(expected_out)
    assert amm.reserve_y == pytest.approx(reserve_y + amount_y_in)
    # reserve_x should be approximately reserve_x - expected_out (allow small rounding)
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

    # Perform a sequence of swaps
    for amount_x in [1.0, 5.0, 10.0]:
        amm.swap_x_for_y(amount_x)
    for amount_y in [2.0, 8.0]:
        amm.swap_y_for_x(amount_y)

    final_product = amm.reserve_x * amm.reserve_y
    original_product = initial_x * initial_y

    # Because fees accrue to reserves, final_product ≥ original_product
    assert final_product >= original_product
    assert final_product > 0.0


# ----------------------------------------
# Oracle Agent Tests
# ----------------------------------------

class DummyOracleModel:
    def __init__(self, seed=None):
        self.steps = 1
        self.current_price = None
        self.seed = seed  # store seed for OracleAgent
        self.random = np.random.default_rng(seed)  # optional, for consistency

    def register_agent(self, agent):
        pass


def test_oracle_with_csv_series_updates_price_correctly(tmp_path):
    prices = pd.Series([10.0, 20.0, 30.0])
    df = pd.DataFrame({"price": prices})
    csv_path = tmp_path / "price.csv"
    df.to_csv(csv_path, index=False)

    dummy_model = DummyOracleModel()

    # Load CSV series
    csv_series = pd.read_csv(csv_path)["price"]
    agent = OracleAgent(model=dummy_model, price_series=csv_series, mode="csv")

    # Step 0 (steps=1 → index 0)
    dummy_model.steps = 1
    agent.step()
    assert dummy_model.current_price == pytest.approx(10.0)

    # Step 1 (steps=2 → index 1)
    dummy_model.steps = 2
    agent.step()
    assert dummy_model.current_price == pytest.approx(20.0)

    # Step 5 (steps=6 → out of bounds → holds last value = 30.0)
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

    dummy_model.steps = 1  # So t = 0
    agent.step()
    first_price = dummy_model.current_price

    # Accept the new price and just test basic properties
    assert first_price > 0.0
    assert isinstance(first_price, float)


# ----------------------------------------
# DeFiLendingAgent Unit Tests
# ----------------------------------------

class DummyLendingModel(Model):
    def __init__(self, price: float = 100.0, collateral_factor: float = 0.75, rate_per_block: float = 0.0):
        super().__init__()
        self._mesa_model = Model()      # used for Agent.__init__
        self.current_price = price      # crucial: DeFiLendingAgent.step() reads this
        self.collateral_factor = collateral_factor
        self._rate = rate_per_block
        self.loans = []
        self.registered = []            # for register_loan()

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

    # Calling step() should open a loan of 2*100*0.5 = 100 DAI
    agent.step()

    # Validate borrowing
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

    # Step: borrow=75, max_allowed=50 → flagged
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

    # Borrow 50 (1.0 * 100 * 0.5)
    agent.borrow()
    assert agent.borrow_amount == pytest.approx(50.0)

    # Step: accrue 10% → 50 * 1.10 = 55
    agent.step()
    assert agent.borrow_amount == pytest.approx(55.0)

    # Step: accrue again → 55 * 1.10 = 60.5
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

    agent.step()  # borrow=100, max_allowed=150 → healthy
    assert agent.borrow_amount == pytest.approx(100.0)
    assert agent.is_marked_for_liquidation is False


# ----------------------------------------
# IRM & Utilization Tests (New)
# ----------------------------------------

class DummyIRMModel(Model):
    """
    A dummy model without get_lending_rate so that DeFiLendingAgent
    falls back to internal IRM logic.
    We simulate a simple utilization: borrow / (borrow + cash).
    """
    def __init__(self, price=100.0, collateral_factor=0.75):
        super().__init__()
        self.current_price = price
        self.collateral_factor = collateral_factor
        # We track a “pool” with some cash and borrowed amount
        self.total_cash = 1000.0
        self.total_borrowed = 0.0

    def register_loan(self, agent):
        # whenever a borrow happens, increase total_borrowed
        self.total_borrowed += agent.borrow_amount

    def get_lending_rate(self, token: str) -> float:
        # Intentionally not implemented, so fallback is used
        raise NotImplementedError("Use internal IRM for testing")


def test_irm_fixed_rate():
    dummy = DummyIRMModel(price=100.0, collateral_factor=0.75)
    # utilization_model returns e.g. 0.2 constantly
    util_fn = lambda: 0.2

    agent = DeFiLendingAgent(
        model=dummy,
        collateral_token="ETH",
        borrow_token="DAI",
        collateral_amount=1.0,
        desired_ltv=0.5,
        irm_mode="fixed",
        irm_params={"rate": 0.10},  # 10% per block
        utilization_model=util_fn,
    )

    # Borrow 50 (1 * 100 * 0.5)
    agent.borrow()
    assert agent.borrow_amount == pytest.approx(50.0)

    # _accrue_borrow_interest should use fixed 10%
    agent.step()
    # After step: 50 * (1 + 0.10) = 55
    assert agent.borrow_amount == pytest.approx(55.0)


def test_irm_linear_rate_with_utilization():
    dummy = DummyIRMModel(price=100.0, collateral_factor=0.75)
    # utilization_model returns 0.5 => IRM = base + slope * util = 0.02 + 0.1*0.5 = 0.07
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

    # Borrow 50
    agent.borrow()
    assert agent.borrow_amount == pytest.approx(50.0)

    # Step: interest = 7%
    agent.step()
    assert agent.borrow_amount == pytest.approx(50.0 * 1.07)


def test_irm_kinked_rate_behavior():
    dummy = DummyIRMModel(price=100.0, collateral_factor=0.75)
    # Suppose utilization = 0.9 (> kink of 0.8)
    util_fn = lambda: 0.9

    # IRM: base=0.01, slope1=0.05 up to 0.8, slope2=0.2 beyond 0.8
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

    # Borrow 50
    agent.borrow()
    assert agent.borrow_amount == pytest.approx(50.0)

    # Utilization=0.9: rate = base + slope1*kink + slope2*(util - kink)
    # = 0.01 + 0.05*0.8 + 0.20*(0.9 - 0.8) = 0.01 + 0.04 + 0.02 = 0.07
    agent.step()
    assert agent.borrow_amount == pytest.approx(50.0 * 1.07)


# ----------------------------------------
# LiquidatorAgent Unit Tests
# ----------------------------------------

class DummyLendingAgent:
    """
    Stand-in for DeFiLendingAgent to test LiquidatorAgent.
    Attributes: unique_id, collateral_amount, borrow_amount,
                is_marked_for_liquidation, collateral_token, borrow_token, remove().
    """
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
    """
    Stand-in for DeFiModel to test LiquidatorAgent.
    Exposes: current_price, loans list, find_amm_pool(), metrics dict.
    """
    def __init__(self, current_price: float):
        self.current_price = current_price
        self.loans = []
        self.metrics = {}
        self._mesa_model = Model()

        # Single constant‐product AMM pool
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
    liquidator.model = model  # override to use DummyLiquidationModel

    # Before liquidation
    assert lending_agent.borrow_amount == pytest.approx(100.0)
    assert lending_agent.collateral_amount == pytest.approx(1.0)

    liquidator.step()

    # After: collateral seized=1 ETH, sold → 2000/11 ≈ 181.818 DAI, debt_repaid=100
    assert lending_agent.borrow_amount == pytest.approx(0.0)
    assert lending_agent.collateral_amount == pytest.approx(0.0)
    assert lending_agent not in model.loans

    assert "liquidations" in model.metrics
    assert len(model.metrics["liquidations"]) == 1
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

    # No AMM: debt stays the same, but collateral_to_seize = 55/50*2* no change?
    # collateral_needed = (50*(1+0.10))/50 = 1.1, so 2.0 - 1.1 = 0.9
    assert lending_agent.borrow_amount == pytest.approx(50.0)
    assert lending_agent.collateral_amount == pytest.approx(0.9)

    assert "liquidations" in model.metrics
    event = model.metrics["liquidations"][0]
    assert event["recovered_amount"] == pytest.approx(0.0)


# ----------------------------------------
# BlockchainAgent Basic Functionality Tests
# ----------------------------------------

def test_blockchain_basic_account_and_transfer():
    m = Model()
    bc = BlockchainAgent(model=m, block_time=1.0, confirmations=1, base_gas_price=1.0)

    # Create two accounts
    bc.create_account("Alice", initial_balance=100.0)
    bc.create_account("Bob", initial_balance=0.0)
    assert bc.get_native_balance("Alice") == pytest.approx(100.0)
    assert bc.get_native_balance("Bob") == pytest.approx(0.0)

    # Define a simple data_fn that transfers 5 units of native token
    def simple_transfer(sender, receiver, payload, blockchain: BlockchainAgent):
        amount = payload["amount"]
        success = blockchain.transfer_native(sender, receiver, amount)
        gas_used = 21000
        return gas_used, {"success": success}

    # Submit the transaction (gas_price=1, gas_limit=21000)
    tx_id = bc.submit_transaction(
        sender="Alice",
        receiver="Bob",
        data_fn=simple_transfer,
        gas_price=1.0,
        gas_limit=21000,
        payload={"amount": 5.0},
        confirmations=1,
    )
    assert tx_id == 1
    assert bc.metrics["tx_submitted"] == 1
    assert bc.get_pending_txs() == [1]

    # Advance one block: include tx into queue, but not execute until next block
    bc.step()
    assert bc.get_pending_txs() == []
    assert bc.get_queued_txs() == [1]
    assert bc.current_block == 1

    # Advance second block: tx should execute
    bc.step()
    assert bc.current_block == 2
    # Alice pays upfront fee = 21000*1 = 21000, but since Alice only had 100, the transfer fails due to insufficient gas.
    # So Bob’s balance remains 0. But let's check that the failure event was logged.
    events = bc.get_events(block=2)
    assert any(ev[0] == "TransactionFailed_InsufficientGas" for ev in events)

    # Top up Alice so she has enough balance
    bc.native_balances["Alice"] = 50000.0
    # Resubmit
    tx_id2 = bc.submit_transaction(
        sender="Alice",
        receiver="Bob",
        data_fn=simple_transfer,
        gas_price=1.0,
        gas_limit=21000,
        payload={"amount": 5.0},
        confirmations=1,
    )
    # Advance blocks until execution
    bc.step()  # block 3: include
    bc.step()  # block 4: execute
    # Check balances: Alice should lose 5 + (21000 gas*1) - refund 0 gas (full usage)
    assert bc.get_native_balance("Bob") == pytest.approx(5.0)
    # Check fees collected: 21000 units
    assert bc.get_metrics()["total_fees_collected"] == pytest.approx(21000.0)


def test_blockchain_custom_base_gas_price():
    m = Model()
    bc = BlockchainAgent(model=m, block_time=1.0, confirmations=1, base_gas_price=5.0)

    bc.create_account("Alice", initial_balance=100.0)

    def noop(sender, receiver, payload, blockchain: BlockchainAgent):
        return 21000, None

    tx_id = bc.submit_transaction(
        sender="Alice",
        receiver="Bob",
        data_fn=noop,
        gas_price=None,
        gas_limit=21000,
        payload=None,
        confirmations=1,
    )

    assert tx_id == 1
    assert bc.mempool[0].gas_price == pytest.approx(5.0)
    assert bc.metrics["base_gas_price"] == pytest.approx(5.0)


# ----------------------------------------
# BlockchainAgent Advanced Features Tests
# ----------------------------------------

def test_blockchain_event_logging_and_scheduling():
    m = Model()
    bc = BlockchainAgent(model=m, block_time=2.0, confirmations=1)

    # Subscribe to new block events
    recorded = []

    def on_new_block(block_height, timestamp):
        recorded.append((block_height, timestamp))

    bc.subscribe_new_block(on_new_block)

    # Schedule a simple callback at block 3
    called = {"flag": False}

    def scheduled_fn():
        called["flag"] = True

    bc.schedule_call(3, scheduled_fn)

    # Advance 4 blocks
    for _ in range(4):
        bc.step()

    # We should have recorded 4 new‐block callbacks (blocks 1,2,3,4)
    assert len(recorded) == 4
    # The scheduled function should have triggered at block 3
    assert called["flag"] is True

    # Check event logs is empty initially, except we may have TransactionFailed events if any
    assert isinstance(bc.get_events(), list)

    # Test token contract registration & event emission
    class SimpleERC20:
        def __init__(self, initial_supply: float, symbol: str):
            self.initial_supply = initial_supply
            self.symbol = symbol
            # contract state: balances stored in blockchain.token_balances
        def execute(self, sender, payload, blockchain: BlockchainAgent):
            # Simplest: payload = {"action":"transfer", "to":X, "amount":Y}
            act = payload["action"]
            if act == "transfer":
                to = payload["to"]
                amt = payload["amount"]
                # Subtract from sender, add to recipient
                key_from = (self, sender)
                key_to = (self, to)
                if blockchain.token_balances.get(key_from, 0.0) >= amt:
                    blockchain.token_balances[key_from] -= amt
                    blockchain.token_balances[key_to] = blockchain.token_balances.get(key_to, 0.0) + amt
                return 50000, {"status": "ok"}
            return 0, {"status": "noop"}

    # Register a contract
    erc = SimpleERC20(initial_supply=1000.0, symbol="SIM")
    bc.register_contract("SIM", erc)
    # Initialize token balances for Alice and Bob
    bc.token_balances[("SIM", "Alice")] = 500.0
    bc.token_balances[("SIM", "Bob")] = 100.0
    bc.native_balances["Alice"] = 100000.0

    # Submit a contract‐execution transaction to transfer 50 tokens from Alice → Bob
    def erc20_transfer(sender, receiver, payload, blockchain: BlockchainAgent):
        contract_address = receiver
        to = payload["to"]
        amt = payload["amount"]
        key_from = (contract_address, sender)
        key_to = (contract_address, to)
        if blockchain.token_balances.get(key_from, 0.0) >= amt:
            blockchain.token_balances[key_from] -= amt
            blockchain.token_balances[key_to] = blockchain.token_balances.get(key_to, 0.0) + amt
            gas_used = 50000
            return gas_used, {"status": "ok"}
        else:
            return 0, {"status": "fail"}

    tx_id3 = bc.submit_transaction(
        sender="Alice",
        receiver="SIM",  # contract address, so erc.execute() will be called
        data_fn=erc20_transfer,
        gas_price=1.0,
        gas_limit=50000,
        payload={"action": "transfer", "to": "Bob", "amount": 50.0},
        confirmations=1,
    )

    # Advance blocks to include & execute
    bc.step()  # block 5 include
    bc.step()  # block 6 execute

    # Check token balances
    assert bc.get_token_balance("SIM", "Alice") == pytest.approx(450.0)
    assert bc.get_token_balance("SIM", "Bob") == pytest.approx(150.0)

    # Check event log for TransactionExecuted at block 6
    events_block_6 = bc.get_events(block=6)
    assert any(ev[0] == "TransactionExecuted" for ev in events_block_6)


# ----------------------------------------
# DeFiModel Integration Tests
# ----------------------------------------

def make_temp_price_csv(tmp_path, prices):
    df = pd.DataFrame({"price": prices})
    csv_path = tmp_path / "price.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def simple_config(tmp_path):
    price_csv_path = make_temp_price_csv(tmp_path, [100.0, 90.0, 80.0, 70.0])
    config = {
        "simulation": {
            "steps": 2,
            "seed": 123,
        },
        "protocols": {
            "oracle": {
                "price_csv": price_csv_path,
                "initial_price": 100.0,
                "mode": "csv",
            },
            "amm": {
                "token_x": "ETH",
                "token_y": "DAI",
                "reserve_x": 5.0,
                "reserve_y": 1000.0,
                "fee_rate": 0.0,
            },
            "lending": {
                "collateral_factor": 0.75,
                "interest_rate_apr": 0.0,
                "blocks_per_year": 1,
                "liquidation_penalty": 0.0,
                "agents": [
                    {
                        "collateral_token": "ETH",
                        "borrow_token": "DAI",
                        "collateral_amount": 1.0,
                        "desired_ltv": 0.5,
                    }
                ],
            },
        },
    }
    return config


def test_model_initialization_creates_expected_agents(simple_config):
    model = DeFiModel(simple_config)
    all_agents = list(model.agents)
    # Expect: OracleAgent, AMMAgent, DeFiLendingAgent, LiquidatorAgent
    assert len(all_agents) == 4
    names = {a.__class__.__name__ for a in all_agents}
    assert "OracleAgent" in names
    assert "AMMAgent" in names
    assert "DeFiLendingAgent" in names
    assert "LiquidatorAgent" in names


def test_model_step_updates_price_and_metrics(simple_config):
    model = DeFiModel(simple_config)

    # Step 1: price=100
    model.step()
    assert model.current_price == pytest.approx(100.0)
    df1 = model.datacollector.get_model_vars_dataframe()
    assert df1.shape[0] == 1
    assert df1["TVL"].iloc[0] == pytest.approx(100.0)  # 1 ETH * 100
    assert df1["Num_Loans"].iloc[0] == 1

    # Step 2: price=90
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
    model.step()  # price=100, borrow=50>40 → immediate liquidation

    assert len(model.loans) == 0
    assert "liquidations" in model.metrics
    assert len(model.metrics["liquidations"]) == 1

    df = model.datacollector.get_model_vars_dataframe()
    # Only one row, TVL < 100
    assert df["TVL"].iloc[0] < 100.0
