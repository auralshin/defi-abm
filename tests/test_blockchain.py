import sys
from pathlib import Path
import pytest
from mesa import Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import importlib
import defi_abm
importlib.reload(defi_abm)
print('MODULE', defi_abm.__file__)

from defi_abm.agents.blockchain import BlockchainAgent


def test_blockchain_basic_account_and_transfer():
    m = Model()
    bc = BlockchainAgent(model=m, block_time=1.0, confirmations=1, base_gas_price=1.0)

    bc.create_account("Alice", initial_balance=100.0)
    bc.create_account("Bob", initial_balance=0.0)
    assert bc.get_native_balance("Alice") == pytest.approx(100.0)
    assert bc.get_native_balance("Bob") == pytest.approx(0.0)

    def simple_transfer(sender, receiver, payload, blockchain: BlockchainAgent):
        amount = payload["amount"]
        success = blockchain.transfer_native(sender, receiver, amount)
        gas_used = 21000
        return gas_used, {"success": success}

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

    bc.step()
    assert bc.get_pending_txs() == []
    assert bc.get_queued_txs() == [1]
    assert bc.current_block == 1

    bc.step()
    assert bc.current_block == 2
    events = bc.get_events(block=2)
    assert any(ev[0] == "TransactionFailed_InsufficientGas" for ev in events)

    bc.native_balances["Alice"] = 50000.0
    tx_id2 = bc.submit_transaction(
        sender="Alice",
        receiver="Bob",
        data_fn=simple_transfer,
        gas_price=1.0,
        gas_limit=21000,
        payload={"amount": 5.0},
        confirmations=1,
    )
    bc.step()
    bc.step()
    assert bc.get_native_balance("Bob") == pytest.approx(5.0)
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
    print('DEBUG', bc.metrics)
    print('Mempool gas', bc.mempool[0].gas_price)
    assert bc.mempool[0].gas_price > 0

    bc = BlockchainAgent(model=m, block_time=1.0, confirmations=1, base_gas_price=5.0, initial_native_balance=10.0)
    assert bc.get_native_balance(bc) >= 0.0


def test_blockchain_event_logging_and_scheduling():
    m = Model()
    bc = BlockchainAgent(model=m, block_time=2.0, confirmations=1)
    recorded = []

    def on_new_block(block_height, timestamp):
        recorded.append((block_height, timestamp))

    bc.subscribe_new_block(on_new_block)
    called = {"flag": False}

    def scheduled_fn():
        called["flag"] = True

    bc.schedule_call(3, scheduled_fn)

    for _ in range(4):
        bc.step()

    assert len(recorded) == 4
    assert called["flag"] is True
    assert isinstance(bc.get_events(), list)

    class SimpleERC20:
        def __init__(self, initial_supply: float, symbol: str):
            self.initial_supply = initial_supply
            self.symbol = symbol

        def execute(self, sender, payload, blockchain: BlockchainAgent):
            act = payload["action"]
            if act == "transfer":
                to = payload["to"]
                amt = payload["amount"]
                key_from = (self, sender)
                key_to = (self, to)
                if blockchain.token_balances.get(key_from, 0.0) >= amt:
                    blockchain.token_balances[key_from] -= amt
                    blockchain.token_balances[key_to] = blockchain.token_balances.get(key_to, 0.0) + amt
                return 50000, {"status": "ok"}
            return 0, {"status": "noop"}

    erc = SimpleERC20(initial_supply=1000.0, symbol="SIM")
    bc.register_contract("SIM", erc)
    bc.token_balances[("SIM", "Alice")] = 500.0
    bc.token_balances[("SIM", "Bob")] = 100.0
    bc.native_balances["Alice"] = 100000.0

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
        receiver="SIM",
        data_fn=erc20_transfer,
        gas_price=1.0,
        gas_limit=50000,
        payload={"action": "transfer", "to": "Bob", "amount": 50.0},
        confirmations=1,
    )
    bc.step()
    bc.step()
    assert bc.get_token_balance("SIM", "Alice") == pytest.approx(450.0)
    assert bc.get_token_balance("SIM", "Bob") == pytest.approx(150.0)
    events_block_6 = bc.get_events(block=6)
    assert any(ev[0] == "TransactionExecuted" for ev in events_block_6)
