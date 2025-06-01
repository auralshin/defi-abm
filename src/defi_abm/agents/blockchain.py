from mesa import Agent
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class Transaction:
    def __init__(
        self,
        tx_id: int,
        sender: Any,
        receiver: Any,
        data_fn: Callable[[Any, Any, Dict[str, Any]], Tuple[int, Any]],
        gas_price: float,
        gas_limit: int,
        payload: Any = None,
        submit_block: int = 0,
        confirmations_required: int = 1,
    ):
        self.tx_id = tx_id
        self.sender = sender
        self.receiver = receiver
        self.data_fn = data_fn
        self.gas_price = float(gas_price)
        self.gas_limit = int(gas_limit)
        self.payload = payload
        self.submit_block = submit_block
        self.confirmations_required = confirmations_required
        self.included_block: Optional[int] = None
        self.executed: bool = False
        self.gas_used: Optional[int] = None
        self.return_value: Any = None


class BlockchainAgent(Agent):
    def __init__(
        self,
        model,
        block_time: float = 1.0,
        confirmations: int = 1,
        base_gas_price: float = 1.0,
        initial_native_balance: float = 0.0,
    ):
        super().__init__(model)
        self.current_block: int = 0
        self.timestamp: float = 0.0
        self.block_time: float = float(block_time)
        self.confirmations_required: int = confirmations
        self._next_tx_id: int = 1
        self.mempool: List[Transaction] = []
        self.tx_queue: List[Transaction] = []
        self.native_balances: Dict[Any, float] = {}
        self.token_balances: Dict[Tuple[Any, Any], float] = {}
        self.contracts: Dict[Any, Any] = {}
        self.event_logs: Dict[int, List[Tuple[str, Any]]] = {}
        self.scheduled: Dict[int, List[Callable[[], None]]] = {}
        self.chain_history: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {
            "tx_submitted": 0,
            "tx_included": 0,
            "tx_executed": 0,
            "total_fees_collected": 0.0,
            "blocks": [],
        }

    def create_account(self, address: Any, initial_balance: Optional[float] = None) -> None:
        self.native_balances[address] = float(initial_balance) if initial_balance else 0.0

    def register_contract(self, contract_address: Any, contract_instance: Any) -> None:
        self.contracts[contract_address] = contract_instance
        if hasattr(contract_instance, "initial_supply") and hasattr(contract_instance, "symbol"):
            self.token_balances[(contract_address, contract_address)] = float(contract_instance.initial_supply)

    def get_native_balance(self, address: Any) -> float:
        return self.native_balances.get(address, 0.0)

    def get_token_balance(self, contract_address: Any, holder: Any) -> float:
        return self.token_balances.get((contract_address, holder), 0.0)

    def transfer_native(self, frm: Any, to: Any, amount: float) -> bool:
        if self.native_balances.get(frm, 0.0) < amount or amount < 0:
            return False
        self.native_balances[frm] -= amount
        self.native_balances[to] = self.native_balances.get(to, 0.0) + amount
        return True

    def transfer_token(self, contract_address: Any, frm: Any, to: Any, amount: float) -> bool:
        key_from = (contract_address, frm)
        key_to = (contract_address, to)
        if self.token_balances.get(key_from, 0.0) < amount or amount < 0:
            return False
        self.token_balances[key_from] -= amount
        self.token_balances[key_to] = self.token_balances.get(key_to, 0.0) + amount
        return True

    def submit_transaction(
        self,
        sender: Any,
        receiver: Any,
        data_fn: Callable[[Any, Any, Dict[str, Any]], Tuple[int, Any]],
        gas_price: Optional[float] = None,
        gas_limit: Optional[int] = None,
        payload: Any = None,
        confirmations: Optional[int] = None,
    ) -> int:
        if gas_price is None:
            gas_price = self.metrics.get("base_gas_price", 1.0)
        if gas_limit is None:
            gas_limit = 21000
        if confirmations is None:
            confirmations = self.confirmations_required

        tx = Transaction(
            tx_id=self._next_tx_id,
            sender=sender,
            receiver=receiver,
            data_fn=data_fn,
            gas_price=gas_price,
            gas_limit=gas_limit,
            payload=payload,
            submit_block=self.current_block,
            confirmations_required=confirmations,
        )
        self._next_tx_id += 1
        self.mempool.append(tx)
        self.metrics["tx_submitted"] += 1
        return tx.tx_id

    def _include_transactions(self) -> None:
        for tx in self.mempool:
            tx.included_block = self.current_block
            self.tx_queue.append(tx)
            self.metrics["tx_included"] += 1
        self.mempool.clear()

    def _execute_transaction(self, tx: Transaction) -> None:
        required_fee = tx.gas_limit * tx.gas_price
        if not self.transfer_native(tx.sender, self, required_fee):
            self._log_event(
                self.current_block,
                "TransactionFailed_InsufficientGas",
                {"tx_id": tx.tx_id, "sender": tx.sender, "receiver": tx.receiver, "required_fee": required_fee},
            )
            return

        gas_used, ret_val = tx.data_fn(tx.sender, tx.receiver, tx.payload, self)
        refund = (tx.gas_limit - gas_used) * tx.gas_price
        if refund > 0:
            self.transfer_native(self, tx.sender, refund)

        self.metrics["total_fees_collected"] += gas_used * tx.gas_price
        tx.executed = True
        tx.gas_used = gas_used
        tx.return_value = ret_val
        self.metrics["tx_executed"] += 1

        self._log_event(
            self.current_block,
            "TransactionExecuted",
            {"tx_id": tx.tx_id, "sender": tx.sender, "receiver": tx.receiver, "gas_used": gas_used, "return_value": ret_val},
        )

    def _log_event(self, block: int, event_name: str, payload: Any) -> None:
        if block not in self.event_logs:
            self.event_logs[block] = []
        self.event_logs[block].append((event_name, payload))

    def get_events(self, block: Optional[int] = None) -> List[Tuple[str, Any]]:
        if block is None:
            all_events = []
            for ev_list in self.event_logs.values():
                all_events.extend(ev_list)
            return all_events
        return self.event_logs.get(block, [])

    def subscribe_new_block(self, callback: Callable[[int, float], None]) -> None:
        if "new_block_listeners" not in self.metrics:
            self.metrics["new_block_listeners"] = []
        self.metrics["new_block_listeners"].append(callback)

    def _notify_new_block(self) -> None:
        for cb in self.metrics.get("new_block_listeners", []):
            try:
                cb(self.current_block, self.timestamp)
            except Exception:
                pass

    def schedule_call(self, block_number: int, fn: Callable[[], None]) -> None:
        if block_number < self.current_block:
            raise ValueError("Cannot schedule call in past block")
        self.scheduled.setdefault(block_number, []).append(fn)

    def _run_scheduled(self) -> None:
        funcs = self.scheduled.pop(self.current_block, [])
        for fn in funcs:
            try:
                fn()
            except Exception:
                pass

    def snapshot_chain(self) -> None:
        snap = {
            "current_block": self.current_block,
            "timestamp": self.timestamp,
            "native_balances": self.native_balances.copy(),
            "token_balances": self.token_balances.copy(),
            "event_logs": {b: list(ev) for b, ev in self.event_logs.items()},
        }
        self.chain_history.append(snap)

    def revert_to_block(self, block_number: int) -> None:
        idx = None
        for i, snap in enumerate(self.chain_history):
            if snap["current_block"] <= block_number:
                idx = i
        if idx is None:
            raise ValueError(f"No snapshot found before or at block {block_number}")

        snap = self.chain_history[idx]
        self.current_block = snap["current_block"]
        self.timestamp = snap["timestamp"]
        self.native_balances = snap["native_balances"].copy()
        self.token_balances = snap["token_balances"].copy()
        self.event_logs = {b: list(ev) for b, ev in snap["event_logs"].items()}
        self.chain_history = self.chain_history[: idx + 1]

    def step(self) -> None:
        self.snapshot_chain()
        self.current_block += 1
        self.timestamp += self.block_time
        self._notify_new_block()
        self._run_scheduled()
        self._include_transactions()

        still_waiting: List[Transaction] = []
        for tx in self.tx_queue:
            if tx.included_block is None:
                tx.included_block = self.current_block
            if self.current_block - tx.included_block >= tx.confirmations_required:
                self._execute_transaction(tx)
            else:
                still_waiting.append(tx)
        self.tx_queue = still_waiting

        self.metrics["blocks"].append({
            "block": self.current_block,
            "timestamp": self.timestamp,
            "tx_count": self.metrics["tx_included"] - sum(
                1 for tx in self.tx_queue if tx.included_block == self.current_block
            ),
        })

    def set_base_gas_price(self, new_price: float) -> None:
        self.metrics["base_gas_price"] = float(new_price)

    def reorder_mempool(self, key_fn: Optional[Callable[[Transaction], Any]] = None) -> None:
        if key_fn is None:
            self.mempool.sort(key=lambda tx: tx.gas_price, reverse=True)
        else:
            self.mempool.sort(key=key_fn)

    def get_current_block(self) -> int:
        return self.current_block

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_pending_txs(self) -> List[int]:
        return [tx.tx_id for tx in self.mempool]

    def get_queued_txs(self) -> List[int]:
        return [tx.tx_id for tx in self.tx_queue]

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def get_account_state(self, address: Any) -> Dict[str, Union[float, Dict]]:
        native = self.get_native_balance(address)
        tokens = {
            c: self.get_token_balance(c, address)
            for c in {c_addr for (c_addr, _) in self.token_balances.keys()}
        }
        return {"native": native, "tokens": tokens}
