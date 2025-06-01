# DeFi-ABM

A modular, agent-based simulation framework for DeFi protocols built on top of Mesa.\
Includes prebuilt agents for:

- **Oracles** (CSV, GBM, static)
- **AMM** (configurable curve)
- **Lending/Borrowing** (with built-in or custom IRM)
- **Liquidators**
- **Blockchain** (block & timestamp, mempool, gas, scheduling, contracts)

______________________________________________________________________

## Table of Contents

1. [Features](#features)
1. [Installation](#installation)
1. [Project Structure](#project-structure)
1. [Quickstart Example](#quickstart-example)
1. [Agent & Model APIs](#agent--model-apis)
   - [DeFiModel Configuration](#defimodel-configuration)
   - [AMMAgent](#ammagent)
   - [OracleAgent](#oracleagent)
   - [DeFiLendingAgent](#defilendingagent)
   - [LiquidatorAgent](#liquidatoragent)
   - [BlockchainAgent](#blockchainagent)
1. [Running Tests](#running-tests)
1. [Contributing](#contributing)
1. [License](#license)

______________________________________________________________________

## Features

- **Mesa 3.0+ Compatibility**\
  Agents and models follow the latest Mesa patterns (auto‐assigned `unique_id`, `model.agents`, `model.steps`, etc.).

- **Pluggable IRM (Interest Rate Model)**\
  Lending agents can either call `model.get_lending_rate(token)` or use built-in IRM modes (`fixed`, `linear`, `kinked`) with a modular utilization function.

- **Configurable AMM Curves**\
  Supply your own `BaseCurve` subclass (e.g. constant‐product, weighted) to define how swap prices are calculated.

- **Oracle Flexibility**\
  Choose between static prices, CSV‐driven series, or synthetic Geometric Brownian Motion (GBM).

- **Automated Liquidation**\
  Liquidator agents monitor lending health and seize collateral via an AMM pool when undercollateralized.

- **Blockchain Simulation**

  - Advances block height & timestamp
  - Mempool, transaction queue, configurable confirmations
  - Native‐token and ERC‐20‐style balances
  - Gas & fee accounting, refund logic
  - Event logs, scheduling, and basic reorg support
  - Contract registry/execution interface

- **Data Collection**\
  Built‐in `DataCollector` for model‐level metrics (TVL, Num_Loans, etc.).

______________________________________________________________________

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/defi-abm.git
   cd defi-abm
   ```

1. **Create a Virtual Environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

1. **(Optional) Install Locally**

   ```bash
   pip install -e .
   ```

______________________________________________________________________

## Project Structure

```markdown
defi-abm/
├── src/
│   └── defi_abm/
│       ├── agents/
│       │   ├── amm.py
│       │   ├── curves.py
│       │   ├── oracle.py
│       │   ├── lending.py
│       │   ├── liquidator.py
│       │   └── blockchain.py
│       ├── models/
│       │   └── defi_model.py
│       ├── utils/
│       │   └── math_helpers.py
│       └── __init__.py
├── tests/
│   └── test_all.py
├── README.md
├── setup.py
├── requirements.txt
└── LICENSE
```

- **`agents/`** – all agent classes (AMM, Oracle, Lending, Liquidator, Blockchain).
- **`curves.py`** – base‐class and example curve (e.g. `ConstantProductCurve`).
- **`models/defi_model.py`** – a turnkey model that wires up Oracle, AMM, Lending, Liquidator.
- **`utils/math_helpers.py`** – helper functions (_e.g._ `get_amount_out`, `accrue_interest`).
- **`tests/test_all.py`** – comprehensive unit & integration tests for every component.

______________________________________________________________________

## Quickstart Example

Below is a minimal snippet showing how to configure and run `DeFiModel`:

```python
from defi_abm.models.defi_model import DeFiModel

# 1. Create a price CSV beforehand, e.g. using pandas:
#    df = pd.DataFrame({"price": [100.0, 95.0, 90.0, 85.0]})
#    df.to_csv("price.csv", index=False)

config = {
    "simulation": {
        "steps": 5,
        "seed": 42,
    },
    "protocols": {
        "oracle": {
            "price_csv": "price.csv",
            "initial_price": 100.0,
        },
        "amm": {
            "token_x": "ETH",
            "token_y": "DAI",
            "reserve_x": 10.0,
            "reserve_y": 2000.0,
            "fee_rate": 0.003,
        },
        "lending": {
            "collateral_factor": 0.75,
            "interest_rate_apr": 0.05,
            "blocks_per_year": 2102400,
            "liquidation_penalty": 0.10,
            "agents": [
                {
                    "collateral_token": "ETH",
                    "borrow_token": "DAI",
                    "collateral_amount": 1.0,
                    "desired_ltv": 0.5,
                },
            ],
        },
    },
}

# 2. Instantiate the model
model = DeFiModel(config)

# 3. Step through the simulation
for _ in range(config["simulation"]["steps"]):
    model.step()

# 4. Inspect results
df = model.datacollector.get_model_vars_dataframe()
print(df)
print("Liquidations:", model.metrics.get("liquidations", []))
```

______________________________________________________________________

## Agent & Model APIs

### DeFiModel Configuration

- **`simulation.steps`** (`int`)\
  Number of blocks to run.

- **`simulation.seed`** (`int`, optional)\
  Random seed for reproducibility.

- **`protocols.oracle`**

  - `price_csv` (`str`) — path to CSV with a `price` column (0-based).
  - `initial_price` (`float`) — fallback price if CSV unavailable.

- **`protocols.amm`**

  - `token_x` (`str`), `token_y` (`str`)
  - `reserve_x`, `reserve_y` (`float`)
  - `fee_rate` (`float`) — e.g. `0.003` for 0.3%.

- **`protocols.lending`**

  - `collateral_factor` (`float`) — e.g. `0.75`.
  - `interest_rate_apr` (`float`) — annual APR.
  - `blocks_per_year` (`int`) — to convert APR → per-block rate.
  - `liquidation_penalty` (`float`) — extra fee on liquidation.
  - `agents` (list of dicts)\
    Each dict:

    ```yaml
    - collateral_token: "ETH"
      borrow_token: "DAI"
      collateral_amount: 1.0
      desired_ltv: 0.5
      # (Optional) risk_tolerance, irm_mode, irm_params, utilization_model
    ```

**DeFiModel** will automatically:

1. Instantiate an **`OracleAgent`** (mode=CSV by default).
1. Instantiate an **`AMMAgent`** with given reserves and fee.
1. Instantiate one **`DeFiLendingAgent`** per entry under `lending.agents`.
1. Instantiate a **`LiquidatorAgent`** with the specified `liquidation_penalty`.
1. Collect model‐level metrics (TVL, Num_Loans) via Mesa’s `DataCollector`.

______________________________________________________________________

### AMMAgent

```python
class AMMAgent(Agent):
    def __init__(
        self,
        model,
        token_x: str,
        token_y: str,
        reserve_x: float,
        reserve_y: float,
        fee_rate: float = 0.003,
        curve: BaseCurve = ConstantProductCurve(),
    ):
        ...
```

- **`get_price()`**\
  Returns `reserve_y / reserve_x`, or `+inf` if `reserve_x ≤ 0`.

- **`swap_x_for_y(amount_x_in)`**\
  Uses `curve.get_amount_out(amount_in, reserve_x, reserve_y, fee_rate)`, updates reserves, returns `amount_y_out`.

- **`swap_y_for_x(amount_y_in)`**\
  Similar to the above, but flips `(x ↔ y)`.

- `step()` is a no-op (all swaps are initiated by others).

**To use a custom curve**, subclass `BaseCurve` (in `curves.py`) and pass `curve=YourCurve()`.

______________________________________________________________________

### OracleAgent

```python
class OracleAgent(Agent):
    def __init__(
        self,
        model,
        price_series: Optional[pd.Series] = None,
        mode: str = "csv",          # one of {"csv", "gbm", "static"}
        gbm_params: Optional[dict] = None,
        static_price: float = 1.0,
        on_price_update: Optional[Callable[[Agent, float, int], None]] = None,
    ):
        ...
```

- **Modes**:

  - `csv` (requires `price_series`): reads from `price_series.iloc[model.steps - 1]`, holds last if out of range.
  - `gbm` (UUID path): uses GBM formula with `mu`, `sigma`, `dt`. First step uses `price_series[0]` or `static_price`.
  - `static`: always returns `static_price`.

- **`step()`** publishes `self.model.current_price = self.current_price` each block.

- **`on_price_update(self, new_price, step)`** hook is called whenever price changes.

______________________________________________________________________

### DeFiLendingAgent

```python
class DeFiLendingAgent(Agent):
    def __init__(
        self,
        model,
        collateral_token: str,
        borrow_token: str,
        collateral_amount: float,
        desired_ltv: float,
        risk_tolerance: float = 0.1,
        irm_mode: str = "fixed",            # one of {"fixed", "linear", "kinked"}
        irm_params: Optional[dict] = None,   # e.g. {"rate": 0.05} or {"base":0.02,"slope":0.1}
        utilization_model: Optional[Callable[[], float]] = None,
        on_borrow: Optional[Callable] = None,
        on_repay: Optional[Callable] = None,
        on_withdraw: Optional[Callable] = None,
    ):
        ...
```

- **`borrow(amount=None)`**\
  Borrows up to `desired_ltv * collateral_value` (if `amount=None`), registers in `model.loans`.

- **`repay(repay_amount)`**\
  Repays toward outstanding `borrow_amount`. Removes from `model.loans` if fully repaid.

- **`withdraw_collateral(amount)`**\
  Withdraws up to `amount` without violating `LTV ≤ collateral_factor`. Returns actual withdrawn.

- **`_get_rate_from_internal_irm()`**

  - `fixed`: returns `irm_params["rate"]`.
  - `linear`: returns `base + slope * utilization`.
  - `kinked`: uses `base`, `slope1`, `slope2`, and `kink` from `irm_params`.

- **`_get_utilization()`**\
  Calls `utilization_model()` if provided; otherwise uses `model.total_borrowed / (model.total_borrowed + model.total_cash)` as fallback.

- **`step()`**

  1. If `borrow_amount == 0`, calls `borrow()`.
  1. Accrues interest:
     - If `model.get_lending_rate()` exists and does not raise, uses that.
     - Otherwise, falls back to `_get_rate_from_internal_irm()`.
  1. Re‐evaluate health (`borrow_amount > collateral_value * collateral_factor` → mark for liquidation).

______________________________________________________________________

### LiquidatorAgent

```python
class LiquidatorAgent(Agent):
    def __init__(self, model, liquidation_penalty: float = 0.05):
        ...
```

- **`step()`**\
  Iterates through `model.loans`; if any lending agent has `is_marked_for_liquidation == True`, calls `_liquidate_position(agent)`.

- **`_liquidate_position(lending_agent)`**

  1. Calculates `total_cover = debt * (1 + penalty)`.
  1. Determines `collateral_needed = total_cover / price`.
  1. Caps to `max_liquidation_fraction = 1.0` (configurable if extended).
  1. Seizes `collateral_to_seize`.
  1. If an AMM pool exists (`model.find_amm_pool(...)`), sells collateral via `swap_x_for_y` or `swap_y_for_x`.
  1. Deducts `debt_repaid = min(recovered_amount, debt)`.
  1. Removes agent from `model.loans` and calls `agent.remove()` if fully liquidated, otherwise partial.
  1. Records a dictionary into `model.metrics["liquidations"]` containing fields:

     ```json
     {
       "agent_id": ...,
       "collateral_seized": ...,
       "recovered_amount": ...,
       "debt_repaid": ...,
       "penalty_paid": ...,
       "remaining_debt": ...,
       "timestamp": model.steps
     }
     ```

______________________________________________________________________

### BlockchainAgent

```python
class BlockchainAgent(Agent):
    def __init__(
        self,
        model,
        block_time: float = 1.0,
        confirmations: int = 1,
        base_gas_price: float = 1.0,
        initial_native_balance: float = 0.0,
    ):
        ...
```

1. **Block & Time**

   - `current_block` (int)
   - `timestamp` (float)
   - `block_time` (seconds/abstract units per block)

1. **Accounts & Balances**

   - `native_balances[address]: float`
   - `token_balances[(contract_address, holder)]: float`

1. **Contracts**

   - `register_contract(contract_address, contract_instance)`
     - `contract_instance` must implement `execute(sender, payload, blockchain) → (gas_used, return_value)`

1. **Mempool & Queue**

   - `submit_transaction(sender, receiver, data_fn, gas_price, gas_limit, payload, confirmations) → tx_id`
     - `data_fn(sender, receiver, payload, blockchain) → (gas_used, return_value)`
   - Internally stores `Transaction` objects with fields:

    ```bash
     tx_id, sender, receiver, data_fn, gas_price, gas_limit,
     payload, submit_block, confirmations_required, included_block, executed, gas_used, return_value
    ```

   - `get_pending_txs()` and `get_queued_txs()` return lists of `tx_id`.

1. **Execution & Fees**

   - Each `step()` does:

     1. `snapshot_chain()` for reorg support.
     1. `current_block += 1; timestamp += block_time`.
     1. Notify subscribers via `subscribe_new_block`.
     1. Run scheduled callbacks in `scheduled[current_block]`.
     1. Move `mempool → tx_queue` (mark `included_block = current_block`).
     1. For each `tx` in `tx_queue`:
        - If `current_block - included_block ≥ confirmations_required`, call `_execute_transaction(tx)`, else keep in queue.

   - **`_execute_transaction(tx)`**:

     1. Charges sender `upfront_fee = tx.gas_limit * tx.gas_price`; if insufficient balance, emits `"TransactionFailed_InsufficientGas"` event.
     1. Calls `gas_used, return_value = tx.data_fn(sender, receiver, payload, self)`.
     1. Refunds `(gas_limit - gas_used) * gas_price`.
     1. Adds `gas_used * gas_price` to `metrics["total_fees_collected"]`.
     1. Marks `tx.executed = True`, appends event `"TransactionExecuted"` with payload.

1. **Event Logging & Pub/Sub**

   - `_log_event(block, event_name, payload)`
   - `get_events(block=None)` returns all logs or those at a specific block.
   - `subscribe_new_block(callback)` to be called each new block.

1. **Scheduling**

   - `schedule_call(block_number, fn)`
   - `_run_scheduled()` in `step()` invokes them.

1. **Reorg Support**

   - `snapshot_chain()` stores snapshots of balances and logs.
   - `revert_to_block(block_number)` rewinds to last snapshot ≤ that block.

1. **Helpers**

   - `transfer_native(frm, to, amount)`
   - `transfer_token(contract_address, frm, to, amount)`
   - `get_native_balance(address)` & `get_token_balance(contract_address, holder)`
   - `set_base_gas_price(new_price)` & `reorder_mempool(key_fn)`

______________________________________________________________________

## Running Tests

All unit and integration tests are in `tests/test_all.py`. To run:

```bash
pytest -q
```

Ensure all tests pass before committing or releasing.

______________________________________________________________________

## Contributing

1. Fork the repository and create a feature branch.
1. Run tests locally (`pytest`).
1. Submit a pull request with clear descriptions of changes.

Please follow standard Python style (PEP 8) and include new tests for any new functionality.

______________________________________________________________________

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
