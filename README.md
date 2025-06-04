# DeFi-ABM

DeFi-ABM is a modular agent-based simulation framework for decentralized finance built as a plugin for the [Mesa](https://mesa.readthedocs.io/) toolkit.

## Features

- Automated market maker, oracle, lending and liquidator agents
- Simple blockchain simulation with accounts and event logging
- Pluggable interest rate and pricing models
- Built-in data collection utilities

## Installation

```bash
pip install -e .
```

## Quickstart

```python
from defi_abm.models.defi_model import DeFiModel

config = {
    "simulation": {"steps": 2},
    "protocols": {
        "oracle": {"mode": "static", "initial_price": 100.0},
        "amm": {"token_x": "ETH", "token_y": "DAI", "reserve_x": 5.0, "reserve_y": 1000.0},
        "lending": {"collateral_factor": 0.75, "agents": [{"collateral_token": "ETH", "borrow_token": "DAI", "collateral_amount": 1.0, "desired_ltv": 0.5}]},
    },
}

model = DeFiModel(config)
for _ in range(config["simulation"]["steps"]):
    model.step()
```

## Documentation

See the Sphinx documentation in `docs/` for full API details.

## Running Tests

```bash
pytest -q
```

## Examples

Sample configuration files and scripts can be found in the `examples/` directory.
Run the basic demonstration with:

```bash
python examples/run_simple.py
```

## License

Released under the MIT License.
