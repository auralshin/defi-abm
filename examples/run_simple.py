# examples/run_simple.py

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from defi_abm.agents.oracle import OracleAgent

from defi_abm.models.defi_model import DeFiModel

def main():
    # 1. Locate and load the YAML configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Use a config that triggers a sharp price drop early on so we
    # can easily demonstrate liquidations and changing TVL
    config_path = os.path.join(script_dir, "config_showcase.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve relative CSV path for the oracle price series
    price_csv = config.get("protocols", {}).get("oracle", {}).get("price_csv")
    if price_csv and not os.path.isabs(price_csv):
        config["protocols"]["oracle"]["price_csv"] = os.path.join(script_dir, price_csv)

    # 2. Instantiate the DeFiModel
    model = DeFiModel(config)

    # 3. Run the simulation for the specified number of steps
    for _ in range(config["simulation"]["steps"]):
        model.step()

    # 4. Retrieve a DataFrame of collected model‐level metrics (TVL, Num_Loans)
    df = model.datacollector.get_model_vars_dataframe()

    # 5. Print the last few rows of the DataFrame
    print("\n=== Final model metrics (last 5 steps) ===")
    print(df.tail())

    # 6. Print any liquidation events that occurred
    print("\n=== Liquidation events recorded ===")
    for event in model.metrics["liquidations"]:
        print(event)

    # 7. Plot TVL and price for quick visualization
    oracle = next(a for a in model.agents if isinstance(a, OracleAgent))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(df.index, df["TVL"], label="TVL", color="tab:blue")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("TVL", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(df.index, oracle.price_history[: len(df)], label="Price", color="tab:orange", linestyle="--")
    ax2.set_ylabel("Price", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle("Protocol TVL and Price Over Time")
    fig.tight_layout()
    fig.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
