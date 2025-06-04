# examples/run_simple.py

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from defi_abm.models.defi_model import DeFiModel

def main():
    # 1. Locate and load the YAML configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_liquidation.yaml")
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

    # 7. Plot TVL over time for quick visualization
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df["TVL"], label="TVL (in borrow‐token units)")
    plt.xlabel("Time Step")
    plt.ylabel("TVL")
    plt.title("Protocol TVL over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
