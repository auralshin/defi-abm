# src/defi_abm/models/defi_model.py

import random
from mesa import Model
from mesa.datacollection import DataCollector

from defi_abm.agents.oracle import OracleAgent
from defi_abm.agents.amm import AMMAgent
from defi_abm.agents.lending import DeFiLendingAgent
from defi_abm.agents.liquidator import LiquidatorAgent

class DeFiModel(Model):
    """
    Central Mesa 3.0+ Model for DeFi Agent‐Based Simulation.

    - Calls super().__init__(seed=…) so that Mesa initializes properly.
    - Agents auto‐register in self.agents at construction.
    - Use self.agents.do("step") or self.agents.shuffle_do("step") to activate.
    - Unique IDs are assigned automatically by Mesa.
    """

    def __init__(self, config: dict):
        # Pass seed (if any) into super().__init__
        sim_cfg = config.get("simulation", {})
        seed = sim_cfg.get("seed", None)
        super().__init__(seed=seed)

        # Number of steps we plan to run (not strictly necessary since Mesa has step_count)
        self.num_steps = sim_cfg.get("steps", 1_000)

        # --- Protocol Parameters ---
        proto_cfg = config.get("protocols", {})

        # Oracle
        oracle_cfg = proto_cfg.get("oracle", {})
        self.initial_price = float(oracle_cfg.get("initial_price", 1.0))

        # AMM
        amm_cfg = proto_cfg.get("amm", {})
        self.amm_token_x = amm_cfg.get("token_x", "TOKEN_X")
        self.amm_token_y = amm_cfg.get("token_y", "TOKEN_Y")
        self.amm_reserve_x = float(amm_cfg.get("reserve_x", 1.0))
        self.amm_reserve_y = float(amm_cfg.get("reserve_y", 1.0))
        self.amm_fee_rate = float(amm_cfg.get("fee_rate", 0.003))

        # Lending
        lending_cfg = proto_cfg.get("lending", {})
        self.collateral_factor = float(lending_cfg.get("collateral_factor", 0.75))
        apr = float(lending_cfg.get("interest_rate_apr", 0.05))
        blocks_per_year = int(lending_cfg.get("blocks_per_year", 2_300_000))
        self.interest_rate_per_block = apr / blocks_per_year
        self.liquidation_penalty = float(lending_cfg.get("liquidation_penalty", 0.05))
        self.lending_agents_cfg = lending_cfg.get("agents", [])

        # --- Global State ---
        # Will be updated by OracleAgent.step()
        self.current_price = self.initial_price

        # Track active DeFiLendingAgent instances with outstanding debt
        self.loans = []

        # Metrics collector
        self.metrics = {
            "tvls": [],
            "num_loans": [],
            "liquidations": [],
        }

        # --- DataCollector ---
        self.datacollector = DataCollector(
            model_reporters={
                # Sum collateral_amount * current_price over all lending agents
                "TVL": lambda m: sum(
                    agent.collateral_amount * m.current_price
                    for agent in m.agents
                    if hasattr(agent, "collateral_amount")
                ),
                "Num_Loans": lambda m: len(m.loans),
            }
        )

        # --- Agent Initialization (auto‐registered in self.agents) ---
        # 1. OracleAgent
        self._init_oracle(oracle_cfg)

        # 2. AMMAgent
        self._init_amm(amm_cfg)

        # 3. DeFiLendingAgent population
        self._init_lending_agents(self.lending_agents_cfg)

        # 4. LiquidatorAgent
        self._init_liquidator(self.liquidation_penalty)

    def _init_oracle(self, oracle_cfg: dict):
        """
        Instantiate OracleAgent (it auto‐registers in self.agents).
        If price_csv is provided, load a Pandas series; otherwise constant.
        """
        price_series = None
        price_csv = oracle_cfg.get("price_csv", None)
        if price_csv:
            import pandas as pd

            df = pd.read_csv(price_csv)
            if "price" not in df.columns:
                raise ValueError("Oracle CSV must have a 'price' column.")
            price_series = df["price"].reset_index(drop=True)

        OracleAgent(self, price_series=price_series)

    def _init_amm(self, amm_cfg: dict):
        """
        Instantiate a single AMMAgent; store a reference in self.amm_pool.
        """
        amm = AMMAgent(
            self,
            token_x=self.amm_token_x,
            token_y=self.amm_token_y,
            reserve_x=self.amm_reserve_x,
            reserve_y=self.amm_reserve_y,
            fee_rate=self.amm_fee_rate,
        )
        self.amm_pool = amm

    def _init_lending_agents(self, agents_cfg: list):
        """
        Instantiate DeFiLendingAgent instances (auto‐registered).
        """
        for cfg in agents_cfg:
            DeFiLendingAgent(
                self,
                collateral_token=cfg.get("collateral_token"),
                borrow_token=cfg.get("borrow_token"),
                collateral_amount=float(cfg.get("collateral_amount", 0.0)),
                desired_ltv=float(cfg.get("desired_ltv", 0.0)),
            )

    def _init_liquidator(self, penalty: float):
        """
        Instantiate a single LiquidatorAgent (auto‐registered).
        """
        LiquidatorAgent(self, liquidation_penalty=penalty)

    def get_lending_rate(self, token: str) -> float:
        """
        Return the per‐block interest rate (constant across tokens for v1.0).
        """
        return self.interest_rate_per_block

    def register_loan(self, lending_agent: DeFiLendingAgent):
        """
        Add lending_agent to self.loans if not already present.
        (LiquidatorAgent will remove it later via lending_agent.remove().)
        """
        if lending_agent not in self.loans:
            self.loans.append(lending_agent)

    def find_amm_pool(self, token_x: str, token_y: str):
        """
        Return the single AMMAgent if token_x/token_y match (either order), else None.
        """
        if (
            self.amm_pool.token_x == token_x and self.amm_pool.token_y == token_y
        ) or (
            self.amm_pool.token_x == token_y and self.amm_pool.token_y == token_x
        ):
            return self.amm_pool
        return None

    def step(self):
        """
        Advance the model one tick:
          1. Mesa auto‐increments self.step_count.
          2. Activate all agents’ step() in a random order (shuffle_do).
          3. Collect data via DataCollector.
          4. Record TVL and number of loans in metrics dict.
        """
        # 1. Shuffle and execute each agent’s step
        self.agents.shuffle_do("step")

        # 2. Collect model‐level metrics
        self.datacollector.collect(self)

        # 3. Record into metrics dictionary
        tvl = sum(
            agent.collateral_amount * self.current_price
            for agent in self.agents
            if hasattr(agent, "collateral_amount")
        )
        self.metrics["tvls"].append(tvl)
        self.metrics["num_loans"].append(len(self.loans))