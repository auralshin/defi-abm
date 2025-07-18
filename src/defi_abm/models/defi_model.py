from mesa import Model
from mesa.datacollection import DataCollector
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from defi_abm.agents.oracle import OracleAgent
from defi_abm.agents.amm import AMMAgent
from defi_abm.agents.lending import DeFiLendingAgent
from defi_abm.agents.liquidator import LiquidatorAgent

class DeFiModel(Model):
    """
    Central Mesa 3.0+ Model for DeFi Agent-Based Simulation.

    This model coordinates the execution of various DeFi agents like oracles,
    AMMs, lenders, and liquidators. It supports configurable parameters via
    a configuration dictionary passed during initialization.

    Attributes:
        current_price (float): Global price of the asset updated each step.
        loans (list): List of DeFiLendingAgent instances with active loans.
        metrics (dict): Collected metrics including TVLs and number of loans.
        datacollector (DataCollector): Mesa data collector for TVL and loan metrics.
        num_steps (int): Total number of steps to simulate.
    """

    def __init__(self, config: dict):
        """Initialize the model with a configuration dictionary."""
        sim_cfg = config.get("simulation", {})
        seed = sim_cfg.get("seed", None)
        super().__init__(seed=seed)

        self.num_steps = sim_cfg.get("steps", 1_000)

        proto_cfg = config.get("protocols", {})

        # Oracle configuration
        oracle_cfg = proto_cfg.get("oracle", {})
        self.initial_price = float(oracle_cfg.get("initial_price", 1.0))

        # AMM configuration - accept single dict or list of dicts
        amm_cfg_raw = proto_cfg.get("amm", {})
        if isinstance(amm_cfg_raw, list):
            self.amm_cfgs = amm_cfg_raw
        else:
            self.amm_cfgs = [amm_cfg_raw]

        # Lending configuration
        lending_cfg = proto_cfg.get("lending", {})
        self.collateral_factor = float(lending_cfg.get("collateral_factor", 0.75))
        apr = float(lending_cfg.get("interest_rate_apr", 0.05))
        blocks_per_year = int(lending_cfg.get("blocks_per_year", 2_300_000))
        self.interest_rate_per_block = apr / blocks_per_year
        self.liquidation_penalty = float(lending_cfg.get("liquidation_penalty", 0.05))
        self.lending_agents_cfg = lending_cfg.get("agents", [])

        self.current_price = self.initial_price
        self.loans = []
        self.metrics = {
            "tvls": [],
            "num_loans": [],
            "liquidations": [],
        }

        self.datacollector = DataCollector(
            model_reporters={
                "TVL": lambda m: sum(
                    agent.collateral_amount * m.current_price
                    for agent in m.agents
                    if hasattr(agent, "collateral_amount")
                ),
                "Num_Loans": lambda m: len(m.loans),
            }
        )

        self._init_oracle(oracle_cfg)
        self._init_amm(self.amm_cfgs)
        self._init_lending_agents(self.lending_agents_cfg)
        self._init_liquidator(self.liquidation_penalty)

    def _init_oracle(self, oracle_cfg: dict):
        """
        Instantiate OracleAgent from config. Supports CSV or constant pricing.
        """
        price_series = None
        price_csv = oracle_cfg.get("price_csv", None)
        if price_csv:
            import pandas as pd
            df = pd.read_csv(price_csv)
            if "price" not in df.columns:
                raise ValueError("Oracle CSV must have a 'price' column.")
            price_series = df["price"].reset_index(drop=True)

            OracleAgent(
                self,
                price_series=price_series,
                mode=oracle_cfg.get("mode", "static"),
                static_price=float(oracle_cfg.get("initial_price", 1.0)),
                gbm_params=oracle_cfg.get("gbm_params", None),
                interpolate=oracle_cfg.get("interpolate", False),
            )

    def _init_amm(self, amm_cfgs: list):
        """
        Instantiate AMMAgent objects from a configuration list.
        Stores them in ``self.amm_pools``. ``self.amm_pool`` references the first
        pool for backwards compatibility.
        """
        self.amm_pools = []
        for cfg in amm_cfgs:
            amm = AMMAgent(
                self,
                token_x=cfg.get("token_x", "TOKEN_X"),
                token_y=cfg.get("token_y", "TOKEN_Y"),
                reserve_x=float(cfg.get("reserve_x", 1.0)),
                reserve_y=float(cfg.get("reserve_y", 1.0)),
                fee_rate=float(cfg.get("fee_rate", 0.003)),
            )
            self.amm_pools.append(amm)

        if self.amm_pools:
            # Keep the original single-pool attribute for older code/tests
            self.amm_pool = self.amm_pools[0]

    def _init_lending_agents(self, agents_cfg: list):
        """
        Create DeFiLendingAgent instances using config entries.
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
        Instantiate a LiquidatorAgent to monitor and trigger liquidations.
        """
        LiquidatorAgent(self, liquidation_penalty=penalty)

    def get_lending_rate(self, token: str) -> float:
        """
        Get constant per-block lending rate. Token-agnostic in this version.
        """
        return self.interest_rate_per_block

    def register_loan(self, lending_agent: DeFiLendingAgent):
        """
        Register a loan agent if not already listed. Used by borrowers.
        """
        if lending_agent not in self.loans:
            self.loans.append(lending_agent)

    def find_amm_pool(self, token_x: str, token_y: str) -> Optional[AMMAgent]:
        """Return AMMAgent matching the token pair, if any."""
        for pool in getattr(self, "amm_pools", []):
            if (
                pool.token_x == token_x and pool.token_y == token_y
            ) or (
                pool.token_x == token_y and pool.token_y == token_x
            ):
                return pool
        return None

    def step(self):
        """
        Advance the model by one simulation tick:
        - Shuffles and steps all agents
        - Collects TVL and loan metrics
        - Updates internal metric tracking
        """
        if not hasattr(self, "steps"):
            self.steps = 1
        else:
            self.steps += 1

        logger.debug("Model step %s", self.steps)

        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

        tvl = sum(
            agent.collateral_amount * self.current_price
            for agent in self.agents
            if hasattr(agent, "collateral_amount")
        )
        self.metrics["tvls"].append(tvl)
        self.metrics["num_loans"].append(len(self.loans))
        logger.info(
            "Step %s - TVL: %.2f, Loans: %s", self.steps, tvl, len(self.loans)
        )
