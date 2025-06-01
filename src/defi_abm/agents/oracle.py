from mesa import Agent
import pandas as pd
import numpy as np
from typing import Optional, Callable, Union, List
from enum import Enum


class OracleMode(str, Enum):
    """
    Enum for specifying oracle operation modes.
    - CSV: Read prices from a CSV/pandas Series.
    - GBM: Generate synthetic prices using Geometric Brownian Motion.
    - STATIC: Use a fixed constant price.
    """
    CSV = "csv"
    GBM = "gbm"
    STATIC = "static"


class OracleAgent(Agent):
    """
    Oracle agent for publishing price data in an agent-based DeFi simulation.

    Attributes:
        mode (OracleMode): Mode of operation.
        price_series (pd.Series): Price data (only for CSV mode).
        mu (float): Drift rate for GBM mode.
        sigma (float): Volatility for GBM mode.
        dt (float): Time step for GBM mode.
        last_price (float): Last recorded price.
        current_price (float): Latest published price.
        price_history (List[float]): History of published prices.
        on_price_update (Callable): Callback on each price update.
        _rng (np.random.Generator): Random number generator for GBM.
    """

    def __init__(
        self,
        model,
        price_series: Optional[pd.Series] = None,
        mode: Union[str, OracleMode] = OracleMode.CSV,
        gbm_params: Optional[dict] = None,
        static_price: float = 1.0,
        on_price_update: Optional[Callable] = None,
        interpolate: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(model)

        self.mode = OracleMode(mode)
        self.on_price_update = on_price_update
        self.price_history: List[float] = []

        if self.mode == OracleMode.CSV:
            if price_series is None:
                raise ValueError("CSV mode requires price_series.")
            if interpolate:
                price_series = price_series.interpolate(method="linear").fillna(method="bfill")
            self.price_series = price_series.reset_index(drop=True).astype(float)
        else:
            self.price_series = None

        if self.mode == OracleMode.GBM:
            params = gbm_params or {}
            self.mu = float(params.get("mu", 0.0))
            self.sigma = float(params.get("sigma", 0.1))
            self.dt = float(params.get("dt", 1.0))
            self.last_price = float(
                price_series.iloc[0] if price_series is not None else static_price
            )
        else:
            self.mu = self.sigma = self.dt = None
            self.last_price = float(static_price)

        self.current_price = float(self.price_series.iloc[0]) if self.mode == OracleMode.CSV else float(static_price)
        self._step_counter = 0
        self._rng = np.random.default_rng(seed or getattr(self.model, "seed", None))

    def _gbm_next(self) -> float:
        """Generate next price using GBM formula."""
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * self._rng.normal()
        next_price = self.last_price * np.exp(drift + diffusion)
        return max(next_price, 0.0)

    def _csv_step(self) -> float:
        """Return price at current step from CSV series."""
        idx = min(self._step_counter, len(self.price_series) - 1)
        return float(self.price_series.iloc[idx])

    def _gbm_step(self) -> float:
        """Advance GBM model and return next price."""
        if self.model.steps <= 1 and self.price_series is not None:
            self.last_price = float(self.price_series.iloc[0])
        else:
            self.last_price = self._gbm_next()
        return self.last_price

    def _static_step(self) -> float:
        """Return the constant price for STATIC mode."""
        return self.current_price

    def step(self):
        """
        Perform one simulation step.

        Updates the oracle price using the configured mode, updates model's global price,
        and triggers callback if defined.
        """
        dispatch = {
            OracleMode.CSV: self._csv_step,
            OracleMode.GBM: self._gbm_step,
            OracleMode.STATIC: self._static_step,
        }

        self.current_price = dispatch[self.mode]()
        self.model.current_price = self.current_price
        self.price_history.append(self.current_price)

        if self.on_price_update:
            self.on_price_update(self, self.current_price, self.model.steps)
        if self.mode == OracleMode.CSV:
            self._step_counter += 1

