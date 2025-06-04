from mesa import Agent
from typing import Optional, Callable
import logging

from defi_abm.utils.math_helpers import accrue_interest

logger = logging.getLogger(__name__)


class DeFiLendingAgent(Agent):
    """
    A DeFi lending agent capable of borrowing against collateral, accruing interest,
    and monitoring for liquidation conditions. Supports customizable interest rate models (IRMs).

    Attributes:
        collateral_token (str): The token used as collateral.
        borrow_token (str): The token being borrowed.
        collateral_amount (float): Amount of collateral deposited.
        desired_ltv (float): Desired loan-to-value ratio.
        risk_tolerance (float): Optional risk buffer for liquidation thresholds.
        irm_mode (str): Interest rate model type ('fixed', 'linear', or 'kinked').
        irm_params (dict): Parameters for the IRM.
        utilization_model (Callable): Optional dynamic utilization function.
        on_borrow, on_repay, on_withdraw (Callable): Optional event hooks.
    """

    def __init__(
        self,
        model,
        collateral_token: str,
        borrow_token: str,
        collateral_amount: float,
        desired_ltv: float,
        risk_tolerance: float = 0.1,
        irm_mode: str = "fixed",
        irm_params: Optional[dict] = None,
        utilization_model: Optional[Callable[[], float]] = None,
        on_borrow: Optional[Callable] = None,
        on_repay: Optional[Callable] = None,
        on_withdraw: Optional[Callable] = None,
    ):
        super().__init__(model)
        self.collateral_token = collateral_token
        self.borrow_token = borrow_token
        self.collateral_amount = float(collateral_amount)
        self.desired_ltv = float(desired_ltv)
        self.risk_tolerance = float(risk_tolerance)

        self.borrow_amount = 0.0
        self.is_marked_for_liquidation = False

        self.irm_mode = irm_mode
        self.irm_params = irm_params or {"rate": 0.05}
        self.utilization_model = utilization_model

        self.on_borrow = on_borrow
        self.on_repay = on_repay
        self.on_withdraw = on_withdraw

    def get_collateral_value(self) -> float:
        """Returns the USD value of current collateral using model price."""
        return self.collateral_amount * self.model.current_price

    def get_health_ratio(self) -> float:
        """Returns health ratio; values >1 mean safe, <=1 means risk of liquidation."""
        if self.borrow_amount <= 0:
            return float("inf")
        max_allowed = self.get_collateral_value() * self.model.collateral_factor
        return max_allowed / self.borrow_amount

    def _get_utilization(self) -> float:
        """Compute current utilization rate from external or fallback model."""
        if self.utilization_model:
            return self.utilization_model()

        total_borrow = getattr(self.model, "total_borrow", self.borrow_amount)
        total_cash = getattr(self.model, "total_cash", 1e6)
        if total_borrow + total_cash == 0:
            return 0.0
        return total_borrow / (total_borrow + total_cash)

    def _get_rate_from_internal_irm(self) -> float:
        """Determine interest rate based on internal IRM configuration."""
        utilization = self._get_utilization()

        if self.irm_mode == "fixed":
            return self.irm_params.get("rate", 0.05)

        elif self.irm_mode == "linear":
            base = self.irm_params.get("base", 0.02)
            slope = self.irm_params.get("slope", 0.2)
            return base + slope * utilization

        elif self.irm_mode == "kinked":
            base = self.irm_params.get("base", 0.02)
            slope1 = self.irm_params.get("slope1", 0.1)
            slope2 = self.irm_params.get("slope2", 0.5)
            kink = self.irm_params.get("kink", 0.8)
            if utilization < kink:
                return base + slope1 * utilization
            else:
                return base + slope1 * kink + slope2 * (utilization - kink)

        raise ValueError(f"Unsupported IRM mode: {self.irm_mode}")

    def borrow(self, amount: Optional[float] = None) -> float:
        """Borrow up to the allowed LTV based on collateral. Defaults to max."""
        collateral_value = self.get_collateral_value()
        max_borrow_allowed = collateral_value * self.desired_ltv

        if amount is None:
            amount = max_borrow_allowed
        else:
            amount = min(amount, max_borrow_allowed)

        if amount <= 0:
            logger.debug("%s attempted zero borrow", self)
            return 0.0

        self.borrow_amount += amount
        self.model.register_loan(self)

        logger.info(
            "Agent %s borrowed %.4f %s", self.unique_id, amount, self.borrow_token
        )

        if self.on_borrow:
            self.on_borrow(self, amount)
        return amount

    def repay(self, repay_amount: float) -> float:
        """Repay some or all borrowed amount. Returns actual repaid value."""
        if repay_amount <= 0 or self.borrow_amount <= 0:
            logger.debug("%s attempted invalid repay", self)
            return 0.0

        actual_repay = min(repay_amount, self.borrow_amount)
        self.borrow_amount -= actual_repay

        if self.borrow_amount <= 0 and self in self.model.loans:
            self.model.loans.remove(self)

        if self.on_repay:
            self.on_repay(self, actual_repay)
        logger.info(
            "Agent %s repaid %.4f %s", self.unique_id, actual_repay, self.borrow_token
        )
        return actual_repay

    def withdraw_collateral(self, amount: float) -> float:
        """
        Withdraw available collateral not required to back current borrow.
        Ensures health remains above liquidation threshold.
        """
        if amount <= 0 or self.collateral_amount <= 0:
            logger.debug("%s attempted invalid withdrawal", self)
            return 0.0

        if self.borrow_amount > 0:
            max_allowed_collateral_value = (self.borrow_amount / self.model.collateral_factor)
            max_allowed_collateral_amount = max_allowed_collateral_value / self.model.current_price
            max_withdrawable = max(0.0, self.collateral_amount - max_allowed_collateral_amount)
        else:
            max_withdrawable = self.collateral_amount

        actual_withdraw = min(amount, max_withdrawable)
        self.collateral_amount -= actual_withdraw

        if self.on_withdraw:
            self.on_withdraw(self, actual_withdraw)
        logger.info(
            "Agent %s withdrew %.4f %s collateral",
            self.unique_id,
            actual_withdraw,
            self.collateral_token,
        )
        return actual_withdraw

    def _accrue_borrow_interest(self):
        """Apply one step of interest accrual to outstanding borrow."""
        if self.borrow_amount <= 0:
            return
        try:
            rate = self.model.get_lending_rate(self.borrow_token)
        except (AttributeError, NotImplementedError):
            rate = self._get_rate_from_internal_irm()

        self.borrow_amount = accrue_interest(self.borrow_amount, rate, 1)

    def _evaluate_health(self):
        """Update liquidation flag based on collateral vs borrow value."""
        if self.borrow_amount <= 0:
            self.is_marked_for_liquidation = False
            return

        max_allowed = self.get_collateral_value() * self.model.collateral_factor
        self.is_marked_for_liquidation = self.borrow_amount > max_allowed

    def step(self):
        """Run a simulation tick: try borrowing, accrue interest, check health."""
        if self.borrow_amount == 0.0:
            self.borrow()
        self._accrue_borrow_interest()
        self._evaluate_health()
