from mesa import Agent
from typing import Optional, Callable, List, Tuple
import logging

logger = logging.getLogger(__name__)


class LiquidatorAgent(Agent):
    """
    Agent responsible for monitoring lending agents and initiating liquidations.

    Attributes:
        liquidation_penalty (float): Penalty applied to debt for liquidation incentive.
        max_liquidation_fraction (float): Maximum portion of collateral to seize per liquidation.
        amm_pool_selector (Callable): Function to select AMM pools for liquidation.
        on_liquidation (Callable): Optional callback triggered upon liquidation event.
    """

    def __init__(
        self,
        model,
        liquidation_penalty: float = 0.05,
        max_liquidation_fraction: float = 1.0,
        amm_pool_selector: Optional[Callable] = None,
        on_liquidation: Optional[Callable] = None,
    ):
        super().__init__(model)
        self.liquidation_penalty = float(liquidation_penalty)
        self.max_liquidation_fraction = float(max_liquidation_fraction)
        self.amm_pool_selector = amm_pool_selector or self._default_amm_selector
        self.on_liquidation = on_liquidation

    def _default_amm_selector(self, collateral_token: str, borrow_token: str) -> List:
        """Select default AMM pool from model matching token pair."""
        pool = self.model.find_amm_pool(collateral_token, borrow_token)
        return [pool] if pool else []

    def _liquidate_position(self, lending_agent) -> dict:
        """
        Executes a liquidation for an undercollateralized lending agent.

        Args:
            lending_agent: The agent to be liquidated.

        Returns:
            dict: Event log including amounts repaid, seized, recovered.
        """
        price = self.model.current_price
        debt_value = lending_agent.borrow_amount
        total_cover = debt_value * (1 + self.liquidation_penalty)
        collateral_needed_unbounded = total_cover / price

        max_seizable = lending_agent.collateral_amount * self.max_liquidation_fraction
        collateral_to_seize = min(collateral_needed_unbounded, max_seizable)

        lending_agent.collateral_amount -= collateral_to_seize
        logger.info(
            "Liquidating agent %s for %.4f debt", lending_agent.unique_id, debt_value
        )

        recovered_amount = 0.0
        amm_pools = self.amm_pool_selector(lending_agent.collateral_token, lending_agent.borrow_token)

        remaining_collateral = collateral_to_seize
        for pool in amm_pools:
            if not pool or pool.reserve_x <= 0 or pool.reserve_y <= 0:
                continue
            if pool.token_x == lending_agent.collateral_token:
                amt = pool.swap_x_for_y(remaining_collateral)
            else:
                amt = pool.swap_y_for_x(remaining_collateral)
            recovered_amount += amt
            break

        debt_repaid = min(recovered_amount, debt_value)
        lending_agent.borrow_amount -= debt_repaid
        penalty_paid = debt_value * self.liquidation_penalty

        fully_liquidated = (
            lending_agent.borrow_amount <= 1e-8
            or lending_agent.collateral_amount <= 1e-8
        )

        if fully_liquidated:
            if lending_agent in self.model.loans:
                self.model.loans.remove(lending_agent)
            lending_agent.remove()

        event = {
            "agent_id": lending_agent.unique_id,
            "collateral_seized": collateral_to_seize,
            "recovered_amount": recovered_amount,
            "debt_repaid": debt_repaid,
            "penalty_paid": penalty_paid,
            "remaining_debt": lending_agent.borrow_amount if not fully_liquidated else 0.0,
            "timestamp": self.model.steps,
        }
        logger.info(
            "Liquidation complete for %s; repaid %.4f", lending_agent.unique_id, debt_repaid
        )
        return event

    def step(self):
        """Scan loans for liquidation candidates and trigger liquidation if required."""
        for lending_agent in list(self.model.loans):
            if lending_agent.is_marked_for_liquidation:
                event = self._liquidate_position(lending_agent)
                self.model.metrics.setdefault("liquidations", []).append(event)
                if self.on_liquidation:
                    self.on_liquidation(self, lending_agent, event)
