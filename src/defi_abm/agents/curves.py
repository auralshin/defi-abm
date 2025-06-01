from abc import ABC, abstractmethod
from typing import Tuple


class BaseCurve(ABC):
    """
    Base class for all AMM pricing curves. To customize behavior,
    implement the swap and LP deposit/withdraw methods.
    """

    @abstractmethod
    def compute_swap(
        self,
        amount_in: float,
        reserve_in: float,
        reserve_out: float,
        fee_rate: float,
    ) -> Tuple[float, float]:
        """
        Calculate output and fee based on an input amount and reserves.
        """
        pass

    @abstractmethod
    def compute_deposit_lp(
        self,
        reserve_x: float,
        reserve_y: float,
        total_lp_supply: float,
        amount_x: float,
        amount_y: float,
    ) -> float:
        """
        Determine LP tokens to mint for a given liquidity deposit.
        """
        pass

    @abstractmethod
    def compute_withdraw_lp(
        self,
        reserve_x: float,
        reserve_y: float,
        total_lp_supply: float,
        lp_tokens: float,
    ) -> Tuple[float, float]:
        """
        Determine token amounts to return when LP tokens are burned.
        """
        pass


class ConstantProductCurve(BaseCurve):
    """
    Implements the classic x * y = k AMM logic (e.g. Uniswap v2).
    """

    def compute_swap(
        self,
        amount_in: float,
        reserve_in: float,
        reserve_out: float,
        fee_rate: float,
    ) -> Tuple[float, float]:
        if amount_in <= 0 or reserve_in < 0 or reserve_out <= 0:
            return 0.0, 0.0

        fee_amount = amount_in * fee_rate
        amount_in_with_fee = amount_in - fee_amount

        denominator = reserve_in + amount_in_with_fee
        if denominator <= 0:
            return 0.0, fee_amount

        amount_out = (amount_in_with_fee * reserve_out) / denominator
        return float(amount_out), float(fee_amount)

    def compute_deposit_lp(
        self,
        reserve_x: float,
        reserve_y: float,
        total_lp_supply: float,
        amount_x: float,
        amount_y: float,
    ) -> float:
        if reserve_x <= 0 or reserve_y <= 0 or total_lp_supply <= 0:
            return (amount_x * amount_y) ** 0.5

        expected_y = amount_x * (reserve_y / reserve_x)
        if abs(amount_y - expected_y) > 1e-8:
            raise ValueError(
                f"Deposit must match pool ratio. Expected y={expected_y:.6f}, got y={amount_y:.6f}"
            )
        return (amount_x / reserve_x) * total_lp_supply

    def compute_withdraw_lp(
        self,
        reserve_x: float,
        reserve_y: float,
        total_lp_supply: float,
        lp_tokens: float,
    ) -> Tuple[float, float]:
        if lp_tokens <= 0 or total_lp_supply <= 0:
            return 0.0, 0.0

        share = lp_tokens / total_lp_supply
        amount_x = share * reserve_x
        amount_y = share * reserve_y
        return float(amount_x), float(amount_y)
