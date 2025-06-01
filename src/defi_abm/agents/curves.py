from abc import ABC, abstractmethod
from typing import Tuple


class BaseCurve(ABC):
    """
    Abstract base class for AMM pricing curves.

    To implement a custom curve, subclass this and implement the three required methods:
    - compute_swap
    - compute_deposit_lp
    - compute_withdraw_lp
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

        Args:
            amount_in (float): Amount of input token provided.
            reserve_in (float): Current reserve of the input token.
            reserve_out (float): Current reserve of the output token.
            fee_rate (float): Swap fee rate (e.g., 0.003).

        Returns:
            Tuple[float, float]: Output amount of the other token, fee charged.
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

        Args:
            reserve_x (float): Reserve of token X.
            reserve_y (float): Reserve of token Y.
            total_lp_supply (float): Total LP token supply.
            amount_x (float): Amount of token X to deposit.
            amount_y (float): Amount of token Y to deposit.

        Returns:
            float: Amount of LP tokens to mint.
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

        Args:
            reserve_x (float): Reserve of token X.
            reserve_y (float): Reserve of token Y.
            total_lp_supply (float): Total LP token supply.
            lp_tokens (float): Amount of LP tokens to burn.

        Returns:
            Tuple[float, float]: Amounts of token X and Y to return.
        """
        pass


class ConstantProductCurve(BaseCurve):
    """
    Implements the constant product curve x * y = k used in many AMMs (e.g., Uniswap v2).
    """

    def compute_swap(
        self,
        amount_in: float,
        reserve_in: float,
        reserve_out: float,
        fee_rate: float,
    ) -> Tuple[float, float]:
        """
        Compute output amount and fee from a swap given reserves and fee rate.

        Args:
            amount_in (float): Input token amount.
            reserve_in (float): Current reserve of input token.
            reserve_out (float): Current reserve of output token.
            fee_rate (float): Swap fee rate.

        Returns:
            Tuple[float, float]: Output token amount, fee collected.
        """
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
        """
        Calculate LP tokens to mint when liquidity is added.

        Args:
            reserve_x (float): Current reserve of token X.
            reserve_y (float): Current reserve of token Y.
            total_lp_supply (float): Current total LP supply.
            amount_x (float): Amount of token X being deposited.
            amount_y (float): Amount of token Y being deposited.

        Returns:
            float: LP tokens to mint.
        """
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
        """
        Calculate token amounts to return when LP tokens are burned.

        Args:
            reserve_x (float): Reserve of token X.
            reserve_y (float): Reserve of token Y.
            total_lp_supply (float): Total LP supply.
            lp_tokens (float): LP tokens to burn.

        Returns:
            Tuple[float, float]: Token X and token Y amounts returned.
        """
        if lp_tokens <= 0 or total_lp_supply <= 0:
            return 0.0, 0.0

        share = lp_tokens / total_lp_supply
        amount_x = share * reserve_x
        amount_y = share * reserve_y
        return float(amount_x), float(amount_y)
