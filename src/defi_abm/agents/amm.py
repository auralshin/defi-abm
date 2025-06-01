from mesa import Agent
from typing import Dict, Optional, Callable, Tuple
from defi_abm.agents.curves import BaseCurve, ConstantProductCurve

class AMMAgent(Agent):
    """
    Automated Market Maker (AMM) agent for simulating token swaps and LP actions.

    Attributes:
        token_x (str): Symbol of the first token in the pair.
        token_y (str): Symbol of the second token in the pair.
        reserve_x (float): Reserve of token_x in the pool.
        reserve_y (float): Reserve of token_y in the pool.
        fee_rate (float): Swap fee rate.
        protocol_fee_rate (float): Fraction of fee sent to protocol reserves.
        curve (BaseCurve): Pricing curve used for swap logic.
        total_lp_supply (float): Total liquidity tokens issued.
        lp_balances (Dict[int, float]): Mapping of agent ID to LP token balance.
        protocol_reserve_y (float): Accumulated protocol fees in token_y.
        on_swap (Callable): Optional hook for swap events.
        on_deposit (Callable): Optional hook for deposit events.
        on_withdraw (Callable): Optional hook for withdraw events.
    """

    def __init__(
        self,
        model,
        token_x: str,
        token_y: str,
        reserve_x: float,
        reserve_y: float,
        fee_rate: float = 0.003,
        protocol_fee_rate: float = 0.0,
        curve: Optional[BaseCurve] = None,
        on_swap: Optional[Callable] = None,
        on_deposit: Optional[Callable] = None,
        on_withdraw: Optional[Callable] = None,
    ):
        """Initialize an AMM agent."""
        super().__init__(model)

        self.token_x = token_x
        self.token_y = token_y
        self.reserve_x = float(reserve_x)
        self.reserve_y = float(reserve_y)
        self.fee_rate = float(fee_rate)
        self.protocol_fee_rate = float(protocol_fee_rate)
        self.curve = curve if curve is not None else ConstantProductCurve()

        self.total_lp_supply = 0.0
        self.lp_balances: Dict[int, float] = {}
        self.protocol_reserve_y = 0.0

        self.on_swap = on_swap
        self.on_deposit = on_deposit
        self.on_withdraw = on_withdraw

    def get_reserves(self) -> Tuple[float, float]:
        """Return current reserves of token_x and token_y."""
        return self.reserve_x, self.reserve_y

    def get_k(self) -> float:
        """Return the invariant constant-product (k = x * y)."""
        return self.reserve_x * self.reserve_y

    def _charge_fee(self, amount_in: float) -> Tuple[float, float, float]:
        """
        Calculate fee breakdown.

        Args:
            amount_in (float): The input amount to be charged a fee.

        Returns:
            Tuple[float, float, float]: Amount after fee, LP fee, and protocol fee.
        """
        total_fee = amount_in * self.fee_rate
        protocol_fee = total_fee * self.protocol_fee_rate
        lp_fee = total_fee - protocol_fee
        return amount_in - total_fee, lp_fee, protocol_fee

    def swap_x_for_y(self, amount_x_in: float) -> float:
        """
        Execute a swap from token X to token Y.

        Args:
            amount_x_in (float): Amount of token X to swap.

        Returns:
            float: Amount of token Y received.
        """
        if amount_x_in <= 0:
            return 0.0

        amt_after_fee, lp_fee_x, protocol_fee_x = self._charge_fee(amount_x_in)

        amount_y_out, _ = self.curve.compute_swap(
            amount_in=amt_after_fee,
            reserve_in=self.reserve_x,
            reserve_out=self.reserve_y,
            fee_rate=0.0,
        )

        self.reserve_x += amount_x_in
        self.reserve_y -= amount_y_out

        if self.reserve_x > 0:
            price = self.reserve_y / self.reserve_x
            self.reserve_y += lp_fee_x * price
            self.protocol_reserve_y += protocol_fee_x * price

        if self.on_swap:
            self.on_swap(self, amount_x_in, amount_y_out, "X→Y")

        return float(amount_y_out)

    def swap_y_for_x(self, amount_y_in: float) -> float:
        """
        Execute a swap from token Y to token X.

        Args:
            amount_y_in (float): Amount of token Y to swap.

        Returns:
            float: Amount of token X received.
        """
        if amount_y_in <= 0:
            return 0.0

        amt_after_fee, lp_fee_y, protocol_fee_y = self._charge_fee(amount_y_in)

        amount_x_out, _ = self.curve.compute_swap(
            amount_in=amt_after_fee,
            reserve_in=self.reserve_y,
            reserve_out=self.reserve_x,
            fee_rate=0.0,
        )

        self.reserve_y += amount_y_in
        self.reserve_x -= amount_x_out

        if self.reserve_y > 0:
            price = self.reserve_x / self.reserve_y
            self.reserve_x += lp_fee_y * price
            if self.reserve_x > 0:
                self.protocol_reserve_y += protocol_fee_y * (self.reserve_y / self.reserve_x)

        if self.on_swap:
            self.on_swap(self, amount_y_in, amount_x_out, "Y→X")

        return float(amount_x_out)

    def deposit_liquidity(self, provider: Agent, amount_x: float, amount_y: float) -> float:
        """
        Deposit tokens into the pool and receive LP tokens.

        Args:
            provider (Agent): Liquidity provider agent.
            amount_x (float): Amount of token X.
            amount_y (float): Amount of token Y.

        Returns:
            float: Amount of LP tokens minted.
        """
        if amount_x <= 0 or amount_y <= 0:
            return 0.0

        lp_to_mint = self.curve.compute_deposit_lp(
            reserve_x=self.reserve_x,
            reserve_y=self.reserve_y,
            total_lp_supply=self.total_lp_supply,
            amount_x=amount_x,
            amount_y=amount_y,
        )

        self.reserve_x += amount_x
        self.reserve_y += amount_y
        self.total_lp_supply += lp_to_mint
        self.lp_balances[provider.unique_id] = self.lp_balances.get(provider.unique_id, 0.0) + lp_to_mint

        if self.on_deposit:
            self.on_deposit(self, provider, lp_to_mint)

        return float(lp_to_mint)

    def withdraw_liquidity(self, provider: Agent, lp_tokens: float) -> Tuple[float, float]:
        """
        Withdraw liquidity by burning LP tokens.

        Args:
            provider (Agent): Liquidity provider agent.
            lp_tokens (float): Amount of LP tokens to burn.

        Returns:
            Tuple[float, float]: Amounts of token X and Y withdrawn.
        """
        provider_balance = self.lp_balances.get(provider.unique_id, 0.0)
        if lp_tokens <= 0 or lp_tokens > provider_balance or self.total_lp_supply <= 0:
            return 0.0, 0.0

        amount_x_out, amount_y_out = self.curve.compute_withdraw_lp(
            reserve_x=self.reserve_x,
            reserve_y=self.reserve_y,
            total_lp_supply=self.total_lp_supply,
            lp_tokens=lp_tokens,
        )

        self.reserve_x -= amount_x_out
        self.reserve_y -= amount_y_out
        self.total_lp_supply -= lp_tokens
        self.lp_balances[provider.unique_id] = provider_balance - lp_tokens

        if self.on_withdraw:
            self.on_withdraw(self, provider, (amount_x_out, amount_y_out))

        return float(amount_x_out), float(amount_y_out)

    def step(self):
        """AMMs are reactive; no internal logic on each step."""
        pass
