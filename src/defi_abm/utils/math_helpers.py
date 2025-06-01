def accrue_interest(principal: float, rate_per_block: float, blocks: int) -> float:
    """
    Apply compound interest to a principal over a number of discrete blocks.

    This function compounds interest using the formula:
    `principal * (1 + rate_per_block) ** blocks`.

    Parameters
    ----------
    principal : float
        The starting amount of value (e.g., borrowed amount).
    rate_per_block : float
        The interest rate applied per block (e.g., 0.0001 for 0.01%).
    blocks : int
        Number of discrete blocks over which to apply the interest.

    Returns
    -------
    float
        The final amount after interest has been compounded.

    Notes
    -----
    Returns the original principal if the rate is zero or blocks are non-positive.
    """
    if principal <= 0 or rate_per_block == 0 or blocks <= 0:
        return principal
    return principal * ((1 + rate_per_block) ** blocks)


def get_amount_out(amount_in: float, reserve_in: float, reserve_out: float, fee_rate: float) -> float:
    """
    Compute the output amount from a swap using the constant-product AMM formula.

    This is based on Uniswap V2's logic:
    `amount_out = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)`

    Parameters
    ----------
    amount_in : float
        The input token amount being swapped into the pool.
    reserve_in : float
        The amount of input token currently in the pool.
    reserve_out : float
        The amount of output token currently in the pool.
    fee_rate : float
        The swap fee rate (e.g., 0.003 for 0.3%).

    Returns
    -------
    float
        The amount of output token the user receives after the swap.

    Notes
    -----
    Returns 0.0 if input is non-positive or reserves are invalid.
    """
    if amount_in <= 0 or reserve_in < 0 or reserve_out <= 0:
        return 0.0

    amount_in_with_fee = amount_in * (1 - fee_rate)
    numerator = amount_in_with_fee * reserve_out
    denominator = reserve_in + amount_in_with_fee
    if denominator == 0:
        return 0.0
    return numerator / denominator
