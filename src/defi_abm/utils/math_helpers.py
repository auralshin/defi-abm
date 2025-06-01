def accrue_interest(principal: float, rate_per_block: float, blocks: int) -> float:
    """
    Compound `principal` by `rate_per_block` over `blocks` discrete intervals.
    """
    if principal <= 0 or rate_per_block == 0 or blocks <= 0:
        return principal
    return principal * ((1 + rate_per_block) ** blocks)


def get_amount_out(amount_in: float, reserve_in: float, reserve_out: float, fee_rate: float) -> float:
    """
    Compute output amount for a constant‐product AMM swap (Uniswap V2 formula).
    """
    if amount_in <= 0 or reserve_in < 0 or reserve_out <= 0:
        return 0.0

    amount_in_with_fee = amount_in * (1 - fee_rate)
    numerator = amount_in_with_fee * reserve_out
    denominator = reserve_in + amount_in_with_fee
    if denominator == 0:
        return 0.0
    return numerator / denominator