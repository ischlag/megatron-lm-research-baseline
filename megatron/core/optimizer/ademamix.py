"""AdEMAMix warmup schedulers used by the master optimizer."""

import math


def linear_warmup_scheduler(
    step: int, alpha_end: float, alpha_start: float = 0.0, warmup: int = 1
) -> float:
    """Linear warmup from alpha_start to alpha_end over `warmup` steps."""
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(
    step: int, beta_end: float, beta_start: float = 0.0, warmup: int = 1
) -> float:
    """Half-life-linear warmup from beta_start to beta_end over `warmup` steps.

    The interpolation is linear in half-life space, then mapped back to a beta
    coefficient. Used for AdEMAMix's beta3 schedule.
    """
    def f(beta: float, eps: float = 1e-8) -> float:
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t: float) -> float:
        return math.pow(0.5, 1 / (t + 1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end
