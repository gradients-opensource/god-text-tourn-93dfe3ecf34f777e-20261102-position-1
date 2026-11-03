import math

def _suggest_learning_rates(
    best_lr: float,
    n: int,
    log_range: float = 0.4
) -> list[float]:
    if n < 0:
        raise ValueError("Number of tries (n) cannot be negative.")
    if n == 0:
        return []
    if n == 1:
        return [best_lr]

    lower_bound = best_lr / (10 ** log_range)
    upper_bound = best_lr * (10 ** log_range)

    log_lower = math.log10(lower_bound)
    log_upper = math.log10(upper_bound)

    log_spaced_values = [
        log_lower + i * (log_upper - log_lower) / (n - 1)
        for i in range(n)
    ]

    learning_rates = [10 ** val for val in log_spaced_values]

    return sorted(learning_rates)


def suggest_learning_rates(
    best_lr: float,
    n: int,
    log_range: float = 0.2
) -> list[float]:
    lrs = _suggest_learning_rates(best_lr, n, log_range)
    if n % 2 == 1:
        return lrs
    else:
        lrs = lrs[1:] + [best_lr]
        lrs = sorted(lrs)
        return lrs


def extend_learning_rates(
    lr: float,
    n: int,
    log_range: float = 0.2
) -> list[float]:
    lrs = _suggest_learning_rates(lr, n, log_range)
    closest_idx = min(range(len(lrs)), key=lambda i: abs(lrs[i] - lr))
    lrs[closest_idx] = lr
    if closest_idx != 0:
        lrs.insert(0, lrs.pop(closest_idx))
    return lrs


def test():
    lr = 0.00014523947500000002
    for n in [3,4,5, 6]:
        lrs = extend_learning_rates(lr, n)
        print(lrs)
        assert lrs[0] == lr

if __name__ == "__main__":
    test()
