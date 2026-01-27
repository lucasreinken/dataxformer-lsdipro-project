from rapidfuzz import fuzz


def fuzzy_score(a: str, b: str, scorer: str = "ratio") -> float:
    """
    Inputs are assumed already tokenized/normalized.
    scorer: 'ratio' | 'token_sort' | 'token_set' | 'max3'
    """
    if not a or not b:
        return 0.0

    if scorer == "ratio":
        return fuzz.ratio(a, b)
    if scorer == "token_sort":
        return fuzz.token_sort_ratio(a, b)
    if scorer == "token_set":
        return fuzz.token_set_ratio(a, b)
    if scorer == "max3":
        return max(
            fuzz.ratio(a, b),
            fuzz.token_sort_ratio(a, b),
            fuzz.token_set_ratio(a, b),
        )
    raise ValueError(f"Unknown scorer: {scorer}")


def fuzzy_match(a: str, b: str, scorer: str = "ratio", threshold: int = 95) -> bool:
    return fuzzy_score(a, b, scorer) >= threshold


def fuzzy_tuple_match(
    t1: tuple, t2: tuple, scorer: str = "ratio", threshold: int = 95
) -> bool:
    if len(t1) != len(t2):
        return False
    return all(fuzzy_match(a, b, scorer, threshold) for a, b in zip(t1, t2))
