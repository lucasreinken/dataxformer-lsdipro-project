from src.database.query_factory import QueryFactory

import atexit

# --------------------------
# Per-process worker state
# --------------------------
_WORKER_QF = None


def init_worker_qf(vertica_config: dict):
    """Runs once per worker process."""
    global _WORKER_QF
    qf = QueryFactory(vertica_config)
    qf.__enter__()  # open connection once
    _WORKER_QF = qf
    atexit.register(lambda: qf.__exit__(None, None, None))


# --------------------------
# worker functions
# --------------------------
def worker_validate_and_find_answers(
    index, ex_x, ex_y, queries, tau, use_fuzzy_matching, fuzzy_scorer, fuzzy_threshold
):
    """
    index: (table_id, *x_col_ids, *y_col_ids)
    Returns (table_id, answer_list) or None if validation fails / no table.
    """
    global _WORKER_QF

    table_id = index[0]
    col_ids = index[1:]

    df = _WORKER_QF.get_table_content(
        table_id=table_id, include_cols=col_ids, print_query=False
    )
    if df is None:
        return None

    col_ids = list(index[1:])  # all X and Y columns in correct order

    if any(c not in df.columns for c in col_ids):
        return None

    # Precompute example row tuples
    ex_rows = set(zip(*ex_x, *ex_y))  # set = O(1) lookup

    hits = 0
    for _, row in df.iterrows():
        row_tup = tuple(row[col_ids].tolist())
        if row_tup in ex_rows:
            hits += 1
            if hits >= tau:
                break

    if hits < tau:
        return None

    # compute answers for queries (same logic as before)
    Q = tuple(tuple(q) for q in queries)
    conditions = list(zip(*Q))
    len_qx = len(queries)

    x_len = len(ex_x)
    y_len = len(ex_y)

    x_cols = list(index[1 : 1 + x_len])
    y_cols = list(index[1 + x_len : 1 + x_len + y_len])

    mask = df.apply(
        lambda r: any(
            all(r[x_cols[i]] == cond[i] for i in range(len_qx)) for cond in conditions
        ),
        axis=1,
    )

    answer_list = (
        df.loc[mask]
        .apply(lambda r: [r[x_cols].tolist(), r[y_cols].tolist()], axis=1)
        .tolist()
    )

    return table_id, answer_list
