from src.database.query_factory import QueryFactory

import atexit
import pandas as pd

# --------------------------
# Per-process worker state
# --------------------------
_WORKER_QF = None


def init_worker_qf(vertica_config: dict):
    """Runs once per worker process."""
    global _WORKER_QF
    qf = QueryFactory(vertica_config)
    qf.__enter__()
    _WORKER_QF = qf
    atexit.register(lambda: qf.__exit__(None, None, None))


# --------------------------
# worker functions
# --------------------------
def worker_validate_and_find_answers(index, ex_x, ex_y, queries, tau, print_query):
    global _WORKER_QF

    table_id = index[0]
    col_ids = list(index[1:])

    df = _WORKER_QF.get_table_content(
        table_id=table_id, include_cols=col_ids, print_query=print_query
    )
    if df is None:
        return None
    if any(c not in df.columns for c in col_ids):
        return None

    # --------------------------
    # Validation (hashed rows)
    # --------------------------
    val_df = df[col_ids]

    ex_df = pd.DataFrame(list(zip(*ex_x, *ex_y)), columns=col_ids)
    ex_hash = pd.util.hash_pandas_object(ex_df, index=False)

    row_hash = pd.util.hash_pandas_object(val_df, index=False)

    hits = int(row_hash.isin(set(ex_hash)).sum())
    if hits < tau:
        return None

    # --------------------------
    # Query answering (merge)
    # --------------------------
    x_len = len(ex_x)
    y_len = len(ex_y)
    x_cols = list(index[1 : 1 + x_len])
    y_cols = list(index[1 + x_len : 1 + x_len + y_len])

    cond_rows = list(zip(*queries))
    if not cond_rows:
        return table_id, []

    cond_df = pd.DataFrame(cond_rows, columns=x_cols).drop_duplicates()

    base = df[x_cols + y_cols]
    matched = base.merge(cond_df, on=x_cols, how="inner")

    if matched.empty:
        return table_id, []

    # --------------------------
    # Build answer_list without row-wise apply
    # --------------------------
    arr = matched.to_numpy()
    x_idx = list(range(len(x_cols)))
    y_idx = list(range(len(x_cols), len(x_cols) + len(y_cols)))

    answer_list = [
        [arr[i, x_idx].tolist(), arr[i, y_idx].tolist()] for i in range(arr.shape[0])
    ]

    return table_id, answer_list
