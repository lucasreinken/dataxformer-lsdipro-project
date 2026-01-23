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


def group_by_table_id(index_list):
    by_table = {}

    for idx in index_list:
        table_id = idx[0]
        if table_id in by_table:
            by_table[table_id].append(idx)
        else:
            by_table[table_id] = [idx]

    return by_table


# --------------------------
# worker functions
# --------------------------
def worker_stable_row_val(index_list_chunk, x_cols, y_cols, tau):
    global _WORKER_QF
    return set(_WORKER_QF.stable_row_val(index_list_chunk, x_cols, y_cols, tau))


def worker_find_answers(indices, Q, len_x):
    global _WORKER_QF

    Q = tuple(tuple(q) for q in Q)
    conditions = list(zip(*Q))

    out = []
    for index in indices:
        table_df = _WORKER_QF.get_table_content(index[0], index[1:])

        x_cols = list(index[1 : 1 + len_x])
        y_cols = list(index[1 + len_x :])

        mask = table_df.apply(
            lambda row: any(
                all(row[x_cols[i]] == cond[i] for i in range(len_x))
                for cond in conditions
            ),
            axis=1,
        )

        answer_list = (
            table_df.loc[mask]
            .apply(lambda row: [row[x_cols].tolist(), row[y_cols].tolist()], axis=1)
            .tolist()
        )

        out.append((index[0], answer_list))

    return out
