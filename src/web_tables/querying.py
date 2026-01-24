from concurrent.futures import as_completed

from src.database.mp_workers import worker_validate_and_find_answers


class WebTableQueryEngine:
    def __init__(
        self,
        config,
        query_factory,
        limit: int | None = 100,
        multi_hop: bool = False,
        print_query: bool = False,
    ) -> None:
        """
        Initializes the WebTablesQuerryEngine.

        Args:
            config,
            query_factory,
            limit: int = 100 (The maximum amount of tables that is returned in one search.)
            multi_hop: bool = False (A flag that indicates if the multi-hop algorithm will be used.)
            print_query: bool = False (A flag inicating if the queries and parameters will be printed.)

        Returns:
            None
        """

        self.tau = config.tau
        self.query_factory = query_factory
        self.limit = limit
        self.multi_hop = multi_hop
        self.print_query = print_query

    def find_candidates(
        self, x_cols, y_cols, previously_seen_tables: set | None = None
    ) -> set:
        """
        Non-parallel version:
        1) get candidate (table_id, *x_col_ids, *y_col_ids) from find_xy_candidates
        2) validate them with stable_row_val in a single call
        """

        candidates = self.query_factory.find_xy_candidates(
            x_cols=x_cols,
            y_cols=y_cols,
            tau=self.tau,
            multi_hop=self.multi_hop,
            limit=self.limit,
            previously_seen_tables=previously_seen_tables,
            print_query=self.print_query,
        )
        return candidates

    def find_answers_parallel(
        self,
        executor,
        indexes: set[tuple],
        ex_x: list[list[str]],
        ex_y: list[list[str]],
        queries: list[list[str]],
    ):
        idx_list = list(indexes)
        if not idx_list:
            return
            yield

        tasks = [
            executor.submit(
                worker_validate_and_find_answers, idx, ex_x, ex_y, queries, self.tau
            )
            for idx in idx_list
        ]

        for fut in as_completed(tasks):
            res = fut.result()
            if res is None:
                continue
            yield res
