from concurrent.futures import as_completed


from src.database.mp_workers import worker_find_answers

from collections.abc import Iterator


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

    def find_columns(
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
        if not candidates:
            return set()

        # stable_row_val returns a list/iterable of validated index tuples
        return set(
            self.query_factory.stable_row_val(candidates, x_cols, y_cols, self.tau)
        )

    def find_answers_parallel(
        self,
        executor,
        table_ids: set[int],
        queries: list[list[str]],
    ) -> Iterator[tuple[int, list[str]]]:
        table_list = list(table_ids)
        if not table_list:
            return
            yield

        len_x = len(queries)

        # one task per table
        tasks = [
            executor.submit(worker_find_answers, [table_id], queries, len_x)
            for table_id in table_list
        ]

        for task in as_completed(tasks):
            for table_id, answer_list in task.result():
                yield table_id, answer_list
