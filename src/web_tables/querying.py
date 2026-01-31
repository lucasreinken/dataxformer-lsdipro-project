from concurrent.futures import as_completed

from src.utils.mp_workers import worker_validate_and_find_answers
from src.database.query_factory import QueryFactory
from src.database.multi_hop import DirectDependencyVerifier


class WebTableQueryEngine:
    def __init__(self, config, query_factory: QueryFactory) -> None:
        """
        Initializes the WebTablesQuerryEngine.

        Args:
            tau: int
            query_factory: QuerryFactory
            limit: int (Default: 100) (The maximum amount of tables that is returned in one search.)
            multi_hop: bool (Default: False) (A flag that indicates if the multi-hop algorithm will be used.)
            print_query: bool (Default: False) (A flag inicating if the queries and parameters will be printed.)

        Returns:
            None
        """

        self.query_factory = query_factory
        self.tau = config.querying.tau
        self.table_limit = config.querying.table_limit
        self.use_multi_hop = config.querying.use_multi_hop
        self.print_query = config.querying.print_query

        if self.use_multi_hop:
            self.fd_verifier = DirectDependencyVerifier(
                self.query_factory, config.multi_hop, self.tau, config.experiment.seed
            )

    def find_candidates(
        self,
        x_cols: list[list[str]],
        y_cols: list[list[str]],
        previously_seen_tables: set | None = None,
    ) -> set:
        """
        Wrapper for QueryFactory.find_xy_candidates.
        Args:
            x_cols: list[list[str]]
            y_cols: list[list[str]]
            previously_seen_tables: set | None (Default: None)

        Returns:
            candidates: set
        """

        candidates = self.query_factory.find_xy_candidates(
            x_cols=x_cols,
            y_cols=y_cols,
            tau=self.tau,
            table_limit=self.table_limit,
            previously_seen_tables=previously_seen_tables,
            print_query=self.print_query,
        )
        return candidates

    def find_answers_parallel(
        self,
        executor,
        indexes: set[tuple],
        x_cols: list[list[str]],
        y_cols: list[list[str]],
        queries: list[list[str]],
        previously_seen_tables=None,
    ):
        if self.use_multi_hop:
            mh_df = self.fd_verifier.my_queue(
                cleaned_x=x_cols,
                cleaned_y=y_cols,
                previously_seen_tables=previously_seen_tables,
                print_query=self.print_query,
            )
            if mh_df is not None and not mh_df.empty:
                x_names = [c for c in mh_df.columns if c.startswith("x_col_")]
                for _, r in mh_df.iterrows():
                    x_part = [r[c] for c in x_names]
                    y_part = [r["y_col_0"]]
                    yield (-1, [[x_part, y_part]])

        idx_list = list(indexes)
        if not idx_list:
            return

        tasks = [
            executor.submit(
                worker_validate_and_find_answers,
                idx,
                x_cols,
                y_cols,
                queries,
                self.tau,
                print_query=self.print_query,
            )
            for idx in idx_list
        ]

        for task in as_completed(tasks):
            result = task.result()
            if result is None:
                continue
            yield result
