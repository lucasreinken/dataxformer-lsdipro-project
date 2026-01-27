from concurrent.futures import as_completed
from collections.abc import Iterator

from src.database.mp_workers import worker_validate_and_find_answers
from src.database.query_factory import QueryFactory


class WebTableQueryEngine:
    def __init__(
        self,
        tau: int,
        query_factory: QueryFactory,
        limit: int | None = 100,
        multi_hop: bool = False,
        print_query: bool = False,
        use_fuzzy_matching: bool = False,
        fuzzy_scorer: str = "ratio",
        fuzzy_threshold: float = 0.95,
    ) -> None:
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

        self.tau = tau
        self.query_factory = query_factory
        self.limit = limit
        self.multi_hop = multi_hop
        self.print_query = print_query
        self.use_fuzzy_matching = use_fuzzy_matching
        self.fuzzy_scorer = fuzzy_scorer
        self.fuzzy_threshold = fuzzy_threshold

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
            multi_hop=self.multi_hop,
            limit=self.limit,
            previously_seen_tables=previously_seen_tables,
        )
        return candidates

    def find_answers_parallel(
        self,
        executor,
        indexes: set[tuple],
        x_cols: list[list[str]],
        y_cols: list[list[str]],
        queries: list[list[str]],
    ) -> Iterator[tuple[int, list]]:
        """
        Finds answers to the open queries in a distributed setting, performing as a generator.

        Args:
            executor
            indexes: set[tuple]
            x_cols: list[list[str]]
            y_cols: list[list[str]]
            queries: list[list[str]]

        Yields:
            tuple(table_id: int, answer_list: list)
        """
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
                self.use_fuzzy_matching,
                self.fuzzy_scorer,
                self.fuzzy_threshold,
            )
            for idx in idx_list
        ]

        for task in as_completed(tasks):
            result = task.result()
            if result is None:
                continue
            yield result
