from concurrent.futures import as_completed


from src.database.mp_workers import (
    chunk_list,
    group_by_table_id,
    worker_find_columns_chunk,
    worker_find_answers,
)

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
        self.query_factory = query_factory  # used for non-parallel calls
        self.limit = limit
        self.multi_hop = multi_hop
        self.print_query = print_query

    def find_columns_parallel(
        self,
        executor,
        x_cols: list[list[str]],
        y_cols: list[list[str]],
        chunk_size: int = 200,
        previously_seen_tables: set | None = None,
    ) -> set:
        """
        Executes the table_id and col_id search in a distributed setting.

        Args:
            executor
            x_cols: list[list[str]] (A list over all x-columns, each consisting of a list of string values.)
            y_cols: list[list[str]] (A list over all y-columns, each consisting of a list of string values.)
            chunk_size: int = 200   (Size of one chunk that is processed by one worker.)
            previously_seen_tables: set | None = None (A set of previously seen tables that get excluded from the table search.)

        Returns:
            candidates: set (A set containing the table_ids and col_ids of candidate tables.)
        """

        table_candidates = self.query_factory.find_xy_candidates(
            x_cols=x_cols,
            y_cols=y_cols,
            tau=self.tau,
            multi_hop=self.multi_hop,
            limit=self.limit,
            previously_seen_tables=previously_seen_tables,
            print_query=self.print_query,
        )
        if not table_candidates:
            return set()

        by_table = group_by_table_id(table_candidates)
        table_ids = list(by_table.keys())
        table_id_chunks = list(chunk_list(table_ids, chunk_size))

        tasks = list()
        for table_ids_in_chunk in table_id_chunks:
            idx_chunk = list()

            for table_id in table_ids_in_chunk:
                idx_chunk.extend(by_table[table_id])

            if idx_chunk:
                tasks.append(
                    executor.submit(
                        worker_find_columns_chunk, idx_chunk, x_cols, y_cols, self.tau
                    )
                )

        candidates = set()
        for task in as_completed(tasks):
            candidates.update(task.result())

        return candidates

    def find_answers_parallel(
        self,
        executor,
        table_ids: set[int],
        queries: list[list[str]],
        chunk_size: int = 10,
    ) -> Iterator[tuple[int, list[str]]]:
        """
        Executes the answer search in a distributed setting as a generator.

        Args:
            executor
            table_ids: set (Set of relevant tables where answers should be retrieved from.)
            queries: list[list[str]] (A list over all query-columns, each consisting of a list of string values.)
            chunk_size: int = 10 (Size of one chunk that is processed by one worker.)

        Yields:
            tuple(table_id: int, answer_list: list[str])
            (A tuple containing the table_id as well as a list of possible answers for a query found in the table.)
        """

        table_list = list(table_ids)
        if not table_list:
            return
            yield

        num_queries = len(queries)
        chunks = list(chunk_list(table_list, chunk_size))
        tasks = [
            executor.submit(worker_find_answers, chunk, queries, num_queries)
            for chunk in chunks
        ]

        for task in as_completed(tasks):
            for table_id, answer_list in task.result():
                yield table_id, answer_list
