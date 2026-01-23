from concurrent.futures import ProcessPoolExecutor


from src.database.query_factory import QueryFactory
from src.web_tables.querying import WebTableQueryEngine
from src.config import get_default_querying_config
from src.database.mp_workers import init_worker_qf

from pprint import pprint


class WebTableRanker:
    def __init__(
        self,
        ranking_config,
        testing_config,
        vertica_config,
        tau: int | None = None,
    ) -> None:
        """
        Initialize the WebTableRanker.

        Args:
            ranking_config
            testing_config
            vertica_config
            tau: int | None = None

        Returns:
            None
        """
        self.epsilon = ranking_config.epsilon
        self.alpha = ranking_config.alpha
        self.max_iterations = ranking_config.max_iterations
        self.table_prior = ranking_config.table_prior

        self.topk = ranking_config.topk

        querying_config = get_default_querying_config()
        if tau:
            querying_config.tau = tau

        self.vertica_config = vertica_config

        self.query_factory = QueryFactory(self.vertica_config)
        self.query_engine = WebTableQueryEngine(querying_config, self.query_factory)
        self.max_workers = ranking_config.max_workers

    def __enter__(self):
        self.query_factory.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.query_factory.__exit__(exc_type, exc_val, exc_tb)

    def expectation_maximization(
        self,
        x_input: list[list[str]],
        y_input: list[list[str]],
        queries: list[list[str]],
        print_iterations: bool = False,
    ) -> dict:
        """
        Run the Expectation-Maximization loop to rank answers and tables.

        Args:
            x_input: list[list[str]]
            y_input: list[list[str]]
            queries: list[list[str]]

        Returns:
            result: dict
        """
        finished_querying = False
        delta_score = float("inf")
        iteration = 0
        previously_seen_tables = set()
        old_scores = dict()

        answers = {
            (tuple(col_x), tuple(col_y)): {"score": 1.0, "tables": set(), "term": None}
            for col_x, col_y in zip(zip(*x_input), zip(*y_input))
            if None not in col_x and None not in col_y
        }

        tables = dict()

        q_cols = list({tuple(col) for col in zip(*queries) if None not in col})
        rearranged_queries = [list(row) for row in zip(*q_cols)]

        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=init_worker_qf,
            initargs=(self.vertica_config,),
        ) as ex:
            while (
                delta_score > self.epsilon or not finished_querying
            ) and iteration < self.max_iterations:
                iteration += 1

                if print_iterations:
                    print("")
                    print(f"Current EM Iteration: {iteration}")

                if not finished_querying:
                    finished_querying = True

                    if not tables:
                        query_values = [
                            q_list + x_list
                            for q_list, x_list in zip(rearranged_queries, x_input)
                        ]
                        x_values = x_input
                        y_values = y_input
                    else:
                        query_values = rearranged_queries
                        requery_pairs = list()
                        x_to_ys = dict()

                        for (x_answer, y_answer), info in answers.items():
                            x_to_ys.setdefault(x_answer, []).append(
                                (y_answer, float(info["score"]))
                            )

                        for x in x_to_ys:
                            ys = x_to_ys[x]

                            total = sum(score for _, score in ys)
                            prob_of_none = max(0.0, 1.0 - total)

                            ys_sorted = sorted(ys, key=lambda t: t[1], reverse=True)
                            answer_limit = min(self.topk, len(ys_sorted))

                            for idx in range(answer_limit):
                                y, score = ys_sorted[idx]
                                if score > prob_of_none:
                                    requery_pairs.append((x, y))

                        x_cols = [x_part for x_part, _ in requery_pairs]
                        y_cols = [y_part for _, y_part in requery_pairs]
                        x_values = [list(row) for row in zip(*x_cols)] if x_cols else []
                        y_values = [list(row) for row in zip(*y_cols)] if y_cols else []

                    candidates = self.query_engine.find_columns(
                        x_values, y_values, previously_seen_tables
                    )

                    previously_seen_tables.update(c[0] for c in candidates)

                    for (
                        table_id,
                        answer_list,
                    ) in self.query_engine.find_answers_parallel(
                        executor=ex,
                        table_ids=candidates,
                        queries=query_values,
                    ):
                        for untupled_answer in answer_list:
                            if None in untupled_answer[1]:
                                continue

                            y_term = tuple(untupled_answer[1])
                            answer = (
                                tuple(untupled_answer[0]),
                                tuple(untupled_answer[1]),
                            )

                            if answer not in answers:
                                finished_querying = False
                                answers[answer] = {
                                    "score": 0.0,
                                    "tables": set(),
                                    "term": y_term,
                                }

                            tables.setdefault(
                                table_id, {"score": 0.0, "answers": set()}
                            )

                            answers[answer]["tables"].add(table_id)
                            tables[table_id]["answers"].add(answer)

                tables = self.update_table_scores(
                    answers,
                    tables,
                    print_iterations,
                )

                answers = self.update_answer_scores(
                    answers,
                    tables,
                    rearranged_queries,
                    print_iterations,
                )

                if finished_querying and old_scores:
                    delta_score = sum(
                        abs(answers[a]["score"] - old_scores.get(a, 0.0))
                        for a in answers
                    )

                old_scores = {key: info["score"] for key, info in answers.items()}

                if print_iterations:
                    result = {
                        x: [
                            (info["term"], info["score"])
                            for (x2, _), info in answers.items()
                            if x2 == x
                        ]
                        for x in zip(*rearranged_queries)
                    }
                    pprint(result)

        result = {
            x: [
                (info["term"], info["score"])
                for (x2, _), info in answers.items()
                if x2 == x
            ]
            for x in zip(*rearranged_queries)
        }
        return result

    def update_table_scores(self, answers, tables, print_):
        """
        Update table reliability scores given current answer probabilities.
        """
        if print_:
            print("")

        x_best_score = {}
        for (x, _), info in answers.items():
            score = info["score"]
            if x not in x_best_score or score > x_best_score[x]:
                x_best_score[x] = score

        for table_id in list(tables.keys()):
            good = 0
            bad = 0
            covered_query_answers_x = set()

            ans_set = tables[table_id]["answers"]
            for table_answer in ans_set:
                x, _ = table_answer
                covered_query_answers_x.add(x)

                answer_score = answers[table_answer]["score"]

                if answer_score == x_best_score[x]:
                    good += answer_score
                else:
                    bad += 1

            unseen_x = 0.0
            for x, best_score in x_best_score.items():
                if x not in covered_query_answers_x:
                    unseen_x += best_score

            table_score = self.alpha * (
                (self.table_prior * good)
                / (self.table_prior * good + (1 - self.table_prior) * (bad + unseen_x))
            )

            if print_:
                print(
                    f"Score of table {table_id} ({table_score}): {covered_query_answers_x}"
                )

            tables[table_id]["score"] = table_score

        return tables

    def update_answer_scores(self, answers, tables, Q, print_):
        """
        Update answer probabilities given current table scores.
        """
        if print_:
            print("")

        for _, info in answers.items():
            info["score"] = 1.0

        q_cols = list(zip(*Q))
        for x_q in q_cols:
            x_answers = {(x, y) for (x, y) in answers if x == x_q}

            tables_for_x = {
                table_id
                for (x, _), info in answers.items()
                if x == x_q
                for table_id in info["tables"]
            }

            score_of_none = 1

            for table_id in tables_for_x:
                score_of_none *= 1 - tables[table_id]["score"]

                for answer in x_answers:
                    if answer in tables[table_id]["answers"]:
                        answers[answer]["score"] *= tables[table_id]["score"]
                    else:
                        answers[answer]["score"] *= 1 - (tables[table_id]["score"])

            denominator = score_of_none + sum(
                (answers[answer]["score"] for answer in x_answers)
            )

            for answer in x_answers:
                answers[answer]["score"] /= denominator

            if print_:
                print(f"Probability of None for {x_q}: {score_of_none / denominator}")

        return answers
