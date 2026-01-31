from concurrent.futures import ProcessPoolExecutor

from datetime import datetime
import time
from copy import deepcopy
import random

from src.database.query_factory import QueryFactory
from src.web_tables.querying import WebTableQueryEngine
from src.utils.mp_workers import init_worker_qf

from pprint import pprint

from src.utils.fuzzy_matching import fuzzy_tuple_match


class WebTableRanker:
    def __init__(
        self,
        config,
    ) -> None:
        """
        Initialize the WebTableRanker.

        Args:
            config: MasterConfig

        Returns:
            None
        """
        self.epsilon = config.ranker.epsilon
        self.alpha = config.ranker.alpha
        self.max_requery_iterations = config.ranker.max_requery_iterations
        self.table_prior = config.ranker.table_prior
        self.topk = config.ranker.topk
        self.max_requery_answers = config.ranker.max_requery_answers
        self.use_fuzzy_matching = config.ranker.use_fuzzy_matching
        self.use_majority_voting = config.ranker.use_majority_voting
        self.fuzzy_scorer = config.ranker.fuzzy_scorer
        self.fuzzy_threshold = config.ranker.fuzzy_threshold
        self.max_workers = config.ranker.parallel_workers
        self.max_iterations = config.ranker.max_iterations

        self.vertica_config = config.vertica
        self.query_factory = QueryFactory(self.vertica_config)
        self.query_engine = WebTableQueryEngine(config, self.query_factory)

    # ---------- debug helpers ----------
    @staticmethod
    def _now_str() -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _log(self, msg: str, enabled: bool) -> None:
        if enabled:
            print(f"[{self._now_str()}] {msg}")

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
        debug_timing: bool = False,
    ) -> dict:
        """
        Run the Expectation-Maximization loop to rank answers and tables.

        Args:
            x_input: list[list[str]]
            y_input: list[list[str]]
            queries: list[list[str]]
            print_iterations: bool (Default: False)
            debug_timing: bool (Default: False)

        Returns:
            result: dict
        """
        finished_querying = False
        delta_score = float("inf")
        iteration = 0
        previously_seen_tables = set()
        old_scores = dict()
        old_y_values = list()
        old_query_y_values = list()
        answers_by_x = {}
        tables_by_x = {}

        answers = {
            (tuple(col_x), tuple(col_y)): {"score": 1.0, "tables": set(), "term": None}
            for col_x, col_y in zip(zip(*x_input), zip(*y_input))
            if None not in col_x and None not in col_y
        }

        tables = dict()

        q_cols = list({tuple(col) for col in zip(*queries) if None not in col})
        rearranged_queries = [list(row) for row in zip(*q_cols)]

        self._log(
            f"Starting EM with max_workers={self.max_workers}, "
            f"epsilon={self.epsilon}, alpha={self.alpha}, max_requery_iterations={self.max_requery_iterations}, topk={self.topk}",
            debug_timing,
        )

        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=init_worker_qf,
            initargs=(self.vertica_config,),
        ) as ex:
            self._log("ProcessPoolExecutor created", debug_timing)

            while (
                delta_score > self.epsilon or not finished_querying
            ) and iteration < self.max_iterations:
                iteration += 1
                iter_start = time.perf_counter()

                if print_iterations:
                    print("")
                    print(f"Current EM Iteration: {iteration}")

                self._log(
                    f"EM iteration {iteration} START (delta_score={delta_score}, finished_querying={finished_querying})",
                    debug_timing,
                )

                t1 = None

                if iteration > (self.max_requery_iterations + 1):
                    finished_querying = True

                if not finished_querying:
                    finished_querying = True
                    query_x_values = list()
                    query_y_values = list()
                    x_to_existing_ys = dict()

                    if not tables:
                        query_values = [
                            q_list + x_list
                            for q_list, x_list in zip(rearranged_queries, x_input)
                        ]
                        x_values = x_input
                        y_values = y_input
                        self._log(
                            f"Iteration {iteration}: initial querying mode (tables empty). query_values rows={len(query_values)}",
                            debug_timing,
                        )
                    else:
                        query_values = rearranged_queries
                        requery_pairs = list()
                        x_to_ys = dict()

                        for (x_answer, y_answer), info in answers.items():
                            x_to_ys.setdefault(x_answer, []).append(
                                (y_answer, float(info["score"]))
                            )
                            x_to_existing_ys.setdefault(x_answer, []).append(y_answer)

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

                    if x_values and len(x_values[0]) > self.max_requery_answers:
                        n_cols = len(x_values[0])

                        keep_idx = random.sample(
                            range(n_cols), self.max_requery_answers
                        )

                        query_x_values = [
                            [row[i] for i in keep_idx] for row in x_values
                        ]
                        query_y_values = [
                            [row[i] for i in keep_idx] for row in y_values
                        ]
                    else:
                        query_x_values = x_values
                        query_y_values = y_values

                    if (
                        query_x_values is None
                        or query_x_values[0] is None
                        or old_y_values != y_values
                        or old_query_y_values != query_y_values
                    ):
                        old_y_values = deepcopy(y_values)
                        old_query_y_values = deepcopy(query_y_values)

                        t0 = time.perf_counter()
                        candidates = self.query_engine.find_candidates(
                            x_cols=query_x_values,
                            y_cols=query_y_values,
                            previously_seen_tables=previously_seen_tables,
                        )
                        self._log(
                            f"find_candidates took {time.perf_counter() - t0:.3f}s (candidates={len(candidates)})",
                            debug_timing,
                        )
                        previously_seen_tables.update(c[0] for c in candidates)
                        t1 = time.perf_counter()
                        n_tables_streamed = 0
                        n_answers_streamed = 0
                        for (
                            table_id,
                            answer_list,
                        ) in self.query_engine.find_answers_parallel(
                            executor=ex,
                            indexes=candidates,
                            x_cols=query_x_values,
                            y_cols=query_y_values,
                            queries=query_values,
                            previously_seen_tables=previously_seen_tables,
                        ):
                            n_tables_streamed += 1
                            for untupled_answer in answer_list:
                                if (
                                    None in untupled_answer[1]
                                    or "" in untupled_answer[1]
                                ):
                                    continue

                                y_term = tuple(untupled_answer[1])
                                answer = (
                                    tuple(untupled_answer[0]),
                                    tuple(untupled_answer[1]),
                                )

                                canonical_answer = answer

                                if self.use_fuzzy_matching:
                                    x_key, y_key = answer

                                    for existing_y in x_to_existing_ys.get(x_key, []):
                                        if fuzzy_tuple_match(
                                            existing_y,
                                            y_key,
                                            scorer=self.fuzzy_scorer,
                                            threshold=self.fuzzy_threshold,
                                        ):
                                            canonical_answer = (x_key, existing_y)
                                            break

                                if canonical_answer not in answers:
                                    finished_querying = False
                                    answers[canonical_answer] = {
                                        "score": 0.0,
                                        "tables": set(),
                                        "term": y_term,
                                    }
                                    answers_by_x.setdefault(
                                        canonical_answer[0], set()
                                    ).add(canonical_answer)

                                    x_to_existing_ys.setdefault(
                                        canonical_answer[0], []
                                    ).append(canonical_answer[1])

                                answer = canonical_answer

                                tables.setdefault(
                                    table_id, {"score": 0.0, "answers": set()}
                                )

                                answers[answer]["tables"].add(table_id)
                                tables[table_id]["answers"].add(answer)
                                tables_by_x.setdefault(answer[0], set()).add(table_id)
                                n_answers_streamed += 1

                if t1:
                    self._log(
                        f"find_answers_parallel took {time.perf_counter() - t0:.3f}s "
                        f"(tables_streamed={n_tables_streamed}, answers_added/seen={n_answers_streamed}, tables_total={len(tables)}, answers_total={len(answers)})",
                        debug_timing,
                    )

                if self.use_majority_voting:
                    answers = self.update_answer_scores_majority(
                        answers=answers,
                        answers_by_x=answers_by_x,
                        rearranged_queries=rearranged_queries,
                        print_winner=print_iterations,
                    )

                else:
                    tables = self.update_table_scores(
                        answers=answers,
                        tables=tables,
                        print_table_score=print_iterations,
                    )

                    answers = self.update_answer_scores(
                        answers=answers,
                        tables=tables,
                        answers_by_x=answers_by_x,
                        tables_by_x=tables_by_x,
                        rearranged_queries=rearranged_queries,
                        print_prob_of_none=print_iterations,
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

                iter_end = time.perf_counter()
                self._log(
                    f"EM iteration {iteration} END total={iter_end - iter_start:.3f}s",
                    debug_timing,
                )

        result = {
            x: [
                (info["term"], info["score"])
                for (x2, _), info in answers.items()
                if x2 == x
            ]
            for x in zip(*rearranged_queries)
        }
        return result

    def update_table_scores(
        self,
        answers: dict,
        tables: dict,
        print_table_score: bool = False,
    ) -> dict:
        """
        Update table reliability scores given current answer probabilities.

        Args:
            answers: dict
            tables: dict
            print_table_score: bool (Default: False)

        Returns:
            tables: dict
        """
        if print_table_score:
            print("")

        x_best_score = dict()
        for (x, _), info in answers.items():
            current_score = info["score"]
            x_best_score[x] = max(x_best_score.get(x, 0.0), current_score)

        total_possible_best_score = sum(x_best_score.values())

        # for all t ∈ tables do
        for table_id, table_info in tables.items():
            # good ← 0
            good = 0.0
            # bad ← 0
            bad = 0.0
            # coveredXs ← {} //holds example x’s appearing in t
            covered_query_answers_x = set()
            table_covered_best_sum = 0.0

            answer_set = table_info.get("answers", set())
            # for all answer(x, y) ∈ t do
            for table_answer in answer_set:
                x, _ = table_answer

                if x not in covered_query_answers_x:
                    # coveredXs ← coveredXs ∪ {x}
                    covered_query_answers_x.add(x)
                    table_covered_best_sum += x_best_score[x]

                # score ← GetScore(x, y)
                answer_score = answers[table_answer]["score"]
                # if IsMax(score, x) then
                if answer_score == x_best_score[x]:
                    # good ← good + score
                    good += answer_score
                # else
                else:
                    # bad ← bad + 1
                    bad += 1

            # unseen_X ← ∑_{x ∉ covered_X, (x,y) ∈ answers} max_y score(x,y)
            unseen_x = total_possible_best_score - table_covered_best_sum

            denominator = (self.table_prior * good) + (1 - self.table_prior) * (
                bad + unseen_x
            )

            # score(t) ← α · (prior · good) / (prior · good + (1 − prior) · (bad + unseenX))
            table_score = self.alpha * ((self.table_prior * good) / denominator)

            if print_table_score:
                print(
                    f"Score of table {table_id} ({table_score:.4f}): {covered_query_answers_x}"
                )

            tables[table_id]["score"] = table_score

        return tables

    def update_answer_scores(
        self,
        answers: dict,
        tables: dict,
        answers_by_x: dict,
        tables_by_x: dict,
        rearranged_queries: list[list[str]],
        print_prob_of_none: bool = False,
    ) -> dict:
        """
        Update answer probabilities given current table scores.

        Args:
            answers: dict
            tables: dict
            rearranged_queries: list[list[str]]
            print_prob_of_none: bool (Default: False)

        Returns:
            answers: dict
        """
        if print_prob_of_none:
            print("")

        # score(x, y) := 1
        for _, info in answers.items():
            info["score"] = 1.0

        # for all x ∈ X do
        for x_q, x_answers in answers_by_x.items():
            # A ← answers.getAnswers(x)

            tables_for_x = tables_by_x[x_q]

            # scoreOfNone ← 1
            score_of_none = 1.0

            # for all table ∈ answers.getT ables(x) do
            for table_id in tables_for_x:
                score_of_none *= 1.0 - tables[table_id]["score"]

                # for all (x, y) ∈ A do
                for answer in x_answers:
                    # if table supports (x, y) then
                    if answer in tables[table_id]["answers"]:
                        # score(x, y) ← score(x, y) · table.score
                        answers[answer]["score"] *= tables[table_id]["score"]
                    # else
                    else:
                        # score(x, y) ← score(x, y) · (1 − table.score)
                        answers[answer]["score"] *= 1 - (tables[table_id]["score"])

            denominator = score_of_none + sum(
                (answers[answer]["score"] for answer in x_answers)
            )

            # for all (x, y) ∈ A do
            for answer in x_answers:
                # score(x, y) <- score(x, y) / (score_of_none + Σ_{(x,y) in A} score(x, y))
                answers[answer]["score"] /= denominator

            if print_prob_of_none:
                print(f"Probability of None for {x_q}: {score_of_none / denominator}")

        return answers

    def update_answer_scores_majority(
        self,
        answers: dict,
        answers_by_x: dict,
        rearranged_queries: list[list[str]],
        print_winner: bool,
    ) -> dict:
        """
        For each x:
        score(x,y)=1 for the y with the most supporting tables (ties broken deterministically)
        score(x,y)=0 for all other y
        Support is simply len(answers[(x,y)]['tables']).
        """

        for info in answers.values():
            info["score"] = 0.0

        for x_q, x_answers in answers_by_x.items():
            best = None
            best_support = -1

            for ans in x_answers:
                support = len(answers[ans]["tables"])
                if support > best_support:
                    best_support = support
                    best = ans
                elif support == best_support and best is not None:
                    if ans[1] < best[1]:
                        best = ans

            if best is not None:
                answers[best]["score"] = 1.0

            if print_winner:
                print(f"[MAJ] x={x_q} winner={best} support={best_support}")

        return answers
