from copy import deepcopy
from src.web_tables.querying import WebTableQueryEngine
from src.config import get_default_querying_config

class WebTableRanker:
    def __init__(self, config):

        self.epsilon = config.epsilon
        self.alpha = config.alpha
        self.max_iterations = config.max_iterations

        querying_config = get_default_querying_config()
        self.query_engine = WebTableQueryEngine(querying_config)

    def score(self, candidates, query_context):
        return

    def rank(self, candidates, query_context):
        scores = self.score(candidates, query_context)
        return

    def expectation_maximization(self, X: list[str], Y: list[str], Q: list[str]):
        # TODO: docstring (X, Y, Q have to be tokenized as input -> return untokenized y values)
        
        # TODO: store them more lightweight (e.g. decouple score)
        # answers: (x,y) → {score, tables={t1,t2,…}}
        # tables:  t_id → {score, answers={(x1,y1),(x2,y2)…}}

        answers = {
            (tuple(col_x), tuple(col_y)): {"score": 1.0, "tables": set(), "term": None}
            for col_x, col_y in zip(zip(*X), zip(*Y))
        }
        # TODO: add Jaccard Similiratity (if labels provided)
        tables = {}

        finished_querying = False
        old_answers = None
        delta_score = float('inf')

        iteration = 0

        while (delta_score > self.epsilon or not finished_querying) and iteration < self.max_iterations:
            iteration+=1
            print(f'Current EM Iteration: {iteration}')

            if not finished_querying:
                # TODO: do we have to query for all answers again (just because of tau?)
                # TODO: query_tables_ids = query_tables_ids.add(tableJoiner(query_answers_set, Q, query_tables_ids))
                # TODO: somehow regulate the quering (really dirty tables will result in score = 0, so the query value score of none will be 1)
                # TODO: -> this is because of uncoveredX (or bad) will explode (examples with score of 1)
                # TODO: could take just top k answers (procentual?) or a prob threshold or just query for new answers if score of none is high (overall?)
                # TODO: play around with hyperparameters (also table prior) -> dirtier tables = lower score
                finished_querying = True

                if not tables:
                    query_values = [q_list + x_list for q_list, x_list in zip(Q, X)]
                    x_values = X
                    y_values = Y
                else:
                    query_values = Q
                    x_cols = [x for x, _ in answers.keys()]
                    y_cols = [y for _, y in answers.keys()]
                    x_values = [list(row) for row in zip(*x_cols)]
                    y_values = [list(row) for row in zip(*y_cols)]

                for table_id, answer_list in self.query_engine.find_answers(x_values, y_values, query_values):
                    # TODO: what about None values etc.???
                    for answer in answer_list:
                        if None in answer[1]:
                            continue
                        y_term = tuple(answer[2])
                        answer = (tuple(answer[0]), tuple(answer[1]))
                        if answer not in answers:
                            finished_querying = False
                            answers[answer] = {"score": 0.0, "tables": set(), "term": y_term}

                        tables.setdefault(table_id, {"score": 0.0, "answers": set()})

                        answers[answer]["tables"].add(table_id)
                        tables[table_id]["answers"].add(answer)

            tables = self.update_table_scores(answers, tables)
            answers = self.update_answer_scores(answers, tables, Q)

            if finished_querying and old_answers:
                delta_score = sum(
                    abs(answers[a]["score"] - old_answers[a]["score"])
                    for a in answers
                )

            old_answers = deepcopy(answers)

            # TODO: outside the loop (just for testing)
            result = {
                x: [(info["term"], info["score"]) for (x2, _), info in answers.items() if x2 == x]
                for x in zip(*Q)
            }

            print(result)

        # x Values are still tokenized (need to be matches with untokenized Q afterwards)
        # TODO: there is not 1:1 matching between tokenized and term (how to handle it?)
        return result

    # TODO: work with class variables for answers and tables?

    def update_table_scores(self, answers, tables):

        x_best_score = {}
        for (x, _), info in answers.items():
            score = info["score"]
            if x not in x_best_score or score > x_best_score[x]:
                x_best_score[x] = score
        
        for table_id in tables:
            good = 0
            bad = 0
            total = 0  # what is it used for?

            covered_query_answers_x = set()

            for table_answer in tables[table_id]["answers"]:
                x, _ = table_answer
                covered_query_answers_x.add(x)

                answer_score = answers[table_answer]["score"]

                if answer_score == x_best_score[x]:
                    good += answer_score
                else:
                    bad += 1

            # print(covered_query_answers_x)
            unseen_x = 0.0
            for x, best_score in x_best_score.items():
                if x not in covered_query_answers_x:
                    unseen_x += best_score

            # TODO: get it from the tables_table vertica (or from config for all)
            table_prior = 0.5

            table_score = self.alpha * ((table_prior * good) / (table_prior * good + (1-table_prior) * (bad + unseen_x)))

            # remove
            # print(f'Score of table {table_id}: {table_score}')

            tables[table_id]["score"] = table_score

        print(tables)

        return tables

    def update_answer_scores(self, answers, tables, Q):

        # TODO: Really set everything to 1 or just the new ones as below???
        for _, info in answers.items():
            info["score"] = 1.0

        for x_q in zip(*Q):
            x_answers = {(x, y) for (x, y) in answers if x == x_q}

            tables_for_x = {
                table_id
                for (x, _), info in answers.items()
                if x == x_q
                for table_id in info["tables"]
            }

            score_of_none = 1

            for table_id in tables_for_x:
                score_of_none *= (1-tables[table_id]["score"])
                for answer in x_answers:
                    # TODO: or just set 1 to the newly initialized?
                    #if answers[answer]["score"] == 0:
                    #    answers[answer]["score"] = 1

                    if answer in tables[table_id]["answers"]:
                        answers[answer]["score"] *= tables[table_id]["score"]
                    else:
                        answers[answer]["score"] *= (1-(tables[table_id]["score"]))

            denominator = score_of_none + sum(
                (answers[answer]["score"] for answer in x_answers)
            )
            
            for answer in x_answers:
                answers[answer]["score"] /= denominator
            
            print(f'Score of None for {x_q}: {score_of_none/denominator}')

        return answers