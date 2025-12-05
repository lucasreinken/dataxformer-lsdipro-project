from copy import deepcopy
from src.database.vertica_client import VerticaClient
from src.config import get_default_vertica_config

class WebTableRanker:
    def __init__(self, config):

        self.epsilon = config.epsilon
        self.alpha = config.alpha

        vertica_config = get_default_vertica_config()
        self.vertica_client = VerticaClient(vertica_config)

    def score(self, candidates, query_context):
        return

    def rank(self, candidates, query_context):
        scores = self.score(candidates, query_context)
        return


    # TODO: put the quering part in quering, so the vertica client is just initialized there
    # TODO: adapt it to handle multi-column
    def expectation_maximization(self, X: list[str], Y: list[str], tau: int, Q: list[str]):
        
        # TODO: store them more lightweight (e.g. decouple score)
        # answers: (x,y) → {score, tables={t1,t2,…}}
        # tables:  t_id → {score, answers={(x1,y1),(x2,y2)…}}

        index_set = set()

        answers = {
            (x, y): {"score": 1.0, "tables": set()}
            for x, y in zip(X, Y)
        }

        # TODO: add Jaccard Similiratity (if labels provided)
        tables = {}
        z = self.vertica_client.find_xy_candidates(X, Y, tau)
        erg = self.vertica_client.row_validation(z, X, Y, tau)

        for index in erg:
            for answer in self.vertica_client.get_columns(index):
                answer = tuple(answer)
                if answer in answers.keys():
                    table_id = index[0]
                    tables.setdefault(table_id, {"score": 0.0, "answers": set()})
                    answers[answer]["tables"].add(table_id)
                    tables[table_id]["answers"].add(answer)

        old_answers = None
        
        delta_score = float('inf')

        finished_querying = False

        while delta_score > self.epsilon or not finished_querying:
            if not finished_querying:
                # TODO: wrapper function for find_xy and validate row (together with the answers)
                # TODO: do we have to query for all answers again (just because of tau?)
                if not erg:
                    X, Y = zip(*answers.keys())
                    X, Y = list(X), list(Y)
                    z = self.vertica_client.find_xy_candidates(X, Y, tau)
                    erg = self.vertica_client.row_validation(z, X, Y, tau)
                # TODO: always work with tuples
                index_set.update([tuple(x) for x in erg])
                erg = None
                # TODO: query_tables_ids = query_tables_ids.add(tableJoiner(query_answers_set, Q, query_tables_ids))
                finished_querying = True
                for index in index_set:
                    for (x, y) in self.vertica_client.get_columns(index):
                        if x in Q:
                            # TODO: what about None values etc.???
                            answer = (x, y)
                            table_id = index[0]
                            if answer not in answers:
                                finished_querying = False

                                answers[answer] = {"score": 0.0, "tables": set()}
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
                x: [(y, info["score"]) for (x2, y), info in answers.items() if x2 == x]
                for x in Q
            }

            # remove
            print(result)

        return result

    # TODO: work with class variables for answers and tables?

    def update_table_scores(self, answers, tables):

        x_best_score = {}
        for (x, _), info in answers.items():
            score = info["score"]
            if x not in x_best_score or score > x_best_score[x]:
                x_best_score[x] = score

        print(x_best_score)
        
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
                    # print(f'Not covered by the table = {x}: {best_score}')
                    unseen_x += best_score

            # TODO: get it from the tables_table vertica (or from config for all)
            table_prior = 0.5

            table_score = self.alpha * ((table_prior * good) / (table_prior * good + (1-table_prior) * (bad + unseen_x)))

            # remove
            # print(f'Score of table {table_id}: {table_score}')

            tables[table_id]["score"] = table_score

        return tables


    def update_answer_scores(self, answers, tables, Q):

        # TODO: Really set everything to 1 or just the new ones as below???
        for _, info in answers.items():
            info["score"] = 1.0

        for x_q in Q:
            
            x_answers = {(x, y) for (x, y) in answers if x == x_q}

            score_of_none = 1

            tables_for_x = {
                table_id
                for (x, _), info in answers.items()
                if x == x_q
                for table_id in info["tables"]
            }
            
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
