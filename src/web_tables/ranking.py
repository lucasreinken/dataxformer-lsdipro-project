import numpy as np

class WebTableRanker:
    def __init__(self, config):
        self.config = config

    def score(self, candidates, query_context):
        return

    def rank(self, candidates, query_context):
        scores = self.score(candidates, query_context)
        return


# work with getter and setter and global / environment variables

def update_table_scores(query_answers, query_tables_ids, tables_df, answer_scores, table_scores):
    alpha = 0.99
    
    for query_table_id in query_tables_ids:
        good = 0
        bad = 0
        covered_query_answers_x = set()

        table_answers = set(zip(
                query_answers.loc[query_answers['table_id'] == query_table_id, 'answer_x'],
                query_answers.loc[query_answers['table_id'] == query_table_id, 'answer_y']
                ))

        for table_answer in table_answers:
            covered_query_answers_x.add(table_answer[0])
            table_answer_score = answer_scores[table_answer]
            if (table_answer_score == max([score for (x, _), score in answer_scores.items() if x == table_answer[0]])):
                good += table_answer_score
            else:
                bad += 1

        unseen_x = 0
        for query_answer in zip(query_answers['answer_x'], query_answers['answer_y']):
            if not query_answer[0] in covered_query_answers_x:
                unseen_x += max([score for answer, score in answer_scores.items() if answer == query_answer])

        table_prior = tables_df.loc[tables_df['id'] == query_table_id, 'prior'].iloc[0]

        table_score = alpha * ((table_prior * good) / (table_prior * good + (1-table_prior) * (bad + unseen_x)))
        table_scores[query_table_id] = table_score

    return table_scores


def update_answer_scores(query_answers, query_answers_set, answer_scores, table_scores):

    query_answer_xs = {x for x, _ in query_answers_set}
    
    for query_answers_x in query_answer_xs:
        
        x_answers = {answer for answer in query_answers_set if answer[0] == query_answers_x}

        score_of_none = 1
        
        for table_id in query_answers.loc[query_answers['answer_x'] == query_answers_x, 'table_id']:
            score_of_none *= (1-table_scores[table_id])
            for x_answer in x_answers:
                answer_score = 1

                if x_answer in set(zip(
                    query_answers.loc[query_answers['table_id'] == table_id, 'answer_x'],
                    query_answers.loc[query_answers['table_id'] == table_id, 'answer_y']
                )):
                    answer_score *= table_scores[table_id]
                else:
                    answer_score *= (1-table_scores[table_id])

        for x_answer in x_answers:
            answer_scores[x_answer] = answer_scores[x_answer] / (score_of_none + np.sum(np.array([score for answer, score in answer_scores if answer in x_answers])))

    return answer_scores
            

def getAnswers(Querries: set, tables:dict, table_list:list) -> dict:

    """
    Returns a Dict of all the Answers to all Querries for all relevant Tables of a Search 
    In: 
        Querries: cleaned set of Querries for wich we need the Answers
        Tables: Dict key: Table_ID -> (Row_ID, (ColX_ID, ColY_ID)) of all the Relevent Tables
        Table_List: A List of all Tables 
    
    Out: 
        Answers: A Dict of Answers Key: x -> Value: Set(y_1, ..., y_n)
        If len(answers[x])==1, no contradictions exist 
    """
    answers = dict()
    for table_id, tup in tables.items(): 
        row_id, col_tup = tup
        x_col_id, y_col_id = col_tup 
        table = table_list.get(table_id)
        for row in table: 
            x_in_question = row[x_col_id]
            for x in Querries: 
                if x == x_in_question: 
                    # answers.setdefault(x, set()).add(cleaner(row[y_col_id])) 
                    continue
    
    return answers