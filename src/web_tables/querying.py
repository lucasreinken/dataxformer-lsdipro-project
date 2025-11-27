class WebTableQueryEngine:
    def __init__(self, index):
        self.index = index

    def query(self, query_spec):
        return


def cleaner(value, create_ngrams:bool = False, ngram_size:int = 2): 
    # sentence = value.lower().translate(str.maketrans("", "", string.punctuation))
    # words = word_tokenize(sentence)
    # stemmed_words = []
    # for word in words: 
        # stemmed_word = ps.stem(word)
    #     stemmed_words.append(stemmed_word)

    # if create_ngrams: 
    #     created_ngrams = ngrams(stemmed_words, ngram_size)
    #     stemmed_words.extend(list(created_ngrams))
    # return stemmed_words
    return

# query_answers format: dataframe(answer_x, answer_y, table_id)

def expectation_maximization(query_answers, epsilon, tables_df):
    query_tables_ids = set(query_answers['table_id'])
    query_answers_set = set(zip(query_answers['answer_x'], query_answers['answer_y']))

    # which values for initilization?
    answer_scores = {answer: 1.0 for answer in query_answers_set}
    table_scores = {table_id: 1.0 for table_id in query_tables_ids}

    # delta_score = np.inf

    old_answer_scores = dict()

    # while delta_score > epsilon:

        # add line 6-15?

        # old_answer_scores = deepcopy(answer_scores)

        # table_scores = update_table_scores(query_answers, query_tables_ids, tables_df, answer_scores, table_scores)
        # answer_scores = update_answer_scores(query_answers, query_answers_set, answer_scores, table_scores)

        # delta_score = np.sum(np.abs(
        #         np.array([answer_scores[answer] for answer in query_answers_set]) -
        #         np.array([old_answer_scores[answer] for answer in query_answers_set])
        #     ))

    print(table_scores)

    return answer_scores