import itertools

from src.web_tables.indexing import WebTableIndexer

class WebTableQueryEngine:
    def __init__(self, index):
        self.index = index

    def indexing(self, tokenized_value:list, projections:dict, key_id:int = None)->dict:
        """
        Return a dict of all the Examples found in the projections
        In: 
            Cleaned_Values: A List of all the Stemped Versions of one Example given
            Projections: A Dict of Projections of all given Tables
        Out: 
            Index_Dict: A dict of all the positions where the Example was found
                        Form: Key: (Table_ID, Row_ID) -> Value: (Col_ID)
        """ 
        key_id = key_id+1
        index_dict = dict() 
        
        value_index = projections.get(tokenized_value, None)

        if value_index: 
            for table_id, row_id, col_id in value_index: 

                if key_id: 
                    index_dict.setdefault((table_id, row_id), set()).add((key_id-1, col_id))
                else: 
                    index_dict.setdefault((table_id, row_id), set()).add((col_id))
        
        return index_dict
    

    # TODO: Adapt it to be able to handle multiple Xs
    def find_direct_tables(self,
                           # TODO: WHY IS EVERYTHING A SET (REALLY NEEDED???)
                            Examples:set,
                            # TODO: ADD TO THE CONFIG FILE (AS SELF VARIABLE)
                            tau: int,
                            indexer:WebTableIndexer):

        evidence = dict()
        E = len(Examples)
        K = None

        if tau > E: 
            raise ValueError(f'At least Tau: {tau} examples must be given!')
        
        for Example in Examples: 

            if not K: 
                K = len(Example)                
            else: 
                if K != len(Example): 
                    raise ValueError(f'All Examples must be of the same Size!')
            
            idx_list = list()               
            for key_id, example_key in enumerate(Example):               
                tokens_of_example_key = indexer.tokenize(example_key)

                dict_of_idx = self.indexing(tokens_of_example_key, projections, key_id)  
                idx_list.append(dict_of_idx)

            key_sets = [set(d.keys()) for d in idx_list]
            unique_shared_keys = set.intersection(*key_sets)
            
            if not unique_shared_keys: 
                continue 

            for key in unique_shared_keys:

                all_mappings_for_this_row = []
                for dict_of_idx in idx_list:

                    mappings = dict_of_idx[key] 
                    all_mappings_for_this_row.append(mappings)

                
                for candidate_mapping in itertools.product(*all_mappings_for_this_row):

                    table_id, row_id = key

                    evidence.setdefault((table_id, candidate_mapping), set()).add(Example)

        
        relevant_tables = dict()

        for key, value in evidence.items(): 
            if len(value) >= tau: 
                table_id, candidate_mapping = key
                relevant_tables.setdefault(table_id, list()).append(dict(candidate_mapping))

        return relevant_tables

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