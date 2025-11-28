class WebTableJoiner:
    def __init__(self):
        return


    def join(self, tables, join_spec):
        return


def find_direct_tables(Examples:set,  
                       projections:dict, 
                       tau: int, 
                       # tokenizer:Tokenizer
                       ):

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
            # tokens_of_example_key = tokenizer(example_key)

            # dict_of_idx = indexing(tokens_of_example_key, projections, key_id)  
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

            
            # for candidate_mapping in itertools.product(*all_mappings_for_this_row):

            #     table_id, row_id = key

            #     evidence.setdefault((table_id, candidate_mapping), set()).add(Example)

    
    relevant_tables = dict()

    for key, value in evidence.items(): 
        if len(value) >= tau: 
            table_id, candidate_mapping = key
            relevant_tables.setdefault(table_id, list()).append(dict(candidate_mapping))

    return relevant_tables