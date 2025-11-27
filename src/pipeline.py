from .web_tables import (
    WebTableIndexer,
    WebTableQueryEngine,
    WebTableJoiner,
    WebTableRanker,
)



import pandas as pd
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
import collections
import nltk 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk import ngrams
import copy
from collections import defaultdict
from functools import cache
import cProfile
import pstats
import time
from tabulate import tabulate
import itertools

from src.config import get_default_config

cfg = get_default_config()

class DataXFormerPipeline:
    def __init__(self, config):
        self.indexer = WebTableIndexer(config.indexing)
        self.query_engine = WebTableQueryEngine(...)
        self.joiner = WebTableJoiner()
        self.ranker = WebTableRanker(config.ranking)

    def run(self, query):
        return

def main():
    data = pd.read_json('/Users/christophhalberstadt/Documents/TU Berlin/LSDIPro/tables.json', 
                    lines=True, nrows=100, )
    stemmer = PorterStemmer()
    # my_tokenizer = Tokenizer(stemmer)
    # table_list = create_table_list(data)

    # projections = create_projections(table_list, my_tokenizer)

    indexing_example = [('1929', 'Robert Crawford'), ('1938', 'John Patrick')]
    tau = 2

    # relevant_tables = find_direct_tables(indexing_example, projections, tau, my_tokenizer)

    indexing_example = [( 'Robert Crawford' ,'Ulster Unionist', '1929')]
    tau = 1

    # relevant_tables = find_direct_tables(indexing_example, projections, tau, my_tokenizer)

    Querries = [('Robert Crawford', 'Ulster Unionist'), ('John Patrick', 'Ulster Unionist')]

    # table = table_list[0]
    my_list = list()
    # for row in table: 
    #     my_list.append(my_tokenizer(row[2]))

    print(my_list)

    my_list = np.array(my_list)

    d = my_tokenizer(Querries[0][0]) 
    print(d)
    print(type(d))

    c = np.where(d == my_list, 1, 0)
    print(c)


    relevant_tables = None
    table_list = None
    my_tokenizer = None



    anz_querries = len(Querries)
    cols = list()
    for relevant_table, possible_mappings in relevant_tables.items(): 
        table = table_list[relevant_table]
        for possible_mapping in possible_mappings: 
            for Ev_Col, Tab_col in possible_mapping.items(): 
                #Ev_Col, Tab_col = mapping.items()
                tokenized_querries = []
                print(Ev_Col, Tab_col)
                print(type(Ev_Col))
                col = list()
                for row in table: 
                    col.append(row[Tab_col])

                np_col = np.array(col)
                for i in range(anz_querries):
                    x = Querries[i]
                    print(x)
                    print(Ev_Col)
                    print(x[Ev_Col])
                    tokenized_x = my_tokenizer(x[Ev_Col])
                    index = np.where(tokenized_x ==np_col)[0]
                    print(index)
                
                cols.append(col)
    for col in cols: 
        print(col)
    

    # TODO test files / connection to vertica etc.

if __name__ == "__main__":
    main()
