import os 
import sys
import random 
import pandas as pd 
import numpy as np
from logger import LoggingContext, timeit
from src.web_tables.indexing import WebTableIndexer
from src.web_tables.ranking import WebTableRanker
from src.config import (
    get_default_indexing_config,
    get_default_vertica_config,
    get_default_ranking_config
)

###Alles hier muss in die Config! 
tau = [1, 2, 3, 4, 5]

percents = [0.25, 0.5, 0.75]

nums_of_evals = 5 




def get_ex(examples: pd.DataFrame, percents:list) -> list: 
    """
    Returns the amount of unmasked examples for a given example df as a list, based on the given percents list. 

    Args: 
        examples: pd.DataFrame
        percents: list 
    
    Returns: 
        count_of_unmasked_examples: list 
    """
    _, rows = examples.shape 
    count_of_unmasked_examples = [round(percent * rows) for percent in percents]
    return count_of_unmasked_examples


# result = {
#                 x: [(info["term"], info["score"]) for (x2, _), info in answers.items() if x2 == x]
#                 for x in zip(*Q)
#             }

# def evaluate(results: dict, xq:list, yq:list, k:int = 5) -> dict:
#     """
#     Evaluates the precision, recall, topk_acc and topall_acc for a given exercise. 

#     Args: 
#         results: dict 
#         XQ: list 
#         YQ: list  
#         k: int = 5 
    
#     Returns: 
#         evaluation: dict 
#     """

#     assert(len(xq) == len(yq))

#     top1 = 0
#     topk = 0
#     topall = 0
#     count_over_threshhold = 0 
#     values_per_exercise = len(xq)

#     for idx, x in enumerate(xq): 
#         result_list = results.get(x)
#         count_over_threshhold += 1 if len(result_list) > 0 else 0 
#         sorted_results = sorted(result_list, key=lambda tup: tup[1], reverse = True )

#         if yq[idx] == sorted_results[0][0]:
#             top1 += 1 
#             topk += 1 
#             topall += 1
#         elif yq[idx] in [val[0] for val in sorted_results[:k]]: 
#             topk +=1
#             topall +=1 
#         elif yq[idx] in [val[0] for val in sorted_results[k:]]: 
#             topall +=1
            
#     top1_acc = recall = top1 / values_per_exercise
#     topk_acc = topk / values_per_exercise 
#     topall_acc = topall / values_per_exercise 
#     precision = top1 / count_over_threshhold 


#     evaluation = {"precision": precision,
#                     "recall": recall,
#                     "topk_acc": topk_acc, 
#                     "topall_acc": topall_acc, 
#                     }
#     return evaluation




def eval_exercise(exercise: pd.DataFrame, 
              x_count: int, 
              count_examples: int, 
              ranker, 
              indexer, 
              repeats:int = 5, 
              k:int = 5, 
              ) -> dict: 
    """
    Evaluation loop for one exercise. 

    Args: 
        exercise: pd.DataFrame 
        x_count: int
        count_examples: int 
        ranker
        indexer
        repeats: int = 5 
        k: int = 5

    Returns: 
        metrics: dict 
    """

    cleaned_cols = []
    for col_name in exercise.columns:
        cleaned_cols.append(indexer.tokenize_list(exercise[col_name].tolist()))

    x_cleaned_cols = cleaned_cols[:x_count]
    y_cleaned_cols = cleaned_cols[x_count:]

    all_indices = list(range(len(exercise)))
    list_of_metrics = list()

    for repitition in range(repeats): 
        example_indices = random.sample(all_indices, count_examples)
        query_indices = [i for i in all_indices if i not in example_indices]

        ex_x = [[col[i] for i in example_indices] for col in x_cleaned_cols]
        ex_y = [[col[i] for i in example_indices] for col in y_cleaned_cols]
        
        querries = [[col[i] for i in query_indices] for col in x_cleaned_cols]
        querry_keys = list(zip(*querries))
        y_true = list(zip(*[[col[i] for i in query_indices] for col in y_cleaned_cols]))

        if len(x_cleaned_cols) == 1:
            querry_keys = [val[0] for val in querry_keys]
        if len(y_cleaned_cols) == 1:
            y_true = [val[0] for val in y_true]

        results = ranker.expectation_maximization(ex_x, ex_y, querries)
        
        run_eval = evaluate_results(results, querry_keys, y_true, k)
        list_of_metrics.append(run_eval)
    
    metrics = {
        key: np.mean([metric[key] for metric in list_of_metrics]) 
        for key in list_of_metrics[0].keys()
    }

    # for key in list(metrics.keys()):
    #     metrics[f"{key}_std"] = np.std([metric[key] for metric in list_of_metrics])

    return metrics


def evaluate_results(results: dict, xq:list, yq:list, k:int = 5) -> dict:
    """
    Evaluates the precision, recall, topk_acc and topall_acc for a given exercise. 

    Args: 
        results: dict 
        XQ: list 
        YQ: list  
        k: int = 5 
    
    Returns: 
        evaluation: dict 
    """

    assert(len(xq) == len(yq))

    top1 = 0
    topk = 0
    topall = 0
    answered = 0 
    values_per_exercise = len(xq)

    for idx, x in enumerate(xq): 
        result_list = results.get(x, [])
        if not result_list:
            continue

        answered += 1  
        sorted_results = sorted(result_list, key=lambda tup: tup[1], reverse = True )

        correct_y = yq[idx]

        if correct_y == sorted_results[0][0]:
            top1 += 1
            topk += 1
            topall += 1
        else:
            preds_k = [res[0] for res in sorted_results[:k]]
            if correct_y in preds_k:
                topk += 1
                topall += 1
            else:
                all_preds = [res[0] for res in sorted_results]
                if correct_y in all_preds:
                    topall += 1
            
    top1_acc = recall = top1 / values_per_exercise
    topk_acc = topk / values_per_exercise 
    topall_acc = topall / values_per_exercise 
    precision = top1 / answered if answered > 0 else 0.0


    evaluation = {"precision": precision,
                    "recall": recall,
                    "topk_acc": topk_acc, 
                    "topall_acc": topall_acc, 
                    }
    return evaluation


def full_loop(exercises: list, 
              prior, 
              epsilon,
              max_iterations, 
              count_examples:int, 
              repeats: int = 5, 
              k:int = 5, 
              tau: int = 2, 
              ) -> None: 
    
    indexing_config = get_default_indexing_config()
    ranking_config = get_default_ranking_config()
    vertica_config = get_default_vertica_config()
    config_list = [indexing_config, ranking_config, vertica_config]

    indexer = WebTableIndexer(indexing_config)
    
    ranking_config.table_prior = prior
    ranking_config.epsilon = epsilon
    ranking_config.max_iterations = max_iterations
    ranker = WebTableRanker(ranking_config, tau)

    with LoggingContext(config_list, 
                        project_name = "DataXFormerTest", 
                        entity= None) as ctx: 

        list_of_exercise_metrics = list()
        for exercise, x_count in exercises: 
            exercise_metrics = eval_exercise(exercise, x_count, count_examples, ranker, indexer, repeats, k)
            list_of_exercise_metrics.append(exercise_metrics)

            
        full_metrics = {
            key: np.mean([metric[key] for metric in list_of_exercise_metrics]) 
            for key in list_of_exercise_metrics[0].keys()
        }

        ctx.log(full_metrics)



