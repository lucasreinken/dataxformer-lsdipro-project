import random
import pandas as pd
import numpy as np
import wandb
from .logger import LoggingContext
from src.web_tables.indexing import WebTableIndexer
from src.web_tables.ranking import WebTableRanker
from src.config import (
    get_default_indexing_config,
    get_default_vertica_config,
    get_default_ranking_config,
    get_default_testing_config,
)

from time import perf_counter


def get_ex(examples: pd.DataFrame, percents: list) -> list:
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


def eval_exercise(
    exercise: pd.DataFrame,
    x_count: int,
    count_examples: int,
    ranker,
    indexer,
    repeats: int = 5,
    k: int = 5,
    return_time: bool = False,
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

    cleaned_cols = list()
    for col_name in exercise.columns:
        cleaned_cols.append(indexer.tokenize_list(exercise[col_name].tolist()))

    x_cleaned_cols = cleaned_cols[:x_count]
    y_cleaned_cols = cleaned_cols[x_count:]

    all_indices = list(range(len(exercise)))
    list_of_metrics = list()

    for repetition in range(repeats):
        example_indices = random.sample(all_indices, count_examples)
        query_indices = [i for i in all_indices if i not in example_indices]

        ex_x = [[col[i] for i in example_indices] for col in x_cleaned_cols]
        ex_y = [[col[i] for i in example_indices] for col in y_cleaned_cols]

        queries = [[col[i] for i in query_indices] for col in x_cleaned_cols]
        query_keys = list(zip(*queries))
        y_true = list(zip(*[[col[i] for i in query_indices] for col in y_cleaned_cols]))

        if len(x_cleaned_cols) == 1:
            query_keys = [val[0] for val in query_keys]
        if len(y_cleaned_cols) == 1:
            y_true = [val[0] for val in y_true]

        if return_time:
            starttime = endtime = perf_counter()

        results = ranker.expectation_maximization(ex_x, ex_y, queries)

        if return_time:
            endtime = perf_counter()
            calc_time = endtime - starttime
            run_eval = evaluate_results(results, query_keys, y_true, k, calc_time)
        else:
            run_eval = evaluate_results(results, query_keys, y_true, k)

        list_of_metrics.append(run_eval)
        print(run_eval)

    metrics = dict()
    for key in list_of_metrics[0].keys():
        valid_values = [m[key] for m in list_of_metrics if m.get(key) is not None]
        if valid_values:
            metrics[key] = np.mean(valid_values)

    # for key in list(metrics.keys()):
    #     metrics[f"{key}_std"] = np.std([metric[key] for metric in list_of_metrics])

    return metrics


def run_single_exercise(
    exercise_tuple,
    indexing_config,
    ranking_config,
    vertica_config,
    count_examples,
    repeats: int,
    k: int,
    tau: int,
    return_time: bool = False,
):
    name, exercise, x_count = exercise_tuple

    indexer = WebTableIndexer(indexing_config)
    ranker = WebTableRanker(ranking_config, vertica_config, tau)

    try:
        print(f"Executing Exercise {name}")
        metrics = eval_exercise(
            exercise,
            x_count,
            count_examples,
            ranker,
            indexer,
            repeats,
            k,
            return_time,
        )
        return name, metrics
    finally:
        ranker.close()

    return name, metrics


def normalize(val):
    if isinstance(val, (tuple, list)) and len(val) == 1:
        return normalize(val[0])

    if isinstance(val, (tuple, list)):
        return tuple(normalize(v) for v in val)

    if isinstance(val, str):
        return val.strip().lower()

    return val


def evaluate_results(
    results: dict,
    xq: list,
    yq: list,
    k: int = 5,
    calc_time: float | None = None,
) -> dict:
    assert len(xq) == len(yq)

    tp = 0
    answered = 0
    topk_found = 0
    topall_found = 0
    total_queries = len(xq)

    for idx, x in enumerate(xq):
        lookup_key = x if isinstance(x, tuple) else (x,)
        result_list = results.get(lookup_key, list())

        if not result_list:
            continue

        answered += 1
        sorted_results = sorted(result_list, key=lambda tup: tup[1], reverse=True)

        correct_y = normalize(yq[idx])
        best_pred = normalize(sorted_results[0][0])

        if correct_y == best_pred:
            tp += 1
            topk_found += 1
            topall_found += 1
        else:
            preds_k = [normalize(res[0]) for res in sorted_results[:k]]
            if correct_y in preds_k:
                topk_found += 1
                topall_found += 1
            else:
                all_preds = [normalize(res[0]) for res in sorted_results]
                if correct_y in all_preds:
                    topall_found += 1

    precision = tp / answered if answered > 0 else 0.0  ###tp / (tp + fp)
    recall = tp / total_queries if total_queries > 0 else 0.0  ###tp / (tp + fn)

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    metric_dict = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "topk_acc": topk_found / total_queries,
        "topall_acc": topall_found / total_queries,
        "answered_rate": answered / total_queries,
    }

    if calc_time is not None:
        metric_dict["calc_time"] = calc_time

    return metric_dict


def full_loop(
    exercises: list,
    count_examples: int = 5,
    repeats: int = 5,
    k: int = 1,
    tau: int = 2,
    max_workers: int = 4,
    return_time: bool = False,
) -> None:
    indexing_config = get_default_indexing_config()
    ranking_config = get_default_ranking_config()
    vertica_config = get_default_vertica_config()
    testing_config = get_default_testing_config()
    config_list = [indexing_config, ranking_config, vertica_config, testing_config]

    with LoggingContext(
        config_list,
        project_name=testing_config.project_name,
        entity=testing_config.entity,
    ) as ctx:
        try:
            num_ex = len(exercises)

            for ex_num, exercise in enumerate(exercises, 1):
                name, metrics = run_single_exercise(
                    exercise,
                    indexing_config,
                    ranking_config,
                    vertica_config,
                    count_examples,
                    repeats,
                    k,
                    tau,
                    return_time,
                )
                ctx.add_eval_result(name, metrics)

                title = name.replace("_", " & ").replace("2", " → ")
                info_msg = f"Finished EX {ex_num}/{num_ex} {title}: precision {metrics['precision']:.2f}, recall {metrics['recall']:.2f}"

                if return_time and "calc_time" in metrics:
                    info_msg += f", calc_time: {metrics['calc_time']:.2f}"

                ctx.info(info_msg)

        except KeyboardInterrupt as exc:
            ctx.warning(
                "Process was interrupted manually (ctr + c). Process is shutting down!"
            )
            wandb.finish(exit_code=1)
            raise exc

            ####Metrics zeigen, dass wir die Tables schneller löschen müssten um RAM freizugeben.
            ###Blocking in EM
            ###Maybe this a as a tool to an agent.
