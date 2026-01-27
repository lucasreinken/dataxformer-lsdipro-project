import random
import pandas as pd
import numpy as np
import wandb
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from .logger import LoggingContext
from src.web_tables.indexing import WebTableIndexer
from src.web_tables.ranking import WebTableRanker


def get_number_of_examples(exercise: pd.DataFrame, percents: list) -> list:
    """
    Returns the amount of unmasked examples for a given exercise df as a list, based on the given percents list.

    Args:
        exercise: pd.DataFrame
        percents: list

    Returns:
        count_of_unmasked_examples: list
    """
    _, rows = exercise.shape
    count_of_unmasked_examples = [round(percent * rows) for percent in percents]
    return count_of_unmasked_examples


def eval_exercise(
    exercise: pd.DataFrame,
    x_count: int,
    count_of_examples: int,
    ranker: WebTableRanker,
    indexer: WebTableIndexer | None,
    repeats: int,
    k: int,
    return_time: bool,
    print_evaluation: bool = False,
) -> dict:
    """
    Evaluation loop for one exercise.

    Args:
        exercise: pd.DataFrame
        x_count: int
        count_of_examples: int
        ranker: WebTableRanker
        indexer: WebTableIndexer | None
        repeats: int
        k: int
        return_time: bool
        print_evaluation: bool (Default: False)

    Returns:
        metrics: dict
    """

    cleaned_cols = list()
    for col_name in exercise.columns:
        if not indexer:
            cleaned_cols.append(
                [val.strip('"') for val in exercise[col_name].astype(str)]
            )
        else:
            cleaned_cols.append(indexer.tokenize_list(exercise[col_name].tolist()))

    x_cleaned_cols = cleaned_cols[:x_count]
    y_cleaned_cols = cleaned_cols[x_count:]

    all_indices = list(range(len(exercise)))
    list_of_metrics = list()

    for repetition in range(repeats):
        example_indices = random.sample(all_indices, count_of_examples)
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

        if print_evaluation:
            print(run_eval)

    metrics = dict()
    for key in list_of_metrics[0].keys():
        valid_values = [m[key] for m in list_of_metrics if m.get(key) is not None]
        if valid_values:
            metrics[key] = np.mean(valid_values)

    return metrics


def run_single_exercise(
    exercise_tuple: tuple,
    config,
):
    """
    A task that is performed in a distributed setting.

    Args:
        exercise_tuple: tuple
        config: MasterConfig

    Returns:
        name: str
        metrics: dict
    """
    name, exercise, x_count = exercise_tuple

    indexer = None if config.experiment.preindexed else WebTableIndexer(config.indexing)
    ranker = WebTableRanker(config)

    try:
        print(f"Executing Exercise {name}")
        with ranker:
            metrics = eval_exercise(
                exercise=exercise,
                x_count=x_count,
                count_of_examples=config.experiment.count_of_examples,
                ranker=ranker,
                indexer=indexer,
                repeats=config.experiment.repeats,
                k=config.experiment.k,
                return_time=config.experiment.return_time,
            )
        return name, metrics
    except Exception as e:
        print(f"Error in Exercise {name}: {e}")
        raise e


def normalize(val):
    """
    Assures that the given value is formatted correctly in order to evaluate equality.

    Args:
        val: tuple | list | str | any

    Returns:
        normalized_val: tuple | str | any
    """
    if isinstance(val, (tuple, list)) and len(val) == 1:
        return normalize(val[0])

    if isinstance(val, (tuple, list)):
        return tuple(normalize(v) for v in val)

    if isinstance(val, str):
        return val.strip().lower()

    return val


def evaluate_results(
    results: dict,
    queries: list,
    ground_truth: list,
    k: int,
    calc_time: float | None = None,
) -> dict:
    """
    Evaluates the results of a single evaluation run.

    Args:
        results: dict
        queries: list
        ground_truth: list
        k: int
        calc_time: float | None (Default: None)

    Returns:
        metric_dict: dict
    """
    assert len(queries) == len(ground_truth)

    true_positive = 0
    answered = 0
    topk_found = 0
    topall_found = 0
    total_queries = len(queries)

    for idx, query in enumerate(queries):
        lookup_key = query if isinstance(query, tuple) else (query,)
        result_list = results.get(lookup_key, list())

        if not result_list:
            continue

        answered += 1
        sorted_results = sorted(result_list, key=lambda tup: tup[1], reverse=True)

        correct_y = normalize(ground_truth[idx])
        best_pred = normalize(sorted_results[0][0])

        if correct_y == best_pred:  ####Hier
            true_positive += 1
            topk_found += 1
            topall_found += 1
        else:
            preds_k = [normalize(result[0]) for result in sorted_results[:k]]
            if correct_y in preds_k:
                topk_found += 1
                topall_found += 1
            else:
                all_preds = [normalize(result[0]) for result in sorted_results]
                if correct_y in all_preds:
                    topall_found += 1

    precision = true_positive / answered if answered > 0 else 0.0  ###tp / (tp + fp)
    recall = (
        true_positive / total_queries if total_queries > 0 else 0.0
    )  ###tp / (tp + fn)

    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    metric_dict = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "topk_acc": topk_found / total_queries,
        "topall_acc": topall_found / total_queries,
        "answered_rate": answered / total_queries,
    }

    if calc_time is not None:
        metric_dict["calc_time"] = calc_time

    return metric_dict


def full_loop(
    exercises: list,
    config,
) -> None:
    """
    Performs the full evaluation loop over all exercises.

    Args:
        exercises: list
        config: MasterConfig

    Returns:
        None
    """
    config_list = [config.indexing, config.ranker, config.vertica, config.experiment]

    with LoggingContext(
        configs=config_list,
        project_name=config.experiment.project_name,
        entity=config.experiment.entity,
        # log_dir=. ###Hier angeben? Sonst konstuiert er eins automatisch.
        # seed=config.experiment.seed,
    ) as ctx:
        try:
            num_ex = len(exercises)

            with ProcessPoolExecutor(config.experiment.parallel_runs) as executor:
                futures = {
                    executor.submit(
                        run_single_exercise,
                        exercise,
                        config,
                    ): exercise
                    for exercise in exercises
                }

                for ex_num, future in enumerate(as_completed(futures), 1):
                    name, metrics = future.result()
                    print(metrics)

                    ctx.add_eval_result(name, metrics)

                    title = name.replace("_", " & ").replace("2", " → ")
                    info_msg = (
                        f"Finished EX {ex_num}/{num_ex} {title}: "
                        f"precision {metrics['precision']:.2f}, "
                        f"recall {metrics['recall']:.2f}"
                    )
                    print(info_msg)

                    if config.experiment.return_time and "calc_time" in metrics:
                        info_msg += f", calc_time: {metrics['calc_time']:.2f}"

                    ctx.info(info_msg)

        except KeyboardInterrupt:
            ctx.warning(
                "Process was interrupted manually (ctrl+c). Process is shutting down!"
            )
            wandb.finish(exit_code=1)
            raise

            ####Metrics zeigen, dass wir die Tables schneller löschen müssten um RAM freizugeben.
            ###Blocking in EM
            ###Maybe this a as a tool to an agent.
