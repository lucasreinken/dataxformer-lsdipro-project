import re
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from src.config import MultiHopConfig
from src.utils.fuzzy_matching import fuzzy_match


class DirectDependencyVerifier:
    def __init__(
        self, query_factory, multi_config: MultiHopConfig, tau: int, seed: int
    ) -> None:
        self.qf = query_factory
        self.conn = query_factory.conn
        self.config = multi_config

        self.tau = tau
        self.seed = seed

        self.max_path_len = self.config.max_path_len
        self.max_tables = self.config.max_tables
        self.adaptive_limits = self.config.adaptive_limits

        self.fd_threshold = self.config.fd_threshold
        self.error_threshold = self.config.error_threshold_for_fd
        self.use_parallel_fd = self.config.use_parallel_fd
        self.max_workers = self.config.max_workers_multi

        self.auto_detect_numeric = self.config.auto_detect_numeric
        self.restrict_nums = self.config.restrict_nums
        self.metric_precision = self.config.metric_precision
        self.none_precision = self.config.none_precision
        self.min_word_len = self.config.min_word_len
        self.strict_uniqueness_in_df = self.config.strict_uniqueness_in_df

        self.use_fuzzy_y_match = self.config.use_fuzzy_y_match
        self.fuzzy_scorer = self.config.fuzzy_scorer_multi
        self.fuzzy_threshold = self.config.fuzzy_threshold_multi

        self.large_df_threshold = self.config.large_df_threshold
        self.sample_size = self.config.sample_size
        self.max_paths_to_keep = self.config.max_paths_to_keep

        self.adaptive_reduction_factor = self.config.adaptive_reduction_factor
        self.overlap_threshold = self.config.overlap_threshold
        self.threshold_for_numeric_cols = self.config.threshold_for_numeric_cols

        self.print_further_information = self.config.print_further_information
        self.reset_counters = self.config.reset_counters

        self.previously_seen_tables = set()
        self.numeric_cache = dict()
        self.survived = defaultdict(int)

        pd.set_option("future.no_silent_downcasting", True)

    def update_seen(self, new_tables: list[tuple]) -> None:
        """
        Helper function to update the previously_seen_tables set.

        Args:
            new_tables: list[tuple]

        Retruns:
            None
        """

        new_table_ids = set(r[0] for r in new_tables)
        self.previously_seen_tables.update(new_table_ids)

    def load_table_batch(self, table_ids: list[int]) -> dict[int, pd.DataFrame]:
        """
        Loades a batch of tables based on their table_ids.

        Args:
            table_ids: list[int]

        Returns:
            dict[int, pd.DataFrame]
        """

        if not table_ids:
            return dict()

        if hasattr(self.qf, "get_table_contents_batch"):
            try:
                return self.qf.get_table_contents_batch(table_ids)
            except Exception as e:
                print(f"Batch loading failed, falling back to sequential: {e}")

        result = dict()
        for table_id in table_ids:
            try:
                df = self.qf.get_table_content(table_id, include_cols=None)
                if df is not None:
                    result[table_id] = df
            except Exception as e:
                print(f"Warning: Failed to load table {table_id}. Error: {e}")

        return result

    def is_numeric_like(self, val: str) -> bool:
        """
        Checks if a value is numeric in order to limit the search space.

        Args:
            val: str

        Returns:
            bool
        """
        cache_key = (type(val).__name__, str(val)[:50])
        if cache_key in self.numeric_cache:
            return self.numeric_cache[cache_key]

        s = str(val).strip()
        cleaned = s.replace(" ", "").replace(",", ".").replace("\xa0", "")

        try:
            float(cleaned)
            result = True
        except ValueError:
            # Dates
            if re.match(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$", s):
                result = True
            # Nums with separation
            elif re.match(r"^[\d\s,\.]+$", s) and any(c.isdigit() for c in s):
                result = True
            else:
                result = False

        self.numeric_cache[cache_key] = result
        return result

    def detect_numeric_columns(self, data_cols: list[list]) -> bool:
        """
        Helper func to detect if the elements of list of columns consists of numeric values above a certain threshhold.

        Args:
            data_cols: list[list]

        Returns:
            bool
        """

        for col in data_cols:
            valid_vals = [v for v in col if v is not None and str(v).strip()]
            if not valid_vals:
                continue

            numeric_count = sum(
                1 for val in valid_vals if self.is_numeric_like(val=val)
            )
            if numeric_count / len(valid_vals) > self.threshold_for_numeric_cols:
                return True

        return False

    def check_fd(
        self,
        x_cols: list,
        df: pd.DataFrame,
    ) -> tuple[list, list, pd.DataFrame] | None:
        """
        Checks for functional dependencies in a dataframe based on (a) given column(s).

        Args:
            x_cols: list
            df: pd.DataFrame

        Returns:
            tuple[list, list, pd.DataFrame] | None
        """

        if df is None or df.empty:
            return None

        df_clean = df.dropna(subset=x_cols)

        if df_clean.empty:
            return None

        n_unique_lhs = df_clean[x_cols].drop_duplicates().shape[0]

        if n_unique_lhs == 0:
            return None

        y_col_candidates = [c for c in df_clean.columns if c not in x_cols]

        if not y_col_candidates:
            return None

        nunique_dict = df_clean[y_col_candidates].nunique()

        valid_y_cols = list()
        threshold = n_unique_lhs * (1.0 - self.error_threshold)

        for y_col in y_col_candidates:
            n_unique_rhs = nunique_dict[y_col]

            if n_unique_rhs == 0:
                valid_y_cols.append(y_col)
                continue

            if n_unique_rhs >= threshold:
                valid_y_cols.append(y_col)

        if not valid_y_cols:
            return None

        filtered_df = df_clean[x_cols + valid_y_cols]

        return (x_cols, valid_y_cols, filtered_df)

    def check_single_fd(self, args: tuple) -> dict | None:
        """
        Defines a task to check for functional dependencies in a distributed setting.

        Args:
            args: tuple

        Returns:
            dict | None
        """
        table_id, x_cols, df = args
        fd_result = self.check_fd(x_cols=x_cols, df=df)
        if fd_result:
            x_cols_result, y_cols_result, filtered_df = fd_result
            return {
                "table_id": table_id,
                "x_cols": x_cols_result,
                "y_cols": y_cols_result,
                "df": filtered_df,
            }
        return None

    def in_loop_fd(
        self,
        merged_df: pd.DataFrame,
        left_join_cols: list,
        actual_z_col: str,
    ) -> tuple[int, int]:
        """
        A reduced FD check to be used inside the tableJoinerInnerLoop.

        Args:
            merged_df: pd.DataFrame
            left_join_cols: list
            actual_z_col: str

        Returns:
            tuple[int, int]
        """
        try:
            fd_check_df = merged_df[[*left_join_cols, actual_z_col]].dropna()
            if not fd_check_df.empty:
                fd_unique = fd_check_df.drop_duplicates()
                n_unique_lhs = fd_unique[left_join_cols].drop_duplicates().shape[0]
                n_unique_rhs = fd_unique[actual_z_col].nunique()
                return n_unique_lhs, n_unique_rhs
        except Exception as e:
            print(f"Warning: FD check failed: {e}")

        return 0, 0

    def reduce_result_df(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Reduces a result dataframe to only the necessary columns.

        Args:
            df: pd.DataFrame

        Returns:
            pd.DataFrame | None
        """
        try:
            x_cols = [c for c in df.columns if c.startswith("x_col_")]
            it_cols = sorted(
                [c for c in df.columns if c.startswith("it")],
                key=lambda x: int(x.replace("it", "")),
            )

            if not it_cols:
                return None

            final_it_col = it_cols[-1]

            cols_to_keep = x_cols + [final_it_col]
            new_df = df[cols_to_keep].copy()
            new_df.rename(columns={final_it_col: "y_col_0"}, inplace=True)

            return new_df

        except Exception as e:
            print(f"Warning: Failed to reduce result df: {e}")
            return None

    def validate_candidates(
        self,
        limited_candidates: list,
        df_dict: dict,
        x_col_count: int,
        max_tables: int,
    ) -> list[dict]:
        """
        Validate candidates by checking FD.

        Args:
            limited_candidates: list
            df_dict: dict
            x_col_count: int
            max_tables: int

        Returns:
            validated_tables: list[dict]
        """
        validated_tables = list()

        if self.use_parallel_fd and len(limited_candidates) > 10:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                fd_args = list()

                for candidate in limited_candidates:
                    table_id = candidate[0]
                    right_x_cols = list(candidate[1 : 1 + x_col_count])
                    df_right = df_dict.get(table_id)

                    if df_right is None:
                        continue
                    if not all(col in df_right.columns for col in right_x_cols):
                        continue

                    fd_args.append((table_id, right_x_cols, df_right))

                for result in executor.map(self.check_single_fd, fd_args):
                    if result:
                        validated_tables.append(result)
                        if len(validated_tables) >= max_tables:
                            break
        else:
            for candidate in limited_candidates:
                table_id = candidate[0]
                right_x_cols = list(candidate[1 : 1 + x_col_count])
                df_right = df_dict.get(table_id)

                if df_right is None:
                    continue
                if not all(col in df_right.columns for col in right_x_cols):
                    continue

                fd_result = self.check_fd(x_cols=right_x_cols, df=df_right)

                if fd_result:
                    x_cols_result, y_cols_result, filtered_df = fd_result
                    validated_tables.append(
                        {
                            "table_id": table_id,
                            "x_cols": x_cols_result,
                            "y_cols": y_cols_result,
                            "df": filtered_df,
                        }
                    )

                    if len(validated_tables) >= max_tables:
                        break

        return validated_tables

    def prepare_flat_y(
        self,
        cleaned_y: list[list[str]],
    ) -> tuple[set, list | None]:
        """
        Prepare flat_y and flat_y_list from cleaned_y.

        Args:
            cleaned_y: list[list[str]]

        Returns:
            flat_y_raw: set
            list(flat_y_raw): list | None
        """
        if cleaned_y and isinstance(cleaned_y[0], list):
            flat_y_raw = set(val for sublist in cleaned_y for val in sublist)
        else:
            flat_y_raw = set(cleaned_y if cleaned_y else [])

        if self.use_fuzzy_y_match:
            return flat_y_raw, list(flat_y_raw)
        else:
            return flat_y_raw, None

    def tableJoinerInnerLoop(
        self,
        candidates: list[tuple],
        left_df: pd.DataFrame,
        left_join_cols: list[str],
        flat_y: set,
        flat_y_list: list | None,
        max_tables: int,
        iteration_index: int = 1,
        restrict_nums: bool | None = None,
    ) -> tuple[list[pd.DataFrame], list[tuple]]:
        """
        Core multi-hop joining logic.

        Args:
            candidates: list[tuple]
            left_df: pd.DataFrame
            left_join_cols: list[str]
            flat_y: set
            flat_y_list: list | None
            max_tables: int
            iteration_index: int (Default: 1)
            restrict_nums: bool | None (Default: None) ##Nur Config?

        Returns:
            results: list[pd.DataFrame]
            next_path: list[tuple]
        """
        if restrict_nums is None:
            restrict_nums = self.config.restrict_nums

        if len(candidates) > max_tables * 10:
            candidates = candidates[: max_tables * 10]

        results = list()
        next_paths = list()
        survived = 0

        x_col_count = len(left_join_cols)
        limited_candidates = candidates[: max_tables * 2]
        table_ids = [candidate[0] for candidate in limited_candidates]

        self.previously_seen_tables.update(table_ids)
        df_dict = self.load_table_batch(table_ids=table_ids)

        validated_tables = self.validate_candidates(
            limited_candidates=limited_candidates,
            df_dict=df_dict,
            x_col_count=x_col_count,
            max_tables=max_tables,
        )

        for table_info in validated_tables:
            table_id = table_info["table_id"]
            right_x_cols = table_info["x_cols"]
            z_cols = table_info["y_cols"]
            df_right = table_info["df"]

            for z_col in z_cols:
                try:
                    cols_to_use = right_x_cols + [z_col]
                    df_right_hop = df_right[cols_to_use].copy()

                    for col in right_x_cols:
                        df_right_hop = df_right_hop[
                            df_right_hop[col]
                            .replace(r"^\s*$", np.nan, regex=True)
                            .notna()
                        ]

                    if df_right_hop.empty:
                        continue

                    df_unique = df_right_hop.drop_duplicates(subset=right_x_cols)
                    col_card = list(df_unique.nunique(dropna=True).to_dict().values())

                    if df_unique is None or df_unique.empty:
                        continue

                    if not np.all(np.array(col_card) >= self.tau):
                        continue

                    new_vals = df_unique[z_col]
                    if not len(new_vals) > 0:
                        continue

                    if restrict_nums:
                        numeric_mask = pd.to_numeric(new_vals, errors="coerce").notna()
                        if numeric_mask.mean() >= self.metric_precision:
                            continue

                    if new_vals.isna().mean() >= self.none_precision:
                        continue

                    if (
                        new_vals.replace(r"^\s*$", np.nan, regex=True).isna().mean()
                        >= self.none_precision
                    ):
                        continue

                    if (
                        new_vals.str.len() < self.min_word_len
                    ).mean() >= self.none_precision:
                        continue

                    merged_df = pd.merge(
                        left_df,
                        df_unique,
                        left_on=left_join_cols,
                        right_on=right_x_cols,
                        how="outer",
                        suffixes=("", "_drop_candidate"),
                    )

                    if len(merged_df) > self.large_df_threshold:
                        example_df = merged_df[merged_df["is_example"]]
                        non_example_df = merged_df[not merged_df["is_example"]]

                        n_non_examples = len(non_example_df)
                        sample_size = min(self.sample_size, n_non_examples)

                        if sample_size > 0 and sample_size < n_non_examples:
                            non_example_df = non_example_df.sample(
                                n=sample_size,
                                random_state=self.seed,
                            )
                            # original_size = len(example_df) + n_non_examples
                            merged_df = pd.concat(
                                [example_df, non_example_df], ignore_index=True
                            )

                    merged_df["is_example"] = merged_df["is_example"].fillna(False)

                    if merged_df.empty:
                        continue

                    cols_to_remove = list()
                    for rc in right_x_cols:
                        if rc in merged_df.columns:
                            cols_to_remove.append(rc)
                        elif f"{rc}_drop_candidate" in merged_df.columns:
                            cols_to_remove.append(f"{rc}_drop_candidate")

                    merged_df.drop(
                        columns=cols_to_remove, inplace=True, errors="ignore"
                    )

                    if merged_df.empty:
                        continue

                    if z_col in merged_df.columns:
                        actual_z_col = z_col
                    elif f"{z_col}_drop_candidate" in merged_df.columns:
                        actual_z_col = f"{z_col}_drop_candidate"
                    else:
                        try:
                            actual_z_col = (
                                merged_df.columns[int(z_col)]
                                if isinstance(z_col, int)
                                else z_col
                            )
                        except (ValueError, IndexError):
                            continue

                    n_unique_lhs, n_unique_rhs = self.in_loop_fd(
                        merged_df=merged_df,
                        left_join_cols=left_join_cols,
                        actual_z_col=actual_z_col,
                    )

                    if (
                        n_unique_lhs == 0
                        or n_unique_rhs / n_unique_lhs < self.fd_threshold
                    ):
                        continue

                    existing_values = set()
                    for col in merged_df.columns:
                        if col != actual_z_col:
                            existing_values.update(
                                merged_df[col].dropna().astype(str).str.lower().unique()
                            )

                    new_values = set(
                        merged_df[actual_z_col]
                        .dropna()
                        .astype(str)
                        .str.lower()
                        .unique()
                    )
                    overlap = new_values.intersection(existing_values)

                    if self.strict_uniqueness_in_df:
                        if len(overlap) > 0:
                            continue
                    else:
                        if (
                            len(new_values) > 0
                            and len(overlap) / len(new_values) > self.overlap_threshold
                        ):
                            continue

                    new_col_name = f"it{iteration_index}"
                    merged_df.rename(columns={actual_z_col: new_col_name}, inplace=True)
                    actual_z_col = new_col_name

                    non_empty_mask = (
                        merged_df[actual_z_col]
                        .replace(r"^\s*$", np.nan, regex=True)
                        .notna()
                    )
                    if merged_df[non_empty_mask][actual_z_col].nunique() < 2:
                        continue

                    example_rows = merged_df[merged_df["is_example"]]
                    found_values = example_rows[actual_z_col].dropna().unique()

                    overlap_count = len(flat_y.intersection(found_values))

                    if overlap_count >= self.tau:
                        results.append(merged_df)
                        continue

                    elif self.use_fuzzy_y_match and flat_y_list:
                        fuzzy_matches = 0
                        for found_val in found_values:
                            found_str = str(found_val).strip().lower()
                            for target_val in flat_y_list:
                                target_str = str(target_val).strip().lower()
                                if fuzzy_match(
                                    found_str,
                                    target_str,
                                    scorer=self.fuzzy_scorer,
                                    threshold=self.fuzzy_threshold,
                                ):
                                    fuzzy_matches += 1
                                    break

                        if fuzzy_matches >= self.tau:
                            results.append(merged_df)
                            continue

                    if (
                        n_unique_lhs > 0
                        and n_unique_rhs / n_unique_lhs >= self.fd_threshold
                    ):
                        if (
                            len(example_rows[example_rows[actual_z_col].notna()])
                            >= self.tau
                        ):
                            next_paths.append((merged_df, actual_z_col))
                            survived += 1

                except Exception as e:
                    print(
                        f"Warning: Error processing z_col {z_col} in table {table_id}: {e}"
                    )
                    continue

        if next_paths:
            scored_paths = list()

            for path_df, path_col in next_paths:
                try:
                    example_rows = path_df[path_df["is_example"]]
                    match_count = len(example_rows[example_rows[path_col].notna()])
                    cardinality = path_df[path_col].nunique()

                    score = match_count * np.log1p(cardinality)
                    scored_paths.append((score, path_df, path_col))

                except Exception as e:
                    print(f"Warning: Error scoring path: {e}")
                    continue

            scored_paths.sort(reverse=True, key=lambda x: x[0])

            max_paths_to_keep = min(self.max_paths_to_keep, len(scored_paths))
            next_paths = [(df, col) for _, df, col in scored_paths[:max_paths_to_keep]]

        self.survived[iteration_index] += survived
        return results, next_paths

    def my_queue(
        self,
        cleaned_x: list[list[str]],
        cleaned_y: list[list[str]],
        previously_seen_tables: set | None = None,
    ) -> pd.DataFrame | list:
        """
        Performs multi-hop table joining to find relevant tables.

        Args:
            cleaned_x: list[list[str]]
            cleaned_y: list[list[str]]
            previously_seen_tables: set | None (Default: None)

        Returns:
            pd.DataFrame | None
        """
        if not cleaned_x or not all(cleaned_x):
            raise ValueError("cleaned_x must be non-empty")

        if not cleaned_y or not all(cleaned_y):
            raise ValueError("cleaned_y must be non-empty")

        if self.reset_counters:
            self.survived.clear()

        restrict_nums = self.config.restrict_nums
        if self.auto_detect_numeric:
            has_numeric_x = self.detect_numeric_columns(data_cols=cleaned_x)
            has_numeric_y = (
                self.detect_numeric_columns(data_cols=cleaned_y) if cleaned_y else False
            )

            if has_numeric_x or has_numeric_y:
                restrict_nums = False

        flat_y, flat_y_list = self.prepare_flat_y(cleaned_y)

        path_len = 1

        current_max_tables = self.max_tables

        input_data_dict = {
            f"x_col_{i}": col_vals for i, col_vals in enumerate(cleaned_x)
        }
        df_input = pd.DataFrame(input_data_dict)
        df_input["is_example"] = True
        input_merge_cols = [c for c in df_input.columns if c != "is_example"]

        if previously_seen_tables is not None:
            self.previously_seen_tables.update(previously_seen_tables)

        multi_z = self.qf.find_xy_candidates(
            x_cols=cleaned_x,
            y_cols=None,
            tau=self.tau,
            multi_hop=True,
            table_limit=self.max_tables * 3,  # Heuristic, not experimentally validated
            previously_seen_tables=self.previously_seen_tables,
        )

        if not multi_z:
            return None

        self.update_seen(multi_z)

        final_tables = list()
        found_tables, current_table_paths = self.tableJoinerInnerLoop(
            candidates=multi_z,
            left_df=df_input,
            left_join_cols=input_merge_cols,
            flat_y=flat_y,
            flat_y_list=flat_y_list,
            max_tables=current_max_tables,
            iteration_index=path_len,
            restrict_nums=restrict_nums,
        )

        for df in found_tables:
            reduced = self.reduce_result_df(df=df)
            if reduced is not None:
                final_tables.append(reduced)

        while path_len < self.max_path_len:
            path_len += 1

            if self.adaptive_limits:
                current_max_tables = max(
                    1,
                    self.max_tables
                    // (self.adaptive_reduction_factor ** (path_len - 1)),
                )

            paths_to_process = current_table_paths
            current_table_paths = list()

            for prev_df, join_col_name in paths_to_process:
                try:
                    example_only_df = prev_df[prev_df["is_example"]]
                    selective_value = (
                        example_only_df[join_col_name].dropna().unique().tolist()
                    )

                    if not selective_value:
                        continue

                    next_z = self.qf.find_xy_candidates(
                        x_cols=[selective_value],
                        y_cols=None,
                        tau=self.tau,
                        multi_hop=True,
                        table_limit=self.max_tables * 3,
                        previously_seen_tables=self.previously_seen_tables,
                    )

                    if not next_z:
                        continue

                    self.update_seen(next_z)

                    found, next_hops = self.tableJoinerInnerLoop(
                        candidates=next_z,
                        left_df=prev_df,
                        left_join_cols=[join_col_name],
                        flat_y=flat_y,
                        flat_y_list=flat_y_list,
                        max_tables=current_max_tables,
                        iteration_index=path_len,
                        restrict_nums=restrict_nums,
                    )

                    for df in found:
                        reduced = self.reduce_result_df(df=df)
                        if reduced is not None:
                            final_tables.append(reduced)
                    current_table_paths.extend(next_hops)

                except Exception as e:
                    print(
                        f"Warning: Error processing path in iteration {path_len}: {e}"
                    )
                    continue

        if not final_tables:
            if self.print_further_information:
                print("Found 0 tables")
                print(f"Survival stats: {dict(self.survived)}")
            return None

        cleaned_results = list()

        for df in final_tables:
            try:
                x_cols = [c for c in df.columns if c.startswith("x_col_")]

                df = df.dropna(subset=x_cols, how="all")

                if "y_col_0" in df.columns:
                    df = df[df["y_col_0"].notna()]
                    df = df[df["y_col_0"].astype(str).str.strip() != ""]
                    dedup_cols = x_cols + ["y_col_0"]
                else:
                    dedup_cols = x_cols

                df.drop_duplicates(subset=dedup_cols, inplace=True)

                if df.empty:
                    continue

                df.reset_index(drop=True, inplace=True)
                cleaned_results.append(df)

            except Exception as e:
                print(f"Error cleaning individual table: {e}")
                continue

        if not cleaned_results:
            if self.print_further_information:
                print("All results filtered out due to cleaning")
            return None

        if self.print_further_information:
            print(f"Found {len(cleaned_results)} distinct tables.")
            print(f"Survival stats: {dict(self.survived)}")

        return cleaned_results
