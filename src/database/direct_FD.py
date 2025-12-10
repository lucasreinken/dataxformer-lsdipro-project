import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import os
from functools import lru_cache 

class DirectDependencyVerifier:

    def __init__(self, query_factory, cache_dir="./fd_verifier_cache"):
        self.qf = query_factory
        self.conn = query_factory.conn
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.result_cache_file = self.cache_dir / "verification_results.pkl"
        self.result_cache = self._load_result_cache()

    def _load_result_cache(self):
        if self.result_cache_file.exists():
            try:
                with open(self.result_cache_file, 'rb') as f:
                    return pickle.load(f)
            except EOFError:
                return {}
        return {}

    def _save_result_cache(self):
        with open(self.result_cache_file, 'wb') as f:
            pickle.dump(self.result_cache, f)

    @lru_cache(maxsize=32)
    def load_table_data(self, table_id: int, include_cols: tuple = None): 

        try:
            return self.qf.get_table_content(table_id, include_cols)
        except Exception as e:
            print(f"Warning: Failed to load table {table_id}. Error: {e}")
            return None



    def verify_candidates(self, candidates, x_col_count, error_threshold=0.05, limit:int | None = None):
        results = []
        for cand in candidates:
            if limit and len(results) >= limit:
                break
            tid = cand[0]
            lhs = list(cand[1:1+x_col_count])
            
            cache_key = (tid, tuple(sorted(lhs)))
            if cache_key in self.result_cache:
                results.append((tid, lhs, self.result_cache[cache_key]))
                if limit and len(results) >= limit:
                    break
                continue

            df = self.load_table_data(tid, include_cols=None) 
            
            if df is None: continue

            valid_rhs = []
            df_clean = df.dropna(subset=lhs)
            
            if not df_clean.empty:
                grouped = df_clean.groupby(lhs)
                rhs_candidates = [c for c in df.columns if c not in lhs]
                
                for rhs in rhs_candidates:
                    def calc_err(x):
                        v = x.dropna()
                        return (len(v) - v.value_counts().iloc[0]) if not v.empty else 0
                    
                    err_sum = grouped[rhs].apply(calc_err).sum()
                    count = df_clean[rhs].notna().sum()
                    
                    if count > 0 and (err_sum / count) <= error_threshold:
                        valid_rhs.append(rhs)
                    elif count == 0:
                         valid_rhs.append(rhs)

            self.result_cache[cache_key] = valid_rhs
            if valid_rhs:
                results.append((tid, lhs, valid_rhs))
                if limit and len(results) >= limit:
                    break
        
        self._save_result_cache()
        return results
    

    def my_queue(self, erg, cleaned_x, cleaned_y, tau, previously_seen_tables=None, max_path_len=1, max_tables=3): 
        # path <- 1
        path_len = 1  

        input_data_dict = {f"input_col_{i}": col_vals for i, col_vals in enumerate(cleaned_x)} ###sollte noch anders aussehen
        df_input = pd.DataFrame(input_data_dict)
        input_merge_cols = list(df_input.columns)

        if previously_seen_tables is None: 
            previously_seen_tables = set()
        erg_table_ids = set(r[0] for r in erg)
        previously_seen_tables.update(erg_table_ids) # T_E

        if cleaned_y and isinstance(cleaned_y[0], list):
            flat_y = set(val for sublist in cleaned_y for val in sublist)
        else:
            flat_y = set(cleaned_y if cleaned_y else [])
                
        # T_X <- QueryForTables(E.X) 
        # Also enforcing T_X <- T_X / T_E
        multi_z = self.qf.find_xy_candidates(
                                        cleaned_x, 
                                        None, 
                                        tau, 
                                        multi_hop=True, 
                                        limit = max_tables*2, 
                                        previously_seen_tables = previously_seen_tables,
                                        )   
        ###*2 ist random, sollte statistisch ermittelt werden 

        final_tables = list() 
        found_tables, current_table_paths = self.tableJoinerInnterLoop(
            multi_z, 
            df_input, 
            input_merge_cols, 
            flat_y, 
            tau, 
            previously_seen_tables, 
            max_tables,
            iteration_index = path_len
        )
        final_tables.extend(found_tables)

        while path_len < max_path_len:
            path_len += 1 

            paths_to_process = current_table_paths
            current_table_paths = list()

            #for all <T_p, Z> in current_table_paths do 
            for prev_df, join_col_name in paths_to_process:

                selective_value = prev_df[join_col_name].dropna().unique().tolist()
                
                if not selective_value: 
                    continue

                next_z = self.qf.find_xy_candidates(
                    [selective_value],
                    None, 
                    tau, 
                    multi_hop=True, 
                    limit=max_tables*2, 
                    previously_seen_tables=previously_seen_tables,
                )
                
                if not next_z: 
                    continue

                found, next_hops = self.tableJoinerInnterLoop(
                    next_z, 
                    prev_df, 
                    [join_col_name], 
                    flat_y, 
                    tau,
                    previously_seen_tables, 
                    max_tables,
                    iteration_index = path_len
                )
                
                final_tables.extend(found)
                current_table_paths.extend(next_hops)

        return final_tables


    def tableJoinerInnterLoop(self, candidates, left_df, left_join_cols, flat_y, tau, 
                     previously_seen, max_tables, iteration_index:int=1):

        results = []
        next_paths = []

        x_col_count = len(left_join_cols)

        sub_mats = self.verify_candidates(candidates, x_col_count, limit=max_tables)

        # for all T in T_X do (mit Limitierung auf max_tables)
        # [Z_0, ..., Z_n] <- findJoinColumns(T, E.X)
        for sub_mat in sub_mats:
            table_ID, right_x_cols, z_cols = sub_mat 
            previously_seen.add(table_ID)
            # for all Z_i in [Z_0...Z_n] do
            for z_col in z_cols:

                cols_to_load = tuple(right_x_cols + [z_col])
                
                df_right = self.load_table_data(table_ID, include_cols=cols_to_load)
                
                if df_right is None or df_right.empty: 
                    continue

                # T_j <- findJoinableTables(Z_i, E, T)
                merged_df = pd.merge(
                    left_df, 
                    df_right, 
                    left_on=left_join_cols, 
                    right_on=right_x_cols, 
                    how='inner',
                    suffixes=('', '_drop_candidate') 
                )

                if merged_df.empty: 
                    continue

                cols_to_remove = []
                for rc in right_x_cols:

                    if rc in merged_df.columns:
                        cols_to_remove.append(rc)

                    elif f"{rc}_drop_candidate" in merged_df.columns:
                        cols_to_remove.append(f"{rc}_drop_candidate")
                
                merged_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')

                if z_col in merged_df.columns:
                    actual_z_col = z_col
                elif f"{z_col}_drop_candidate" in merged_df.columns:
                    actual_z_col = f"{z_col}_drop_candidate"
                else:
                    try:
                        actual_z_col = merged_df.columns[int(z_col)] if isinstance(z_col, int) else z_col
                    except (ValueError, IndexError):
                        continue
                new_col_name = f"it{iteration_index}"
                merged_df.rename(columns={actual_z_col: new_col_name}, inplace=True)
                actual_z_col = new_col_name
                try:
                    found_values = merged_df[actual_z_col].unique()
                except KeyError:
                    continue

                overlap_count = len(flat_y.intersection(found_values))

                print(merged_df)
                if overlap_count >= tau:
                    results.append(merged_df)

                next_paths.append((merged_df, actual_z_col))
        
        return results, next_paths