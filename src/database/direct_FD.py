import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import os
from functools import lru_cache 
import re
from collections import defaultdict

class DirectDependencyVerifier:

    def __init__(self, query_factory, cache_dir="./fd_verifier_cache"):
        self.qf = query_factory
        self.conn = query_factory.conn
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.result_cache_file = self.cache_dir / "verification_results.pkl"
        self.result_cache = self._load_result_cache()

        self.survived = defaultdict(int)  # Initialisiert neue Keys automatisch mit 0                      ####<---- Nur für Debugg 

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

    def verify_candidates(self, candidates, x_col_count, error_threshold=0.05, limit: int | None = None):

        results = []
        
        for cand in candidates:
            if limit and len(results) >= limit:
                break
                
            tid = cand[0]
            lhs = list(cand[1:1+x_col_count])

            cache_key = (tid, tuple(sorted(lhs)))
            if cache_key in self.result_cache:
                cached_result = self.result_cache[cache_key]
                if cached_result:
                    results.append((tid, lhs, cached_result))
                if limit and len(results) >= limit:
                    break
                continue

            df = self.load_table_data(tid, include_cols=None)
            if df is None:
                continue

            df_clean = df.dropna(subset=lhs)
            
            if df_clean.empty:
                self.result_cache[cache_key] = []
                continue
            
            valid_rhs = []
            rhs_candidates = [c for c in df_clean.columns if c not in lhs]
            
            for rhs in rhs_candidates:
                df_for_fd = df_clean[[*lhs, rhs]].dropna()
                
                if df_for_fd.empty:
                    valid_rhs.append(rhs)
                    continue

                df_unique = df_for_fd.drop_duplicates()

                n_unique_lhs = df_unique[lhs].drop_duplicates().shape[0]

                n_unique_rhs = df_unique[rhs].nunique()

                if n_unique_lhs == 0:
                    valid_rhs.append(rhs)
                    continue
                
                uniqueness_ratio = n_unique_rhs / n_unique_lhs

                if uniqueness_ratio >= (1.0 - error_threshold):
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

        if multi_z: 
            new_tabs = set(r[0] for r in erg)               ###Helper? 
            previously_seen_tables.update(new_tabs)          ###Hier

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
                
                new_new_tabs = set(r[0] for r in erg)               ###helper? 
                previously_seen_tables.update(new_new_tabs)               ###hier

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
        print(final_tables)
        print(self.survived)
        return final_tables


    def tableJoinerInnterLoop(self, 
                              candidates, 
                              left_df, 
                              left_join_cols, 
                              flat_y, 
                              tau:int, 
                              previously_seen:set, 
                              max_tables:int, 
                              iteration_index:int=1, 
                              fd_threshold:float=0.95, 
                              restrict_nums:bool = True, 
                              metric_precision:float = 0.1, 
                              none_precision:float = 0.1, 
                              min_word_len: int = 2,  
                              stopwords: set| None = None,
                              strict_uniquness_in_df: bool = False, 
                              ):


        results = list()
        next_paths = list()

        survived = 0                                        #<---- Nur für debugg 

        x_col_count = len(left_join_cols)

        sub_mats = self.verify_candidates(candidates, 
                                          x_col_count, 
                                          limit=max_tables, 
                                          )

        # for all T in T_X do (mit Limitierung auf max_tables)
        # [Z_0, ..., Z_n] <- findJoinColumns(T, E.X)
        for sub_mat in sub_mats:
            table_ID, right_x_cols, z_cols = sub_mat 
            previously_seen.add(table_ID)


            # for all Z_i in [Z_0...Z_n] do
            for z_col in z_cols:

                cols_to_load = tuple(right_x_cols + [z_col])
                
                df_right = self.load_table_data(table_ID, include_cols=cols_to_load)
                df_unique = df_right.drop_duplicates(subset=right_x_cols)
                
                col_card = list(df_unique.nunique(dropna = True).to_dict().values()) if df_unique is not None else {} 

                ##Rausschmeißer 
                if df_unique is None or df_unique.empty: 
                    continue

                if not np.all(np.array(col_card) >= tau): 
                    continue
                
                ###Das hier ist neu und nicht aus dem Paper. Ich habe das Gefühl, dass einzelne Zeilen, die nur aus Zahlen bestehen, ungeeignet für Multi-Hop sind.
                ###Hat zumindest in der Vergangenheit Quatsch Ergebnisse geliefert und sortiert weiter aus! 
                ##Muss vielleicht lieber in Val? 

                new_vals = df_unique[z_col]
                if not len(new_vals) > 0: 
                    continue
                ###Kill Nums as they just return bs val s
                if restrict_nums and np.mean([self.is_numeric_like(val) for val in new_vals]) >= metric_precision: 
                    continue 
                ###Kill Nones up to a precision 
                if np.mean([val is None for val in new_vals]) >= none_precision: 
                    continue                    ###Vielleicht nicht continure sondern die rauswerfen? 
                ###Kill empty dfs up to the same precision 
                if new_vals.replace(r'^\s*$', np.nan, regex=True).isna().mean() >= none_precision: 
                    continue                
                ###Kill everything with words below a certian word len 
                cleares_min_len = new_vals.astype(str).str.len() < min_word_len 
                if cleares_min_len.mean() >= none_precision: 
                    continue                ####Alternativ alle Werte beim bilden der Projections auf None setzen?
                

                # Furthermore, we check whether the cardinality of the
                # discovered Zi is greater or equal than the corresponding E.Y.
                # If this is not the case, we know apriori that the functional path
                # to the transformation result cannot be maintained.

                # T_j <- findJoinableTables(Z_i, E, T)
                merged_df = pd.merge(
                    left_df, 
                    df_unique, 
                    left_on=left_join_cols, 
                    right_on=right_x_cols, 
                    how='inner',
                    suffixes=('', '_drop_candidate') 
                )
                num_rows = len(merged_df.index)

                if merged_df.empty or num_rows < tau: ##zweiter Check um zu schauen, ob für E überhaupt merges möglich sind. 
                    continue
                # Our strategy to reduce the search space is to require
                # the functional dependency constraint on at least τ of the initial
                # examples throughout the indirection and no contradiction with
                # regard to the given examples, as shown in Algorithm 1
                

                ###Doppelte Cols droppen
                cols_to_remove = []
                for rc in right_x_cols:

                    if rc in merged_df.columns:
                        cols_to_remove.append(rc)

                    elif f"{rc}_drop_candidate" in merged_df.columns:
                        cols_to_remove.append(f"{rc}_drop_candidate")
                
                merged_df.drop(columns=cols_to_remove, inplace=True, errors='ignore')                       
                
                ###Namen ändern, damit namen nicht in höheren Itteartionen gleich sind und verbuggen 
                if z_col in merged_df.columns:
                    actual_z_col = z_col
                elif f"{z_col}_drop_candidate" in merged_df.columns:
                    actual_z_col = f"{z_col}_drop_candidate"
                else:
                    try:
                        actual_z_col = merged_df.columns[int(z_col)] if isinstance(z_col, int) else z_col
                    except (ValueError, IndexError):
                        continue

                ###FD Check 
                n_unique_lhs, n_unique_rhs = self.in_loop_fd(merged_df, left_join_cols, actual_z_col)
                if n_unique_lhs == 0 or n_unique_rhs / n_unique_lhs < fd_threshold:
                        continue


                ###Cols mit weitestgehend doppelten Werten killen 
                existing_values = set() 
                for col in merged_df.columns:
                    if col != actual_z_col:
                        existing_values.update(merged_df[col].dropna().astype(str).str.lower().unique())

                new_values = set(merged_df[actual_z_col].dropna().astype(str).str.lower().unique())
                overlap = new_values.intersection(existing_values)

                if strict_uniquness_in_df: 
                    if len(overlap) > 0:
                        continue
                else: 
                    if len(new_values) > 0 and len(overlap) / len(new_values) > 0.5:
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
                    continue         
                
                ###FD Check 
                n_unique_lhs, n_unique_rhs = self.in_loop_fd(merged_df, left_join_cols, actual_z_col)
                if n_unique_lhs > 0 and n_unique_rhs / n_unique_lhs >= fd_threshold:
                        next_paths.append((merged_df, actual_z_col)) 
                        survived +=1                                            #<---- Nur für Debugg

        self.survived[iteration_index] += survived                #<---- Nur für Debugg 

        return results, next_paths
    
    
    
    def is_numeric_like(self, val):
        s = str(val).strip()
        
        cleaned = s.replace(' ', '').replace(',', '.').replace('\xa0', '')
        
        try:
            float(cleaned)
            return True
        except ValueError:
            pass
        
        ##Dates
        if re.match(r'^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$', s):
            return True

        #Nums with Seperation, eg. , or space 
        if re.match(r'^[\d\s,\.]+$', s) and any(c.isdigit() for c in s):
            return True
        
        return False

    def in_loop_fd(self, merged_df, left_join_cols, actual_z_col): 
        fd_check_df = merged_df[[*left_join_cols, actual_z_col]].dropna()
        if not fd_check_df.empty:
            fd_unique = fd_check_df.drop_duplicates()
            n_unique_lhs = fd_unique[left_join_cols].drop_duplicates().shape[0]
            n_unique_rhs = fd_unique[actual_z_col].nunique()
            return n_unique_lhs, n_unique_rhs
        
    


# import cProfile
# import pstats
# profiler = cProfile.Profile()
# profiler.enable()
#####Hier könnte ihre Func stehen 
# profiler.disable()
# stats = pstats.Stats(profiler)
# stats.strip_dirs()
# stats.sort_stats("time")
# stats.print_stats()