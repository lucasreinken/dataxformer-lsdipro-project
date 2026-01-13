import pandas as pd
import numpy as np
import re
from collections import defaultdict

class DirectDependencyVerifier:

    def __init__(self, query_factory,):
        self.qf = query_factory
        self.conn = query_factory.conn
        self.previously_seen_tables = set()

        self.survived = defaultdict(int)
    
    def load_table_data(self, table_id: int, include_cols: tuple = None): 

        try:
            return self.qf.get_table_content(table_id, include_cols)
        except Exception as e:
            print(f"Warning: Failed to load table {table_id}. Error: {e}")
            return None
        

    def load_table_batch(self, whattoload: list[tuple]): 
        
        return {table_id: self.qf.get_table_content(table_id, include_cols=None) for table_id in whattoload}
        
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
    
    def update_seen(self, new_tables): 
        new_table_ids = set(r[0] for r in new_tables)
        self.previously_seen_tables.update(new_table_ids)

    
    def check_fd(self, 
                x_cols, 
                df:pd.DataFrame,
                error_threshold=0.05, 
                ):
        
        if df is None:
            return None

        df_clean = df.dropna(subset=x_cols)
        
        if df_clean.empty:
            return None
        
        valid_y_cols = []
        y_col_candidates = [c for c in df_clean.columns if c not in x_cols]
        
        for y_col in y_col_candidates:
            df_for_fd = df_clean[[*x_cols, y_col]].dropna()
            
            if df_for_fd.empty:
                valid_y_cols.append(y_col)
                continue

            df_unique = df_for_fd.drop_duplicates()

            n_unique_lhs = df_unique[x_cols].drop_duplicates().shape[0]

            n_unique_rhs = df_unique[y_col].nunique()

            if n_unique_lhs == 0:
                valid_y_cols.append(y_col)
                continue
            
            uniqueness_ratio = n_unique_rhs / n_unique_lhs

            if uniqueness_ratio >= (1.0 - error_threshold):
                valid_y_cols.append(y_col)
        
        if valid_y_cols:
            filtered_df = df_clean[x_cols + valid_y_cols].copy()
            return (x_cols, valid_y_cols, filtered_df)
        
        return None

    
    def itterate_candidates(self, 
                            candidates:list[tuple], 
                            x_col_count:int, 
                            limit:int|None = None,
                            ): 
        
        results = list() 
        anz_valid = 0
        for candidate in candidates:
            if limit and anz_valid >= limit: 
                break
            Table_ID = candidate[0]
            self.previously_seen_tables.add(Table_ID)
            x_cols = list(candidate[1:1+x_col_count])
            df = self.load_table_data(Table_ID, include_cols=None)
            
            if df is None:
                continue
            
            fd_result = self.check_fd(x_cols, df)
            if fd_result:
                x_cols_result, y_cols_result, filtered_df = fd_result
                results.append({
                    'table_id': Table_ID,
                    'x_cols': x_cols_result,
                    'y_cols': y_cols_result,
                    'df': filtered_df,
                })
                anz_valid += 1 

        return results
    
    def itterate_batch(self, 
                            candidates:list[tuple], 
                            x_col_count:int, 
                            tau:int, 
                            limit:int|None = None,
                            ): 
        results = list()
        
        df_dict = self.load_table_batch(candidates[:limit])
        for idx, Table_ID, df in enumerate(df_dict.items()): 
            if df is None: 
                continue
            if not all(col in df.columns for col in x_cols):
                continue
            if len(df) < tau: 
                continue 
            x_cols = list(candidates[idx][1: 1+x_col_count])
            fd_result = self.check_fd(x_cols, df)
            if fd_result:
                x_cols_result, y_cols_result, filtered_df = fd_result
                results.append({
                    'table_id': Table_ID,
                    'x_cols': x_cols_result,
                    'y_cols': y_cols_result,
                    'df': filtered_df,
                })

        return results
    
    def tableJoinerInnterLoop(self, 
                              candidates, 
                              left_df, 
                              left_join_cols, 
                              flat_y, 
                              tau:int, 
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

        survived = 0

        x_col_count = len(left_join_cols)

        dfs = self.itterate_candidates(candidates, 
                                          x_col_count, 
                                          limit=max_tables, 
                                          )

        for table_info in dfs:
            table_id = table_info['table_id']
            right_x_cols = table_info['x_cols']
            z_cols = table_info['y_cols']
            df_right = table_info['df']

            for z_col in z_cols:

                cols_to_use = right_x_cols + [z_col]
                df_unique = df_right[cols_to_use].drop_duplicates(subset=right_x_cols)
                
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
                    continue
                ###Kill empty dfs up to the same precision 
                if new_vals.replace(r'^\s*$', np.nan, regex=True).isna().mean() >= none_precision: 
                    continue                
                ###Kill everything with words below a certian word len 
                cleares_min_len = new_vals.astype(str).str.len() < min_word_len 
                if cleares_min_len.mean() >= none_precision: 
                    continue
                

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

                if merged_df.empty or num_rows < tau:
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
                if n_unique_lhs > 0 and n_unique_rhs / n_unique_lhs >= fd_threshold:
                        next_paths.append((merged_df, actual_z_col)) 
                        survived +=1

        self.survived[iteration_index] += survived

        return results, next_paths
    
    def in_loop_fd(self, merged_df, left_join_cols, actual_z_col): 
        fd_check_df = merged_df[[*left_join_cols, actual_z_col]].dropna()
        if not fd_check_df.empty:
            fd_unique = fd_check_df.drop_duplicates()
            n_unique_lhs = fd_unique[left_join_cols].drop_duplicates().shape[0]
            n_unique_rhs = fd_unique[actual_z_col].nunique()
            return n_unique_lhs, n_unique_rhs
        return 0, 0


    def my_queue(self, 
                 erg, 
                 cleaned_x, 
                 cleaned_y, 
                 tau:int, 
                 max_path_len:int=1, 
                 max_tables:int=3,
                 ): 
        # path <- 1
        path_len = 1  

        input_data_dict = {f"x_col_{i}": col_vals for i, col_vals in enumerate(cleaned_x)}
        df_input = pd.DataFrame(input_data_dict)
        input_merge_cols = list(df_input.columns)

        self.update_seen(erg)

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
                                        previously_seen_tables = self.previously_seen_tables,
                                        )
        ###*2 ist random, sollte statistisch ermittelt werden 

        if not multi_z: 
            return []
        
        self.update_seen(multi_z)

        final_tables = list() 
        found_tables, current_table_paths = self.tableJoinerInnterLoop(
            multi_z, 
            df_input, 
            input_merge_cols, 
            flat_y, 
            tau,  
            max_tables,
            iteration_index = path_len,
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
                    previously_seen_tables=self.previously_seen_tables,
                )
                
                if not next_z: 
                    continue
                
                self.update_seen(next_z)

                found, next_hops = self.tableJoinerInnterLoop(
                    next_z, 
                    prev_df, 
                    [join_col_name], 
                    flat_y, 
                    tau,
                    max_tables,
                    iteration_index = path_len,
                )
                
                final_tables.extend(found)
                current_table_paths.extend(next_hops)
        print(final_tables)
        print(self.survived)
        return final_tables