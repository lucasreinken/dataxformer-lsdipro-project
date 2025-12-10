import os
import vertica_python
import pandas as pd
from dotenv import load_dotenv
from functools import cache
import pandas as pd

class QueryFactory:

    def __init__(self, config) -> None:
        load_dotenv()

        self.host = os.getenv("VERTICA_HOST")
        self.port = os.getenv("VERTICA_PORT")
        self.user = os.getenv("VERTICA_USER")
        self.password = os.getenv("VERTICA_PASSWORD")
        self.database = os.getenv("VERTICA_DATABASE")

        self.conn = vertica_python.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            autocommit=True,
        )

        self.cells_table = config.cells_table
        self.tables_table = config.tables_table
        self.columns_table = config.columns_table

        self.table_column = config.table_column
        self.column_column = config.column_column
        self.row_column = config.row_column
        self.term_column = config.term_column
        self.term_token_column = config.term_token_column
        self.table_url_column = config.table_url_column
        self.table_title_column = config.table_title_column
        self.table_weight_column = config.table_weight_column
        self.header_column = config.header_column
        self.header_token_column = config.header_token_column 
    
    def close(self):
        self.conn.close()

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass

    def get_prefix(self, idx, cutoff):                        ###Helper Func to destinquish between X and Y cols also by Name in the Querry 
        if idx < cutoff:
            return f"X{idx}"
        else:
            return f"Y{idx - cutoff}"
        
    # TODO: why is the tokenized projection not used (can see it through EXPLAIN)???
    def find_xy_candidates(
        self,
        X_cols: list[list[str]],                 ###mandatory input cols
        Y_cols: list[list[str]] | None,          ###y-cols, only important for dirct search 
        tau: int,                                ###precision parameter
        multi_hop:bool = False,                  ###enables the multi-hop szenario, allwoing y_cols to be None
        limit:int | None = None,                 ###limits the number of tables 
        previously_seen_tables: set| None = None ### already seen tables get excluded from the search 
        ):
        if multi_hop: 
            Y_cols = list()

        if not X_cols or not Y_cols and not multi_hop:
            raise ValueError("X_cols and Y_cols must not be empty!")
        
        params = [subelem for elem in (X_cols + Y_cols) for subelem in elem]

        anz_x_cols = len(X_cols)
        anz_rows = len(next(iter(X_cols)))

        Querry = list() 
        Col_names = list()
        placeholders = ", ".join(["%s"] * anz_rows)

        for idx, elem in enumerate(X_cols + Y_cols): 

            col_name = f"{self.get_prefix(idx, anz_x_cols)}"
            col_sql = f"""{col_name} AS (
            SELECT {self.table_column}, {self.column_column}
            FROM {self.cells_table} /*+ PROJS('public.tokenized_proj') */ 
            WHERE {self.term_token_column} IN ({placeholders})
            GROUP BY {self.table_column}, {self.column_column}
            HAVING COUNT(DISTINCT {self.term_token_column}) >= {tau}
            )"""
            Col_names.append(col_name)
            Querry.append(col_sql)
        
        b_parts = list()
        equal_parts = list()
        for idx, col_name in enumerate(Col_names): 

            col_sql = f"""{col_name}.{self.column_column} AS {self.get_prefix(idx, anz_x_cols)}_column_id"""
            equal_sql = f"""{col_name} ON {Col_names[0]}.{self.table_column} = {col_name}.{self.table_column}"""
            b_parts.append(col_sql)
            if idx != 0: 
                equal_parts.append(equal_sql)
            else: 
                equal_parts.append(f"{col_name}")

        unequal_part = list()
        
        for idx, col_name in enumerate(Col_names):
            for jdx in range(idx+1, len(Col_names)): 
                unequal_sql = f"""{col_name}.{self.column_column}<>{Col_names[jdx]}.{self.column_column}"""
                unequal_part.append(unequal_sql)



        if previously_seen_tables:

            ignore_list = list(previously_seen_tables)
            ignore_placeholders = ", ".join(["%s"] * len(ignore_list))

            exclusion_sql = f"""{Col_names[0]}.{self.table_column} NOT IN ({ignore_placeholders})"""
            unequal_part.append(exclusion_sql)

            params.extend(ignore_list)


        A_part = f"""WITH \n""" + ", \n".join(Querry)
        B_part = f"""\nSELECT \n{Col_names[0]}.{self.table_column} AS table_id, \n""" + ", \n".join(b_parts)
        C_part = f""" \nFROM """ + "\nJOIN ".join(equal_parts)
        D_part = f"""\nWHERE """ + " \nAND ".join(unequal_part) if unequal_part else """"""
        sql_parts = [A_part, B_part, C_part, D_part]
        
        if limit:
            sql_parts.append(f"\nLIMIT {limit}")
            
        sql = "".join(sql_parts) + ";"

        # print(sql)
        # print(params)

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall() 
    


    def stable_row_val(self, index_list:list, X_lists:list[list[tuple]], Y_lists:list[list[tuple]], tau:int): 
        
        if not index_list:      ####Muss eigentlich bereits eine eben drüber abgefangen werden. Brauchen sowas wie nen Decorator, der bei einer leeren Liste direkt abbricht. 
            return []

        anz_rows = len(next(iter(X_lists)))
        x_len = len(X_lists)
        y_len = len(Y_lists)
        len_cols = x_len + y_len


        ex_pairs = [subelem for elem in zip(*(X_lists+Y_lists)) for subelem in elem]
        
        selects = ", ".join([f"""%s::varchar as val_{self.get_prefix(idx, x_len)}""" for idx in range(len_cols)])
        joined_selects= " UNION ALL ".join([f"SELECT {selects}"] * anz_rows)

        Table_IDs = list(set(idx[0] for idx in index_list))
        table_id_placeholders = ", ".join(["%s"] * len(Table_IDs))

        params = ex_pairs + Table_IDs

        select_str = ", ".join([f"""{self.get_prefix(0, x_len)}.{self.table_column}"""] + [f"""{self.get_prefix(idx, x_len)}.{self.column_column}""" for idx in range(len_cols)])

        from_part = f"FROM {self.cells_table} {self.get_prefix(0, x_len)}"


        joins = []
        conditions = []
        for idx in range(len_cols):

            condition = f"{self.get_prefix(idx, x_len)}.{self.term_token_column} = e.val_{self.get_prefix(idx, x_len)}"
            conditions.append(condition)

            if idx != 0: 
                join_str = f"""
                JOIN {self.cells_table} {self.get_prefix(idx, x_len)}
                    ON {self.get_prefix(0, x_len)}.{self.table_column} = {self.get_prefix(idx, x_len)}.{self.table_column} 
                    AND {self.get_prefix(0, x_len)}.{self.row_column} = {self.get_prefix(idx, x_len)}.{self.row_column}
                """
                joins.append(join_str)
        
        conditions_str = " AND ".join(conditions)


        sql = f'''
            WITH Examples AS (
                {joined_selects}
            )
            SELECT 
                {select_str},
                COUNT(*) as matches
            {from_part}
            {"".join(joins)}
            JOIN Examples e
                ON {conditions_str}
            WHERE 
                {self.get_prefix(0, x_len)}.{self.table_column} IN ({table_id_placeholders})
            GROUP BY 
                {select_str}
            HAVING 
                COUNT(*) >= {tau}
        '''

        # print(sql)
        # print(params)

        validated_results = list()
        candidates_set = set(tuple(idx) for idx in index_list)


        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            
            for row in rows:
                result_tuple = tuple(row[:-1]) 
                
                if result_tuple in candidates_set:
                    validated_results.append(result_tuple)

        return validated_results



    @cache
    def stable_get_y(self, idx: tuple, Querries: tuple): 

        ###Soll er lieber Tokens oder die originalen Werte zurückgeben? Für den Join würde ersteres natürlich mehr Sinn ergeben. Bool Val einfügen um beides zu ermöglichen? 
        # print(f"Querries: {Querries}")
        anz_cols_q = len(Querries)
        anz_rows_q = len(next(iter(Querries)))
        # print(f"Idx: {idx}, ")
 
        Table_ID = idx[0]
        X_COL_IDs = idx[1:1+anz_cols_q]
        Y_COL_IDs = idx[1+anz_cols_q:]
       

        anz_cols_answer = len(Y_COL_IDs)
        # print(f"Anz X-Col: {anz_cols_q}, Anz Y-Col: {anz_cols_answer}")
        # print(f"X-Cols: {X_COL_IDs}, Y-Cols: {Y_COL_IDs}")



        querry_pairs = [subelem for elem in zip(*(Querries)) for subelem in elem]
        
        input_selects = ", ".join([f"""%s::varchar as val_{self.get_prefix(idx, anz_cols_q)}""" for idx in range(anz_cols_q)])
        joined_input_selects= " UNION ALL ".join([f"SELECT {input_selects}"] * anz_rows_q)



        ###Das hier gibt es genau so bei Val. In eine Funktion packen und da machen lassen? 
        x_selects = [f"p.val_{self.get_prefix(idx, anz_cols_q)}" for idx in range(anz_cols_q)]
        y_selects = []
        for idx in range(anz_cols_answer):
            alias = self.get_prefix(idx + anz_cols_q, anz_cols_q)
            y_selects.append(f"{alias}.{self.term_token_column}")
            y_selects.append(f"{alias}.{self.term_column}")


        # print(y_selects)
        all_selects = ", ".join(x_selects + y_selects)





        joins = []               
        for idx, col_id in enumerate(X_COL_IDs + Y_COL_IDs):
            alias = self.get_prefix(idx, anz_cols_q)
            
            conditions = [
                f"{alias}.{self.table_column} = {Table_ID}",
                f"{alias}.{self.column_column} = {col_id}",
            ]

            if idx < anz_cols_q: 
                conditions.append(f"{alias}.{self.term_token_column} = p.val_{alias}")
            
            if 0 != idx: 
                conditions.append(f"{alias}.{self.row_column} = {self.get_prefix(0, anz_cols_q)}.{self.row_column}")
            
            join_sql = f"JOIN {self.cells_table} {alias} ON {' AND '.join(conditions)}"
            joins.append(join_sql)

        sql = f'''
        WITH Inputs ({", ".join([f"val_{self.get_prefix(idx, anz_cols_q)}" for idx in range(anz_cols_q)])}) AS (
            {joined_input_selects}
        )
        SELECT 
            {all_selects},
            COUNT(*) as freq
        FROM Inputs p
        {chr(10).join(joins)}
        GROUP BY 
            {all_selects}
        '''

        # print(sql)
        # print(querry_pairs)     ###Mache ich auch jedes mal. Decorator!!! 

        
        with self.conn.cursor() as cur:
            cur.execute(sql, querry_pairs)
            return cur.fetchall()

    def get_table_content(self, table_id: int, include_cols: tuple | list = None) -> pd.DataFrame | None:
        params = [table_id]
        where_clause = f"WHERE {self.table_column} = %s"
        
        if include_cols:
            placeholders = ', '.join(['%s'] * len(include_cols))
            where_clause += f" AND {self.column_column} IN ({placeholders})"
            params.extend(include_cols)

        sql = f"""
        SELECT 
            {self.row_column} as row_id,
            {self.column_column} as col_id,
            {self.term_token_column} as val
        FROM {self.cells_table}  /*+PROJS('public.inv_index_proj')*/
        {where_clause}
        """
        
        df_raw = pd.read_sql(sql, self.conn, params=params)
        
        if df_raw.empty:
            return None

        df_pivot = df_raw.pivot(index='row_id', columns='col_id', values='val')

        return df_pivot