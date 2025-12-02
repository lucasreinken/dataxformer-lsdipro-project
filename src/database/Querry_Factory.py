import os
import vertica_python
from dotenv import load_dotenv

class QuerryFactory:

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
        
    def find_xy_candidates(
        self,
        X_cols: list[list[str]],
        Y_cols: list[list[str]],
        tau: int,
        ):

        if not X_cols or not Y_cols:
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
            FROM {self.cells_table}
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

        A_part = f"""WITH \n""" + ", \n".join(Querry)
        B_part = f"""\nSELECT \n{Col_names[0]}.{self.table_column} AS table_id, \n""" + ", \n".join(b_parts)
        C_part = f""" \nFROM """ + "\nJOIN ".join(equal_parts)
        D_part = f"""\nWHERE """ + " \nAND ".join(unequal_part)
        sql = A_part + B_part + C_part + D_part + ";"

        print(sql)
        print(params)

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

        select_str = ", ".join([f"""X0.{self.table_column}"""] + [f"""{self.get_prefix(idx, x_len)}.{self.column_column}""" for idx in range(len_cols)])

        from_part = f"FROM {self.cells_table} X0"


        joins = []
        conditions = []
        for idx in range(len_cols):

            condition = f"{self.get_prefix(idx, x_len)}.{self.term_token_column} = e.val_{self.get_prefix(idx, x_len)}"
            conditions.append(condition)

            if idx != 0: 
                join_str = f"""
                JOIN {self.cells_table} {self.get_prefix(idx, x_len)}
                    ON X0.{self.table_column} = {self.get_prefix(idx, x_len)}.{self.table_column} 
                    AND X0.{self.row_column} = {self.get_prefix(idx, x_len)}.{self.row_column}
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
                X0.{self.table_column} IN ({table_id_placeholders})
            GROUP BY 
                {select_str}
            HAVING 
                COUNT(*) >= {tau}
        '''

        print(sql)
        print(params)

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


    def get_y(self, idx: tuple, Querries: list): 
            """
            Returns the corresponding Y-Values for the x \in Querries as well as a Count of their Frequency
            In: 
                idx: (Table_ID, XCOL_ID, YCOL_ID)
                Querries: list[]
            Out: 
                List[Tuple(x_val, y_val, freq)]
            """

            Table_ID, XCOL_ID, YCOL_ID = idx

            Querries_placeholders = ' UNION ALL '.join(["SELECT %s"] * len(Querries))

            sql = f'''
            WITH Inputs (x_val) AS ({Querries_placeholders})
            SELECT 
                p.x_val, 
                c2.{self.term_token_column},
                COUNT(*)
            FROM Inputs p 
            JOIN {self.cells_table} c1
            ON p.x_val = c1.{self.term_token_column}
            AND c1.tableid = {Table_ID}
            AND c1.{self.column_column} = {XCOL_ID}
            JOIN {self.cells_table} c2 
            ON c1.{self.row_column} = c2.{self.row_column}
            AND c2.tableid = {Table_ID}
            AND c2.{self.column_column} = {YCOL_ID}
            GROUP BY p.x_val, c2.{self.term_token_column}
            '''
            
            with self.conn.cursor() as cur:
                cur.execute(sql, Querries)
                return cur.fetchall()


