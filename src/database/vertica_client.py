import os
import vertica_python
from dotenv import load_dotenv

class VerticaClient:
    # TODO: DOCSTRING

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

    # TODO: Adapt it to be able to handle multiple Xs
    # TODO: Add assertion errors
    # TODO: Rename the function
    # TODO: Check if we need to check if same x and y row
    def find_xy_candidates(
        self,
        X: list[str],
        Y: list[str],
        tau: int,
    ):
        # TODO: Docstring
        if not X or not Y:
            return None

        # TODO: create helper functions for this (also execute...???)
        x_placeholders = ", ".join(["%s"] * len(X))
        y_placeholders = ", ".join(["%s"] * len(Y))

        # limit amount of scanning (just for reason).  Nur die ersten 1000000, später raus
        sql = f'''
        WITH colX AS (
            SELECT {self.table_column}, {self.column_column}
            FROM {self.cells_table}
            WHERE {self.term_token_column} IN ({x_placeholders})
              AND tableid < 10000000
            GROUP BY {self.table_column}, {self.column_column}
            HAVING COUNT(DISTINCT {self.term_token_column}) >= %s
        ),
        colY AS (
            SELECT {self.table_column}, {self.column_column}
            FROM {self.cells_table}
            WHERE {self.term_token_column} IN ({y_placeholders})
              AND tableid < 10000000
            GROUP BY {self.table_column}, {self.column_column}
            HAVING COUNT(DISTINCT {self.term_token_column}) >= %s
        )
        SELECT
            colX.{self.table_column} AS table_id,
            colX.{self.column_column} AS x_column_id,
            colY.{self.column_column} AS y_column_id
        FROM colX
        JOIN colY
        ON colX.{self.table_column} = colY.{self.table_column}
        AND colX.{self.column_column} <> colY.{self.column_column};
        '''

        print(sql)

 
        params = X + [tau] + Y + [tau]

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall() 
        

        #######Sollte eine Liste von Tuplen die jeweils immer Table_ID, XCol_ID, YCol_ID enthalten, zurückgeben. 

    def close(self):
        self.conn.close()

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass

    def row_validation(self, index_list:list, X:list, Y:list, tau:int): 
        
        examples = ' UNION ALL '.join(["SELECT %s, %s"] * len(X)) 
        params = [j for i in zip(X, Y) for j in i] + [tau]

        validated_results = list()

        for idx in index_list: 
            Table_ID, XCOL_ID, YCOL_ID = idx
            sql = f''' 
            WITH input_pairs (val_x, val_y) AS ({examples})
            SELECT (

                SELECT COUNT(*)
                FROM input_pairs p

                JOIN {self.cells_table} c1
                ON p.val_x = c1.{self.term_token_column}
                AND c1.tableid = {Table_ID}
                AND c1.{self.column_column} = {XCOL_ID}

                JOIN {self.cells_table} c2
                ON p.val_y = c2.{self.term_token_column}
                AND c2.tableid = {Table_ID}
                AND c2.{self.column_column} = {YCOL_ID}
                AND c1.{self.row_column} = c2.{self.row_column}
            
            ) >= %s
            '''

            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                match =  cur.fetchone()[0]
                if match: 
                    validated_results.append(idx)
        
        return validated_results
    #######Hier nach müsste dann Expectation Maximization kommen und die beste(n) ausgewählt werden. 
        
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
            
#TODO: Mehrere X und Y Vals 


######Was wir haben wäre grobes Table suchen, Validieren und extracten. 
######Da "Inductive Approach", siehe knapp über III C. müssten wir noch sowas wie "Num_It_Steps" oder so festlegen oder das
#####Ganze als WHILE machen, bis es konvergiert. 
#####Konnte das hier wegen den Connection Issues nicht Testen, Edu funktioniert grade nicht und mit Cisco lädt nichtmal Google
#####Sonst sollte das hier fast alles sein, was wir schon vorher hatten. 