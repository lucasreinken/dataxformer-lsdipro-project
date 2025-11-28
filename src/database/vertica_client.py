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

        # limit amount of scanning (just for reason)
        sql = f'''
        WITH colX AS (
            SELECT {self.table_column}, {self.column_column}
            FROM {self.cells_table}
            WHERE {self.term_token_column} IN ({x_placeholders})
            GROUP BY {self.table_column}, {self.column_column}
            HAVING COUNT(DISTINCT {self.term_token_column}) >= %s
        ),
        colY AS (
            SELECT {self.table_column}, {self.column_column}
            FROM {self.cells_table}
            WHERE {self.term_token_column} IN ({y_placeholders})
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

    def close(self):
        self.conn.close()

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass
