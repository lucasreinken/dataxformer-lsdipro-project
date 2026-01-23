import os
import vertica_python
import pandas as pd
from dotenv import load_dotenv


class QueryFactory:
    def __init__(self, config, logger=None) -> None:
        load_dotenv()

        self.host = os.getenv("VERTICA_HOST")
        self.port = os.getenv("VERTICA_PORT")
        self.user = os.getenv("VERTICA_USER")
        self.password = os.getenv("VERTICA_PASSWORD")
        self.database = os.getenv("VERTICA_DATABASE")

        self.conn = None

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

        self.logger = logger if logger else None

    def __enter__(self):
        self.conn = vertica_python.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            autocommit=True,
            tlsmode="disable",
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if self.logger:
            self.logger.info("Closed Vertica connection.")

    def close(self):
        self.conn.close()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def get_prefix(self, idx, cutoff):
        """
        Helper func to destinquish between x and y cols also by name in the query.

        Args:
            idx: int
            cutoff: int

        Returns:
            str
        """
        if idx < cutoff:
            return f"X{idx}"
        else:
            return f"Y{idx - cutoff}"

    def find_xy_candidates(
        self,
        x_cols: list[list[str]],
        y_cols: list[list[str]] | None,
        tau: int,
        multi_hop: bool = False,
        limit: int | None = 100,  # None,
        previously_seen_tables: set | None = None,
        print_query: bool = False,
    ):
        """
        Finds candidate tables that contain the given X and Y columns with at least tau matching values.

        Args:
            x_cols: list[list[str]]
            y_cols: list[list[str]] | None
            tau: int
            multi_hop: bool = False
            limit: int | None = None
            previously_seen_tables: set | None = None
            print_query: bool = False

        Returns:
            list of tuples: Each tuple contains (table_id, x_col_id1, x_col_id2, ..., y_col_id1, y_col_id2, ...)

        """
        if multi_hop:
            y_cols = list()

        if not x_cols or (not y_cols and not multi_hop):
            raise ValueError(
                "x_cols must not be empty and y_cols must not be empty unless multi_hop is True"
            )

        params = [subelem for elem in (x_cols + y_cols) for subelem in elem]

        x_col_count = len(x_cols)
        row_count = len(next(iter(x_cols)))

        query = list()
        col_names = list()
        placeholders = ", ".join(["%s"] * row_count)

        for idx, elem in enumerate(x_cols + y_cols):
            col_name = f"{self.get_prefix(idx, x_col_count)}"
            col_sql = f"""{col_name} AS (
            SELECT {self.table_column}, {self.column_column}
            FROM {self.cells_table} /*+ PROJS('public.tokenized_proj') */ 
            WHERE {self.term_token_column} IN ({placeholders})
            GROUP BY {self.table_column}, {self.column_column}
            HAVING COUNT(DISTINCT {self.term_token_column}) >= {tau}
            )"""
            col_names.append(col_name)
            query.append(col_sql)

        b_parts = list()
        equal_parts = list()
        for idx, col_name in enumerate(col_names):
            col_sql = f"""{col_name}.{self.column_column} AS {self.get_prefix(idx, x_col_count)}_column_id"""
            equal_sql = f"""{col_name} ON {col_names[0]}.{self.table_column} = {col_name}.{self.table_column}"""
            b_parts.append(col_sql)
            if idx != 0:
                equal_parts.append(equal_sql)
            else:
                equal_parts.append(f"{col_name}")

        unequal_part = list()

        for idx, col_name in enumerate(col_names):
            for jdx in range(idx + 1, len(col_names)):
                unequal_sql = f"""{col_name}.{self.column_column}<>{col_names[jdx]}.{self.column_column}"""
                unequal_part.append(unequal_sql)

        if previously_seen_tables:
            ignore_list = list(previously_seen_tables)
            ignore_placeholders = ", ".join(["%s"] * len(ignore_list))

            exclusion_sql = (
                f"""{col_names[0]}.{self.table_column} NOT IN ({ignore_placeholders})"""
            )
            unequal_part.append(exclusion_sql)

            params.extend(ignore_list)

        a_part = """WITH \n""" + ", \n".join(query)
        b_part = (
            f"""\nSELECT \n{col_names[0]}.{self.table_column} AS table_id, \n"""
            + ", \n".join(b_parts)
        )
        c_part = """ \nFROM """ + "\nJOIN ".join(equal_parts)
        d_part = (
            """\nWHERE """ + " \nAND ".join(unequal_part) if unequal_part else """"""
        )
        sql_parts = [a_part, b_part, c_part, d_part]

        if limit:
            sql_parts.append(f"\nLIMIT {limit}")

        sql = "".join(sql_parts) + ";"

        if print_query:
            print(sql)
            print(params)

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def stable_row_val(
        self,
        index_list: list,
        x_lists: list[list[tuple]],
        y_lists: list[list[tuple]],
        tau: int,
        print_query: bool = False,
    ):
        """
        Validates if the x and y of a given example are not only in the same DataFrame in their respective columns,
        but also in the same row, for all examples.

        Args:
            index_list: list
            x_lists: list[list[tuple]]
            y_lists: list[list[tuple]]
            tau: int
            print_query: bool = False

        Returns:
            validated_results: list[tuple]
        """

        if not index_list:
            return list()

        row_count = len(next(iter(x_lists)))
        x_len = len(x_lists)
        y_len = len(y_lists)
        len_cols = x_len + y_len

        ex_pairs = [subelem for elem in zip(*(x_lists + y_lists)) for subelem in elem]

        selects = ", ".join(
            [
                f"""%s::varchar as val_{self.get_prefix(idx, x_len)}"""
                for idx in range(len_cols)
            ]
        )
        joined_selects = " UNION ALL ".join([f"SELECT {selects}"] * row_count)

        table_ids = list(set(idx[0] for idx in index_list))
        table_id_placeholders = ", ".join(["%s"] * len(table_ids))

        params = ex_pairs + table_ids

        select_str = ", ".join(
            [f"""{self.get_prefix(0, x_len)}.{self.table_column}"""]
            + [
                f"""{self.get_prefix(idx, x_len)}.{self.column_column}"""
                for idx in range(len_cols)
            ]
        )

        from_part = f"FROM {self.cells_table} {self.get_prefix(0, x_len)}"

        joins = list()
        conditions = list()
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

        sql = f"""
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
        """

        if print_query:
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

    def get_table_content(
        self,
        table_id: int,
        include_cols: tuple | list | None = None,
        print_query: bool = False,
    ) -> pd.DataFrame | None:
        """
        Returns the content of a table as a pandas DataFrame.
        If include_cols are specified, only those will be returned.

        Args:
            table_id: int
            include_cols: tuple | list | None = None
            print_query: bool = False

        Returns:
            df_pivot: pd.DataFrame | None
        """
        params = [table_id]
        where_clause = f"WHERE {self.table_column} = %s"

        if include_cols:
            placeholders = ", ".join(["%s"] * len(include_cols))
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

        if print_query:
            print(sql)
            print(params)

        df_raw = pd.read_sql(sql, self.conn, params=params)

        if df_raw.empty:
            return None

        df_pivot = df_raw.pivot(index="row_id", columns="col_id", values="val")

        return df_pivot
