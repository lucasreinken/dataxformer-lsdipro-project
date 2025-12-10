import itertools

from src.web_tables.indexing import WebTableIndexer

from src.database.query_factory import QueryFactory
from src.config import get_default_vertica_config

from tqdm import tqdm

import pandas as pd

class WebTableQueryEngine:
    def __init__(self, config):
        self.tau = config.tau

        vertica_config = get_default_vertica_config()
        self.querry_factory = QueryFactory(vertica_config)

    def find_columns(self, X: list[str], Y: list[str]):
        z = self.querry_factory.find_xy_candidates(X, Y, self.tau)
        erg = set(self.querry_factory.stable_row_val(z, X, Y, self.tau))

        return erg

    def find_answers(self, erg, Q: list[str]):

        len_x = len(Q)
        Q = tuple(tuple(q) for q in Q)

        # build query tuples from Q
        conditions = list(zip(*Q))

        for index in tqdm(erg, desc="Find answers in candidate tables"):
            table_df = self.querry_factory.get_table_content(index[0], index[1:])

            x_cols = list(index[1:1 + len_x])
            y_cols = list(index[1 + len_x:])

            mask = table_df.apply(
                lambda row: any(
                    all(row[x_cols[i]] == cond[i] for i in range(len_x))
                    for cond in conditions
                ),
                axis=1
            )

            answer_list = (
                table_df.loc[mask]
                        .apply(
                            lambda row: [
                                row[x_cols].tolist(),
                                row[y_cols].tolist()
                            ],
                            axis=1
                        )
                        .tolist()
            )

            yield index[0], answer_list