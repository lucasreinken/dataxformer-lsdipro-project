import itertools

from src.web_tables.indexing import WebTableIndexer

from src.database.query_factory import QueryFactory
from src.config import get_default_vertica_config

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

        print(erg)

        len_x = len(Q)
        Q = tuple(tuple(q) for q in Q)

        print(erg)

        for index in erg:
            answer_list = self.querry_factory.stable_get_y(index, Q)
            if answer_list:
                answer_list = [
                    [
                        answer[:len_x],
                        answer[len_x:-1][::2],
                        answer[len_x:-1][1::2],
                    ]
                    for answer in answer_list
                ]
                yield index[0], answer_list