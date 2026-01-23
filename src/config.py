from dataclasses import dataclass, field


@dataclass
class IndexingConfig:
    batch_size: int = 3000
    stop_words: list[str] | None = field(
        default_factory=lambda: [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "for",
            "of",
            "to",
            "is",
            "are",
            "was",
            "be",
            "it",
            "this",
            "that",
            "no, as",
            "there",
        ]
    )
    stemmer: str | None = None
    create_ngrams: bool = False
    ngram_size: int = 2
    min_word_len: int = 0
    read_path: str = "C:/Studium/LSDIPro/DataXFormer/data/dresden_test"


@dataclass
class QueryingConfig:
    tau = 2


@dataclass
class RankingConfig:
    epsilon = 0.001
    alpha = 0.99
    max_iterations = float("inf")
    table_prior = 0.5
    topk = 1


@dataclass
class VerticaConfig:
    cells_table = "main_tokenized"
    tables_table = None
    columns_table = "columns_tokenized"
    table_column = "tableid"
    column_column = "colid"
    row_column = "rowid"
    term_column = "term"
    term_token_column = "tokenized"
    table_url_column = None
    table_title_column = None
    table_weight_column = None
    header_column = "header"
    header_token_column = "header_tokenized"


@dataclass
class TestingConfig:
    project_name = "DataXFormerTest"
    entity = "DataXFormer"
    repeats: int = 5
    k: int = 5  ##topk
    max_workers: int = 4
    return_time: bool = True


def get_default_indexing_config() -> IndexingConfig:
    return IndexingConfig()


def get_default_ranking_config() -> RankingConfig:
    return RankingConfig()


def get_default_querying_config() -> QueryingConfig:
    return QueryingConfig()


def get_default_vertica_config() -> VerticaConfig:
    return VerticaConfig()


def get_default_testing_config() -> TestingConfig:
    return TestingConfig()
