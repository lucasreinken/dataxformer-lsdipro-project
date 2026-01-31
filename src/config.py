from dataclasses import dataclass, field
from typing import Optional


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
class IndexingConfig:
    batch_size: int = 3000
    stop_words: list[str] | None = field(
        default_factory=lambda: [
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "but",
            "by",
            "for",
            "if",
            "in",
            "into",
            "is",
            "it",
            "no",
            "not",
            "of",
            "on",
            "or",
            "such",
            "that",
            "the",
            "their",
            "then",
            "there",
            "these",
            "they",
            "this",
            "to",
            "was",
            "will",
            "with",
        ]
    )
    stemmer: Optional[str] = None
    read_path: str = "./data/dresden_test"


@dataclass
class RankerConfig:
    epsilon: float = 0.01
    alpha: float = 0.99
    table_prior: float = 0.5
    topk: int = 1
    parallel_workers = 4
    max_requery_iterations: int = 2
    max_requery_answers: int = 50
    use_fuzzy_matching: bool = True
    use_majority_voting: bool = False
    fuzzy_scorer: str = "max3"
    fuzzy_threshold: float = 0.95
    max_iterations: int = float("inf")
    print_iterations: bool = False
    debug_timing: bool = False


@dataclass
class QueryConfig:
    tau: int = 2
    table_limit: int = 100
    use_multi_hop: bool = False
    print_query: bool = False


@dataclass
class ExperimentConfig:
    project_name: str = "DataXFormerTest"
    entity: str = "DataXFormer"
    repeats: int = 5
    parallel_runs: int = 4
    count_of_examples: int = 5
    return_time: bool = True
    seed: int = 2
    preindexed: bool = True
    print_evaluation: bool = False


@dataclass
class MultiHopConfig:
    """Configuration for multi-hop table joining."""

    max_path_len: int = 3
    max_tables: int = 25
    adaptive_limits: bool = True

    fd_threshold: float = 0.95
    error_threshold_for_fd: float = 0.05
    use_parallel_fd: bool = True
    max_workers_multi: int = 4

    auto_detect_numeric: bool = True
    restrict_nums: bool = True
    metric_precision: float = 0.1
    none_precision: float = 0.1
    min_word_len: int = 2
    strict_uniqueness_in_df: bool = False

    use_fuzzy_y_match: bool = False
    fuzzy_scorer_multi: str = "ratio"
    fuzzy_threshold_multi: float = 0.95

    large_df_threshold: int = 100000
    sample_size: int = 5000
    max_paths_to_keep: int = 20

    adaptive_reduction_factor: int = 2
    overlap_threshold: float = 0.5
    threshold_for_numeric_cols: float = 0.5

    print_further_information: bool = False
    reset_counters: bool = True


@dataclass
class MasterConfig:
    vertica: VerticaConfig = field(default_factory=VerticaConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    querying: QueryConfig = field(default_factory=QueryConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    multi_hop: MultiHopConfig = field(default_factory=MultiHopConfig)
