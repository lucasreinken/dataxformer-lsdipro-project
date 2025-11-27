from dataclasses import dataclass

@dataclass
class IndexingConfig:
    batch_size: int = 3000
    stop_words: None | list[str] = None
    stemmer: str = "porter"
    create_ngrams: bool = False
    ngram_size: int = 2
    min_word_len: int = 0
    read_path: str = "C:/Studium/LSDIPro/DataXFormer/data/dresden_test"


def get_default_config() -> IndexingConfig:
    """Return the default indexing configuration."""
    return IndexingConfig()
