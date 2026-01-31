from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer

from pathlib import Path
import gzip
import orjson
from typing import Iterator

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")


class NoOpStemmer:
    def stem(self, x):
        return x


class WebTableIndexer:
    def __init__(self, config):
        stemmer_map = {
            "porter": PorterStemmer,
        }

        if config.stemmer is None:
            self.stemmer = NoOpStemmer()
        else:
            if config.stemmer not in stemmer_map:
                raise ValueError(
                    f"Unknown stemmer: {config.stemmer}."
                    f"Available options: {list(stemmer_map.keys())}"
                )

            self.stemmer = stemmer_map[config.stemmer]()

        if config.stop_words:
            self.stopwords = [self.stem(stop_word) for stop_word in config.stop_words]
        else:
            self.stopwords = []

        self.min_word_len = config.min_word_len
        # self.batch_size = config.batch_size

        self.read_path = Path(config.read_path)

    def tokenize_list(self, in_list: list) -> list:
        tokenized_col = list()
        for elem in in_list:
            token = self.tokenize(elem)
            tokenized_col.append(token)

        return tokenized_col

    # @cache
    def tokenize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        if not text or not text.strip():
            return None

        words = self.process_words(text)

        stemmed_words = [
            stemmed_word
            for word in words
            if (stemmed_word := self.stem(word)) not in self.stopwords
            and len(stemmed_word) >= self.min_word_len
        ]

        if not stemmed_words:
            return None
        joined_words = " ".join(stemmed_words)
        return joined_words

    # TODO: make it configurable how to cache
    # @cache
    def process_words(self, text: str) -> list:
        """
        Processes the input text into a lowercase, punctuation free and tokenized list of strings.

        In:
            text: str

        Out:
            words: list(str)
        """
        words = text.lower().translate(str.maketrans("", "", string.punctuation))
        processed_words = word_tokenize(words)
        return processed_words

    # @cache
    def stem(self, word: str):
        """
        Returns the stemmed version of the input word.

        In:
            Word: str

        Out:
            The stemmed version of the word: str
        """
        stemmed_word = self.stemmer.stem(word)
        return stemmed_word

    def _iter_file(self, path: Path) -> Iterator[dict]:
        # TODO: Docstring (just for line-delimited JSON)
        opener = gzip.open if path.suffix == ".gz" else open

        with opener(path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                yield orjson.loads(line)

    def iter_webtables(self) -> Iterator[dict]:
        # TODO: Docstring (just for line-delimited JSON)

        if self.read_path.is_file():
            if self.read_path.suffix in {".json", ".jsonl", ".ndjson", ".gz"}:
                yield from self._iter_file(self.read_path)

        else:
            file_iter = sorted(self.read_path.glob("*"))

            for path in file_iter:
                if path.is_file() and path.suffix in {
                    ".json",
                    ".jsonl",
                    ".ndjson",
                    ".gz",
                }:
                    yield from self._iter_file(path)

    def create_dicts(
        self,
    ) -> tuple[
        dict[tuple[int, int, int], tuple[str, str]],
        dict[int, tuple[str, str, float]],
        dict[tuple[int, int], str | None],
    ]:
        # TODO: Docstring

        cells_dict: dict[tuple[int, int, int], tuple[str, str]] = {}
        columns_dict: dict[tuple[int, int], str] = {}
        tables_dict: dict[int, tuple[str, str, float]] = {}

        for table_index, web_table in enumerate(self.iter_webtables(), 1):
            table_url = web_table["url"]
            table_title = web_table["title"]
            # TODO: Add this to the config file
            table_initial_weight = 0.5
            header_index = 0

            tables_dict[table_index] = (table_url, table_title, table_initial_weight)

            if web_table["hasHeader"]:
                # TODO: how to deal with MIXED!, FIRST_COLUMN!, NONE!
                if (
                    (web_table.get("headerPosition") or "")
                    .strip()
                    .strip(string.punctuation)
                    .lower()
                ) == "first_row":
                    header_index = 1
                else:
                    print(f"Unknown headerPosition: {web_table['headerPosition']}!")

            table_data = web_table["relation"]
            for column_index, column in enumerate(table_data, 1):
                for row_index, term in enumerate(column, 1):
                    if row_index == header_index:
                        if term:
                            columns_dict[(table_index, column_index)] = term
                        else:
                            columns_dict[(table_index, column_index)] = None
                        continue
                    elif row_index == 1 and header_index == 0:
                        columns_dict[(table_index, column_index)] = None

                    term_tokenized = self.tokenize(term)

                    row_index = row_index - header_index
                    cells_dict[(table_index, column_index, row_index)] = (
                        term,
                        term_tokenized,
                    )

        return cells_dict, tables_dict, columns_dict

    def create_projections(self, dfs) -> dict[str, list[tuple[int, int, int]]]:
        # TODO: Docstring

        # Isn't the actual term also needed and we can easily do the projection in vertica???

        projections: dict[str, list[tuple[int, int, int]]] = {}

        for table_index, table_data in enumerate(dfs["tableData"]):
            for row_index, row in enumerate(table_data):
                for cell_index, cell in enumerate(row):
                    token = self.tokenize(cell.get("text"))
                    if token:
                        projections.setdefault(token, []).append(
                            (table_index, row_index, cell_index)
                        )

        return projections

    def compare_it(indexes, examples):
        table_id, xcol_id, ycol_id = indexes
