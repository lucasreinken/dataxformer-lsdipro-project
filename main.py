from dotenv import load_dotenv
from src.testing import full_loop
from pathlib import Path
import pandas as pd
import io
import warnings
from src.config import MasterConfig

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")


def main():
    load_dotenv()
    base_dir = Path(__file__).resolve().parent
    csv_dir = base_dir / "data" / "tokenized_queries"

    exercises: list[list] = []

    for csv_file in csv_dir.glob("*.csv"):
        raw = csv_file.read_text(encoding="utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw), sep=",", header=0)

        stem = csv_file.stem
        if stem.endswith("_tokenized"):
            stem = stem[:-10]
        left = stem.split("2", 1)[0]
        k = left.count("_") + 1

        exercises.append([stem, df, k])

    exercises.sort(key=lambda x: len(x[1]), reverse=True)

    master_config = MasterConfig()
    full_loop(exercises, config=master_config)


if __name__ == "__main__":
    main()
