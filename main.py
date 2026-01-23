from dotenv import load_dotenv
from src.testing import full_loop
from pathlib import Path
import pandas as pd
import io

import warnings

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")


def main():
    load_dotenv()
    base_dir = Path(__file__).resolve().parent
    csv_dir = base_dir / "data" / "DataXFormer-Queries"

    exercises: list[list] = []

    for csv_file in csv_dir.glob("*.csv"):
        raw = csv_file.read_text(encoding="utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw), sep=",", header=0)

        stem = csv_file.stem
        left = stem.split("2", 1)[0]
        k = left.count("_") + 1

        exercises.append([stem, df, k])

    full_loop(exercises, return_time=True)


if __name__ == "__main__":
    main()
