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

    slow_order = [
        "Country2Area",
        "CountryCode2Country",
        "State2Abbreviation",
        "Ticker2Company",
        "Company2Ticker",
        "City2Country",
        "Country2ThreeLettersISOCode",
        "Country2TwoLettersISOCode",
        "Movie2Year",
        "Country2Adjective",
        "Airport2Country",
        "Element2Symbol",
    ]

    order_index = {name: i for i, name in enumerate(slow_order)}

    exercises.sort(key=lambda x: order_index.get(x[0], float("inf")))

    tau_list = [1, 2, 3, 4, 5]

    for tau in tau_list:
        master_config = MasterConfig()
        master_config.querying.tau = tau
        full_loop(
            exercises, config=master_config, experiment_name=f"tau->{tau}"
        )


if __name__ == "__main__":
    main()
