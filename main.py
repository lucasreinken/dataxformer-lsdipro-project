from src.web_tables.indexing import WebTableIndexer
from src.web_tables.ranking import WebTableRanker
from src.config import (
    get_default_vertica_config,
    get_default_indexing_config,
    get_default_ranking_config
    )
from src.database import VerticaClient

def main():

    X = [
    "ord", "dfw", "atl", "mia", "bos",
    "yyz", "yvr", "yul",
    "syd", "mel", "akl"
    ]

    Y = [
    "chicago", "dallas", "atlanta", "miami", "boston",
    "toronto", "vancouver", "montreal",
    "sydney", "melbourne", "auckland"
    ]

    tau = 2

    config = get_default_ranking_config()

    ranker = WebTableRanker(config)

    print(ranker.expectation_maximization(X, Y, tau, ["led", "gva", "kei", "hel", "lis", "svp", "osl", "arn", "bru", "zrh"]))

if __name__ == "__main__":
    main() 
