from src.web_tables.indexing import WebTableIndexer
from src.web_tables.querying import WebTableQueryEngine
from src.web_tables.ranking import WebTableRanker
from src.config import (
    get_default_vertica_config,
    get_default_indexing_config,
    get_default_querying_config,
    get_default_ranking_config
    )
from src.database import VerticaClient

def main():

    X = [[
    "ord", "dfw", "atl", "mia", "bos",
    "yyz", "yvr", "yul",
    "syd", "mel", "akl"
    ],
    [
    "chicago", "dallas", "atlanta", "miami", "boston",
    "toronto", "vancouver", "montreal",
    "sydney", "melbourne", "auckland"
    ]]

    Y = [[
    "usa", "usa", "usa", "usa", "usa",
    "canada", "canada", "canada",
    "australia", "australia", "new zealand"
    ]]

    config = get_default_ranking_config()

    ranker = WebTableRanker(config)

    print(ranker.expectation_maximization(X, Y, [["led", "gva", "kei", "hel", "lis", "svp", "osl", "arn", "bru", "zrh"],
                                                 ["saint petersburg", "geneva", "kepi", "helsinki", "lisbon", "kuito", "oslo", "stockholm", "brussels", "zurich"]
                                                ]))

if __name__ == "__main__":
    main() 
