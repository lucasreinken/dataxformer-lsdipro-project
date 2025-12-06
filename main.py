from src.web_tables.indexing import WebTableIndexer
from src.config import (
    get_default_vertica_config,
    get_default_indexing_config
    )
from src.database import VerticaClient

def main():
    
    config = get_default_indexing_config()
    indexer = WebTableIndexer(config)

    # cells_dict, tables_dict, columns_dict = indexer.create_dicts()

    # print(cells_dict)
    # print(tables_dict)
    # print(columns_dict)

    print(indexer.tokenize("the Berlin"))

    X = ["ber"]
    Y = ["berlin"]
    tau = 1

    config = get_default_vertica_config()

    vertica_client = VerticaClient(config)

    # 2x
    # WHERE {self.term_token_column} IN ({x_placeholders})  
    #   AND tableid < 1000000

    print(vertica_client.find_xy_candidates(X, Y, tau))

if __name__ == "__main__":
    main() 
