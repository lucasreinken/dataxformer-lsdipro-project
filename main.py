from src.web_tables.indexing import WebTableIndexer
from src.config import get_default_config

def main():
    
    config = get_default_config()
    indexer = WebTableIndexer(config)

    cells_dict, tables_dict, columns_dict = indexer.create_dicts()

    # print(cells_dict)
    # print(tables_dict)
    # print(columns_dict)

if __name__ == "__main__":
    main()
