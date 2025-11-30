from indexing import WebTableIndexer

def preprocess_examples(X:list, Y:list): ###Muss ich dann von vornhinein X und Y überhaupt trennen? Dann kann ich mir alle Cols auch so geben lassen. 
                                        ###Vielleicht könnte man auch nur eine Liste übergeben, mit stride Values wie 
                                        ###anz_x, anz_y, anz_zeilen und darüber mit np.strided oder so die sache hier einfacher machen
                                        ### Bzw. sequential Loading unterstüten. Ich bräuchte, wenn ich x und y trennen möchte
                                        ### sogar nur anz_zeilen bzw anz_x + anz_y 
                                        ### Dann müssten die Tabelle auch als Column Store behandelt werden, dann brauch 
                                        ### Ich nur Anz_Zeilen und dann kann ich fröhlich da durch itterieren. 

    indexer = WebTableIndexer() ###Wie muss ich den nun Configurieren?
    anz_x = len(X)
    anz_y = len(Y)
    tokenized_cols = list()
    for col in (X+Y): 
        tokenized_col = list()
        for elem in col: 
            token = indexer.process_words(elem)
            tokenized_col.append(token)
        tokenized_cols.append(tokenized_col)
    

if __name__ == "__main__": 
    print("Hi")