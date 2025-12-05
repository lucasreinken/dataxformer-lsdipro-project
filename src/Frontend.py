import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


import streamlit as st
import pandas as pd
from src.web_tables.indexing import WebTableIndexer
from src.config import IndexingConfig as cnf
from src.config import get_default_vertica_config
from src.database.query_factory import QueryFactory

indexer = WebTableIndexer(cnf)

st.set_page_config(page_title="Frontend", layout="wide")

if 'tables_configured' not in st.session_state:
    st.session_state.tables_configured = False
if 'training_data' not in st.session_state:
    st.session_state.training_data = {}
if 'query_data' not in st.session_state:
    st.session_state.query_data = {}
if 'num_x_cols' not in st.session_state:
    st.session_state.num_x_cols = 1
if 'num_y_cols' not in st.session_state:
    st.session_state.num_y_cols = 1
if 'num_training_rows' not in st.session_state:
    st.session_state.num_training_rows = 3
if 'num_query_rows' not in st.session_state:
    st.session_state.num_query_rows = 2
if 'data_submitted' not in st.session_state:
    st.session_state.data_submitted = False

def configure_tables():
    st.session_state.tables_configured = True
    
    st.session_state.training_data = {}
    for i in range(st.session_state.num_x_cols):
        st.session_state.training_data[f'X{i+1}'] = [''] * st.session_state.num_training_rows
    for i in range(st.session_state.num_y_cols):
        st.session_state.training_data[f'Y{i+1}'] = [''] * st.session_state.num_training_rows
    
    st.session_state.query_data = {}
    for i in range(st.session_state.num_x_cols):
        st.session_state.query_data[f'QX{i+1}'] = [''] * st.session_state.num_query_rows

def reset_tables():
    st.session_state.tables_configured = False
    st.session_state.training_data = {}
    st.session_state.query_data = {}
    st.session_state.data_submitted = False

def submit_all_data():
    training_x_lists = []
    training_y_lists = []
    query_x_lists = []
    
    for i in range(st.session_state.num_x_cols):
        col_name = f'X{i+1}'
        training_x_lists.append(st.session_state.training_data[col_name].copy())
    
    for i in range(st.session_state.num_y_cols):
        col_name = f'Y{i+1}'
        training_y_lists.append(st.session_state.training_data[col_name].copy())
    
    for i in range(st.session_state.num_x_cols):
        col_name = f'QX{i+1}'
        query_x_lists.append(st.session_state.query_data[col_name].copy())
    
    st.session_state.submitted_x = training_x_lists
    st.session_state.submitted_y = training_y_lists
    st.session_state.submitted_queries = query_x_lists
    st.session_state.data_submitted = True


st.header("Table Configuration")

col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

with col1:
    num_x_cols = st.number_input(
        "Amount of X Columns",
        min_value=1,
        max_value=10,
        value=st.session_state.num_x_cols,
        key='input_x_cols'
    )

with col2:
    num_y_cols = st.number_input(
        "Amount of Y Columns",
        min_value=1,
        max_value=10,
        value=st.session_state.num_y_cols,
        key='input_y_cols'
    )

with col3:
    num_training_rows = st.number_input(
        "Training Rows",
        min_value=1,
        max_value=100,
        value=st.session_state.num_training_rows,
        key='input_training_rows'
    )

with col4:
    num_query_rows = st.number_input(
        "Query Rows",
        min_value=1,
        max_value=100,
        value=st.session_state.num_query_rows,
        key='input_query_rows'
    )

st.write("")
tau_col1, tau_col2 = st.columns([2, 6])
with tau_col1:
    tau = st.number_input(
        "Tau Threshold",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=1.0,
        key='input_tau',
        help="Threshold value for similarity matching"
    )

with col5:
    st.write("")
    st.write("")
    if st.button("Create Tables", type="primary"):
        st.session_state.num_x_cols = num_x_cols
        st.session_state.num_y_cols = num_y_cols
        st.session_state.num_training_rows = num_training_rows
        st.session_state.num_query_rows = num_query_rows
        configure_tables()
        st.rerun()

if st.session_state.tables_configured:
    st.write("---")
    st.header("Training Data Table")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("X-Values")
        
        x_cols = st.columns(st.session_state.num_x_cols)
        for col_idx in range(st.session_state.num_x_cols):
            with x_cols[col_idx]:
                st.write(f"**X{col_idx+1}**")
                for row_idx in range(st.session_state.num_training_rows):
                    key = f'X{col_idx+1}_row{row_idx}'
                    value = st.text_input(
                        f"Row {row_idx+1}",
                        value=st.session_state.training_data[f'X{col_idx+1}'][row_idx],
                        key=key,
                        label_visibility="collapsed"
                    )
                    st.session_state.training_data[f'X{col_idx+1}'][row_idx] = value
    
    with col_right:
        st.subheader("Y-Values")
        
        y_cols = st.columns(st.session_state.num_y_cols)
        for col_idx in range(st.session_state.num_y_cols):
            with y_cols[col_idx]:
                st.write(f"**Y{col_idx+1}**")
                for row_idx in range(st.session_state.num_training_rows):
                    key = f'Y{col_idx+1}_row{row_idx}'
                    value = st.text_input(
                        f"Row {row_idx+1}",
                        value=st.session_state.training_data[f'Y{col_idx+1}'][row_idx],
                        key=key,
                        label_visibility="collapsed"
                    )
                    st.session_state.training_data[f'Y{col_idx+1}'][row_idx] = value
    
    st.write("---")
    st.header("Query Data Table")
    
    st.subheader("Query X-Values")
    
    query_x_cols = st.columns(st.session_state.num_x_cols)
    for col_idx in range(st.session_state.num_x_cols):
        with query_x_cols[col_idx]:
            st.write(f"**Query X{col_idx+1}**")
            for row_idx in range(st.session_state.num_query_rows):
                key = f'QX{col_idx+1}_row{row_idx}'
                value = st.text_input(
                    f"Query {row_idx+1}",
                    value=st.session_state.query_data[f'QX{col_idx+1}'][row_idx],
                    key=key,
                    label_visibility="collapsed"
                )
                st.session_state.query_data[f'QX{col_idx+1}'][row_idx] = value
    
    st.write("")
    st.write("---")
    
    action_col1, action_col2, action_col3 = st.columns([1, 1, 3])
    
    with action_col1:
        if st.button("Save Data", type="secondary", width="stretch"):
            submit_all_data()
            st.success("Data Saved!")
    
    with action_col2:
        if st.button("Reset Tables", width="stretch"):
            reset_tables()
            st.rerun()
    
    with action_col3:
        if st.session_state.data_submitted:
            if st.button("🚀 START PROCESSING", type="primary", width="stretch"):
                st.session_state.start_processing = True
                st.rerun()


if 'start_processing' in st.session_state and st.session_state.start_processing:
    st.session_state.start_processing = False
    
    st.write("---")
    st.header("Processing Results")
    
    with st.spinner("Processing data..."):

        ###Da das hier die "Hauptlogik" ist, diese vielleicht in die Main packen und das dann aufrufen?
        ###Dann müssen wir hier so gut wie nichts ändern und können, zum testen, immer nur den Logik befehl hier drin tauschen. 

        ##Collection of Input
        x_lists = st.session_state.submitted_x
        y_lists = st.session_state.submitted_y
        query_x_lists = st.session_state.submitted_queries
        
        ##Tokenization of Input
        cleaned_x_lists     = [indexer.tokenize_list(col) for col in x_lists]
        cleaned_y_lists     = [indexer.tokenize_list(col) for col in y_lists]
        tokenized_querries  = [indexer.tokenize_list(col) for col in query_x_lists]
        tau = st.session_state.input_tau

        ##Init of Algorithm 
        config = get_default_vertica_config()
        qf = QueryFactory(config)


        ##Logic
        print("Start 1")
        z = qf.find_xy_candidates(cleaned_x_lists, cleaned_y_lists, tau)        ###Warum werden die denn in einer unterschiedlichen Reihenfolge je nach itteration angezeigt. 
        print("Finish")                                                         ###Ist der Prozess nicht deterministisch? Liegt das am Kernel und Parallel Processing? 
        erg = qf.stable_row_val(z, cleaned_x_lists, cleaned_y_lists, tau)
        print("Done with erg")
        st.subheader("All Direct Tables")
        st.write(erg)
        answers = qf.stable_get_y(next(iter(erg)), tokenized_querries) ###Erwartet Tuple, keine Liste. Erg ist eine Liste von Tuplen! 






        ##Erster Querry für Multi-Hop. Collected alle T_x 
        
        multi_z = qf.find_xy_candidates(cleaned_x_lists, None, tau, True)
        st.subheader("Multi-Hop Tables")
        st.write(multi_z)

        erg_set = set(Index[0] for Index in erg)

        clean = list()
        for Multi_IDX in multi_z: 
            Table_ID, *_ = Multi_IDX
            if Table_ID not in erg_set: 
                clean.append(Table_ID) #####jaja, verlust von Col_ids, aber das hier ist nur ein Test 
        
        st.subheader("T_X <- T_X/T_E")
        st.write(clean)


        ####Das hier muss alles überarbeitet werden, muss in eigene Datei "Logic" oder "Pipeline"
        # EM, Multihop etc. fehlen noch hier 


        # ####Hier EM und itterativ. Multi erst hier nach? Erst starten, wenn normales nicht für alle x in Q 
        # ####ein Ergebnis gebracht hat? 



        ###Visualisation of Results
        st.subheader("Results")
            
        if answers:

            rows = []
            for idx, answer_tuple in enumerate(answers):
                if answer_tuple:  
                    row_dict = {}
                    
                    row_dict['Query #'] = idx + 1

                    tuple_data = list(answer_tuple)

                    num_x = st.session_state.num_x_cols
                    for i in range(min(num_x, len(tuple_data))):
                        row_dict[f'X{i+1}'] = tuple_data[i]

                    num_y = st.session_state.num_y_cols
                    for i in range(num_y):
                        if num_x + i < len(tuple_data):
                            row_dict[f'Y{i+1}'] = tuple_data[num_x + i]
                    
                    if len(tuple_data) > num_x + num_y:
                        row_dict['Frequency'] = tuple_data[num_x + num_y]
                    
                    rows.append(row_dict)
            
            if rows:
                df_results = pd.DataFrame(rows)
                st.dataframe(df_results, width="stretch", hide_index=True)
                
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No results to display")
        else:
            st.warning("No answers found")
        
        st.success("Processing Complete!")


###Example Bill Gates Microsoft line 21 (unser Stemmer stemmed ihn leider als bill gate, token ist aber gates) Mark Zuckerberg Facebook (passt) Steve Ballmer Microsoft 175 Kevin Mitnick Hacker 113
###Table 30,440,039

#Q: Bob Frankston? 26 Biz Stone 25

##Todo: Check if all Values are Filled in 
##Todo: Demo Mode with Predefined Values 
#streamlit run /Users/christophhalberstadt/Documents/GitHub/ProjectDataXFormer/src/Frontend.py  