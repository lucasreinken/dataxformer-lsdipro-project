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
# from src.database import VerticaClient
from src.database.Querry_Factory import QuerryFactory

indexer = WebTableIndexer(cnf)

st.set_page_config(page_title="Frontend", layout="wide")

###Example Table
if 'table_configured' not in st.session_state:
    st.session_state.table_configured = False
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'num_x_cols' not in st.session_state:
    st.session_state.num_x_cols = 1
if 'num_y_cols' not in st.session_state:
    st.session_state.num_y_cols = 1
if 'num_rows' not in st.session_state:
    st.session_state.num_rows = 3

###Query Table
if 'query_table_configured' not in st.session_state:
    st.session_state.query_table_configured = False
if 'query_data' not in st.session_state:
    st.session_state.query_data = {}
if 'num_query_x_cols' not in st.session_state:
    st.session_state.num_query_x_cols = 1
if 'num_query_rows' not in st.session_state:
    st.session_state.num_query_rows = 2


###Example Table 
def configure_table():
    """Create Table"""
    st.session_state.table_configured = True

    st.session_state.data = {}
    for i in range(st.session_state.num_x_cols):
        st.session_state.data[f'X{i+1}'] = [''] * st.session_state.num_rows
    for i in range(st.session_state.num_y_cols):
        st.session_state.data[f'Y{i+1}'] = [''] * st.session_state.num_rows

def reset_table():
    """Reset Table"""
    st.session_state.table_configured = False
    st.session_state.data = {}

def submit_data():
    """Submit the Data"""
    x_lists = []
    y_lists = []
    

    for i in range(st.session_state.num_x_cols):
        col_name = f'X{i+1}'
        x_lists.append(st.session_state.data[col_name].copy())
    

    for i in range(st.session_state.num_y_cols):
        col_name = f'Y{i+1}'
        y_lists.append(st.session_state.data[col_name].copy())
    
    st.session_state.submitted_x = x_lists
    st.session_state.submitted_y = y_lists
    st.session_state.data_submitted = True
    
    return x_lists, y_lists

def configure_query_table():
    """Create Query Table"""
    st.session_state.query_table_configured = True

    st.session_state.query_data = {}
    for i in range(st.session_state.num_query_x_cols):
        st.session_state.query_data[f'QX{i+1}'] = [''] * st.session_state.num_query_rows

def reset_query_table():
    """Reset Query Table"""
    st.session_state.query_table_configured = False
    st.session_state.query_data = {}

def submit_query_data():
    """Submit the Query Data"""
    query_x_lists = []
    
    for i in range(st.session_state.num_query_x_cols):
        col_name = f'QX{i+1}'
        query_x_lists.append(st.session_state.query_data[col_name].copy())
    
    st.session_state.submitted_queries = query_x_lists
    st.session_state.query_data_submitted = True
    
    return query_x_lists


st.header("Configuration of Submission Table")

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

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
    num_rows = st.number_input(
        "Amount of Rows",
        min_value=1,
        max_value=100,
        value=st.session_state.num_rows,
        key='input_rows'
    )

st.write("") 
tau_col1, tau_col2, tau_col3 = st.columns([2, 2, 4])
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

with col4:
    st.write("")  
    st.write("")  
    if st.button("Create Table", type="primary"):
        st.session_state.num_x_cols = num_x_cols
        st.session_state.num_y_cols = num_y_cols
        st.session_state.num_rows = num_rows
        configure_table()
        st.rerun()

if st.session_state.table_configured:
    st.header("Datasubmission Table")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("X-Values")
        
        x_cols = st.columns(st.session_state.num_x_cols)
        for col_idx in range(st.session_state.num_x_cols):
            with x_cols[col_idx]:
                st.write(f"**X{col_idx+1}**")
                for row_idx in range(st.session_state.num_rows):
                    key = f'X{col_idx+1}_row{row_idx}'
                    value = st.text_input(
                        f"Zeile {row_idx+1}",
                        value=st.session_state.data[f'X{col_idx+1}'][row_idx],
                        key=key,
                        label_visibility="collapsed"
                    )
                    st.session_state.data[f'X{col_idx+1}'][row_idx] = value
    
    with col_right:
        st.subheader("Y-Values")
        
        y_cols = st.columns(st.session_state.num_y_cols)
        for col_idx in range(st.session_state.num_y_cols):
            with y_cols[col_idx]:
                st.write(f"**Y{col_idx+1}**")
                for row_idx in range(st.session_state.num_rows):
                    key = f'Y{col_idx+1}_row{row_idx}'
                    value = st.text_input(
                        f"Zeile {row_idx+1}",
                        value=st.session_state.data[f'Y{col_idx+1}'][row_idx],
                        key=key,
                        label_visibility="collapsed"
                    )
                    st.session_state.data[f'Y{col_idx+1}'][row_idx] = value
    
    st.write("")
    col_submit, col_reset = st.columns([1, 4])
    
    with col_submit:
        if st.button("Save Training Data", type="secondary", use_container_width=True):
            x_lists, y_lists = submit_data()
            st.success("Training Data Saved!")
        
    with col_reset:
        if st.button("Reset Training Table", use_container_width=False):
            reset_table()
            st.rerun()


###Querry Table 
st.write("---")
st.header("Configuration of Query Table")

qcol1, qcol2, qcol3 = st.columns([2, 2, 4])

with qcol1:
    num_query_x_cols = st.number_input(
        "Amount of Query X Columns",
        min_value=1,
        max_value=10,
        value=st.session_state.num_query_x_cols,
        key='input_query_x_cols'
    )

with qcol2:
    num_query_rows = st.number_input(
        "Amount of Query Examples",
        min_value=1,
        max_value=100,
        value=st.session_state.num_query_rows,
        key='input_query_rows'
    )

with qcol3:
    st.write("")  
    st.write("")  
    if st.button("Create Query Table", type="primary"):
        st.session_state.num_query_x_cols = num_query_x_cols
        st.session_state.num_query_rows = num_query_rows
        configure_query_table()
        st.rerun()

if st.session_state.query_table_configured:
    st.header("Query Input Table")
    
    st.subheader("Query X-Values")
    
    query_x_cols = st.columns(st.session_state.num_query_x_cols)
    for col_idx in range(st.session_state.num_query_x_cols):
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
    qcol_submit, qcol_reset = st.columns([1, 4])
    
    with qcol_submit:
        if st.button("Submit Queries", type="primary", use_container_width=True):
            query_x_lists = submit_query_data()
            st.success("Queries Submitted Successfully!")
    
    with qcol_reset:
        if st.button("Reset Query Table", use_container_width=False):
            reset_query_table()
            st.rerun()




###Logic 
if 'data_submitted' in st.session_state and st.session_state.data_submitted and \
   'query_data_submitted' in st.session_state and st.session_state.query_data_submitted:
    
    st.write("---")
    st.header("Processing Results")
    
    with st.spinner("Processing data..."):


        ###Da das hier die "Hauptlogik" ist, diese vielleicht in die Main packen und das dann aufrufen?
        ###Dann müssen wir hier so gut wie nichts ändern und können, zum testen, immer nur den Logik befehl hier drin tauschen. 

        x_lists = st.session_state.submitted_x
        y_lists = st.session_state.submitted_y
        query_x_lists = st.session_state.submitted_queries
        

        cleaned_x_lists     = [indexer.tokenize_list(col) for col in x_lists]
        cleaned_y_lists     = [indexer.tokenize_list(col) for col in y_lists]
        tokenized_querries  = [indexer.tokenize_list(col) for col in query_x_lists]
        tau = st.session_state.input_tau


        config = get_default_vertica_config()

        qf = QuerryFactory(config)
        print("Start 1")
        z = qf.find_xy_candidates(cleaned_x_lists, cleaned_y_lists, tau)
        print("Finish")
        erg = qf.stable_row_val(z, cleaned_x_lists, cleaned_y_lists, tau)
        print("Done with erg")
        

        st.write(erg)



        ##
        ##Example with matching Tokens: Table 30440039 Mark Zuckerberg Facebook Line 128, Biz Stone Twitter Line 25 
        ####Das hier muss alles überarbeitet werden, EM, Multihop etc. 
        

        # Q = next(iter(tokenized_querries))


        # ####Hier EM und itterativ. Multi erst hier nach? Erst starten, wenn normales nicht für alle x in Q 
        # ####ein Ergebnis gebracht hat? 
        # print(erg[3])
        # for i in range(5): 
        #     st.write(erg[i])
        #     st.write(qf.get_y(erg[i], Q))


        # answers = qf.get_y(erg[1], Q)





        # ###Visualisation of Results
        # st.subheader("Results")
            
        # if answers:

        #     rows = []
        #     for idx, answer_tuple in enumerate(answers):
        #         if answer_tuple:  
        #             row_dict = {}
                    
        #             row_dict['Query #'] = idx + 1

        #             tuple_data = list(answer_tuple)

        #             num_x = st.session_state.num_x_cols
        #             for i in range(min(num_x, len(tuple_data))):
        #                 row_dict[f'X{i+1}'] = tuple_data[i]

        #             num_y = st.session_state.num_y_cols
        #             for i in range(num_y):
        #                 if num_x + i < len(tuple_data):
        #                     row_dict[f'Y{i+1}'] = tuple_data[num_x + i]
                    
        #             if len(tuple_data) > num_x + num_y:
        #                 row_dict['Frequency'] = tuple_data[num_x + num_y]
                    
        #             rows.append(row_dict)
            
        #     if rows:
        #         df_results = pd.DataFrame(rows)
        #         st.dataframe(df_results, use_container_width=True, hide_index=True)
                
        #         csv = df_results.to_csv(index=False)
        #         st.download_button(
        #             label="Downlaod as CSV",
        #             data=csv,
        #             file_name="query_results.csv",
        #             mime="text/csv"
        #         )
        #     else:
        #         st.warning("No results to display")
        # else:
        #     st.warning("No answers found")
        
        # st.success("Processing Complete!")


###Example Bill Gates Microsoft line 21 (unser Stemmer stemmed ihn leider als bill gate, token ist aber gates) Mark Zuckerberg Facebook (passt) Steve Ballmer Microsoft 175 Kevin Mitnick Hacker 113
###Table 30,440,039

##Todo: Check if all Values are Filled in 
##Todo: Demo Mode with Predefined Values 
#streamlit run /Users/christophhalberstadt/Documents/GitHub/ProjectDataXFormer/src/Frontend.py  