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
from src.database import VerticaClient

indexer = WebTableIndexer(cnf)

st.set_page_config(page_title="Frontend", layout="wide")

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

# NEU: Tau-Eingabefeld hinzufügen
st.write("")  # Spacing
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
        if st.button("Submit", type="primary", use_container_width=True):
            x_lists, y_lists = submit_data()







            ###Das hier ist alles, was so auch im Main wäre. Das hier ist basically unsere neue Main. 
            cleaned_x_lists = [indexer.tokenize_list(col) for col in x_lists]
            cleaned_y_lists = [indexer.tokenize_list(col) for col in y_lists]
            tau = st.session_state.input_tau

            config = get_default_vertica_config()

            vertica_client = VerticaClient(config)

            X = next(iter(cleaned_x_lists)) ####Nötig, da es ja eine Liste in einer Liste ist. Wenn der Multicolumn Input kann
                                            ####Sollte das hier nicht mehr nötig sein. 
            Y = next(iter(cleaned_y_lists))
            z = vertica_client.find_xy_candidates(X, Y, tau)
            erg = vertica_client.row_validation(z, X, Y, tau)
            st.write("Found Tables") 
            #st.write(erg)
            #print(type(erg))
            #print(erg[0]) 
            Querries = ["Istanbul", "Madrid"]   ####Muss auch noch ins Frontend 
            tokenized_querries = indexer.tokenize_list(Querries)
            list_of_answers = list()
            for i in range(5): #####Das hier statt EM, da müsste eignetlich die Funktion greifen. 
                answers = vertica_client.get_y(erg[i], tokenized_querries)
                list_of_answers.append(answers)
            st.write(list_of_answers)   ####Darstellung muss überarbeitet werden. 







            st.success("Submit Successful!")
    
    with col_reset:
        if st.button("Reset", use_container_width=False):
            reset_table()
            st.rerun()



##Todo: Check if all Values are Filled in 
##Todo: Demo Mode with Predefined Values 
#streamlit run /Users/christophhalberstadt/Documents/GitHub/ProjectDataXFormer/src/Frontend.py  