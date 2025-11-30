import streamlit as st
import pandas as pd

st.set_page_config(page_title="Frontend", layout="wide")

if 'table_configured' not in st.session_state:
    st.session_state.table_configured = False
if 'data' not in st.session_state:
    st.session_state.data = {}
if 'num_x_cols' not in st.session_state:
    st.session_state.num_x_cols = 2
if 'num_y_cols' not in st.session_state:
    st.session_state.num_y_cols = 2
if 'num_rows' not in st.session_state:
    st.session_state.num_rows = 5

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
            st.success("Submit Successful!")
    
    with col_reset:
        if st.button("Reset", use_container_width=False):
            reset_table()
            st.rerun()



##Todo: Check if all Values are Filled in 
##Todo: Demo Mode with Predefined Values 