import sys
import os
import ast
import io

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


import streamlit as st
import pandas as pd
from src.web_tables.indexing import WebTableIndexer
from src.web_tables.ranking import WebTableRanker
from src.config import IndexingConfig as cnf
from src.config import (
    get_default_vertica_config,
    get_default_ranking_config
)
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
if 'results_available' not in st.session_state:
    st.session_state.results_available = False

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

    st.session_state.results_available = False
    
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

mode = st.radio(
    "Configuration mode",
    ["Manual", "CSV upload"],
    horizontal=True,
)

if mode == "Manual":

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

elif mode == "CSV upload":
        
    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        accept_multiple_files=False,
        help="Upload a comma- or semicolon-separated CSV file with missing values"
    )

    has_header = st.checkbox(
        "First row contains column names",
        value=True,
        help="Disable if your file has no header row (columns will be numbered)",
        key="csv_has_header",
    )

    x_selected = []
    y_selected = []

    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        sample = raw[:2000]
        delimiter = "," if sample.count(",") > sample.count(";") else ";"

        uploaded_df = pd.read_csv(
            io.StringIO(raw),
            delimiter=delimiter,
            header=0 if has_header else None,
        )

        st.session_state["uploaded_df"] = uploaded_df

        if has_header:
            all_cols = list(uploaded_df.columns)
            display_options = all_cols
        else:
            all_cols = list(range(uploaded_df.shape[1]))
            display_options = [i for i in all_cols]

        x_selected = st.session_state.get("csv_x_cols", [])
        y_selected = st.session_state.get("csv_y_cols", [])

        x_options = [o for o in display_options if o not in y_selected]

        y_options = [o for o in display_options if o not in x_selected]

        col_x, col_y = st.columns(2)

        with col_x:
            x_selected = st.multiselect(
                "X columns",
                options=x_options,
                default=[v for v in x_selected if v in x_options],
                help="Select one or more feature/input columns.",
                key="csv_x_cols",
            )

        with col_y:
            y_selected = st.multiselect(
                "Y columns",
                options=y_options,
                default=[v for v in y_selected if v in y_options],
                help="Select one or more target/output columns.",
                key="csv_y_cols",
            )

        st.session_state["X_columns"] = x_selected
        st.session_state["Y_columns"] = y_selected

    if uploaded is not None:

        st.success(f"Sucessfully loaded!")
        uploaded_df = st.session_state["uploaded_df"]
        st.dataframe(uploaded_df, width='stretch')
        st.session_state.data_submitted = True

st.header("Hyperparameter Configuration")

st.write("")

if mode == "Manual":
    col_tau, col_eps, col_table_prior, col_maxiter, _ = st.columns(5)
else:
    col_tau, col_eps, col_table_prior, col_maxiter = st.columns(4)

with col_tau:
    tau = st.number_input(
        "Tau",
        min_value=1,
        max_value=100,
        value=2,
        step=1,
        key="input_tau",
        help="Threshold value for similarity matching"
    )

with col_eps:
    epsilon = st.number_input(
        "Epsilon",
        min_value=0.0,
        max_value=1000.0,
        value=0.001,
        step=0.0001,
        format="%.4f",
        key="input_epsilon",
        help="Minimum answer probabilites change required between iterations before stopping (convergence tolerance)"
    )

with col_table_prior:
    table_prior = st.number_input(
        "Tables prior",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
        format="%.2f",
        key="input_tp",
        help="Inital assumption about overall table dirtiness (lower value = dirtier tables)"
    )

with col_maxiter:
    c1, c2 = st.columns([4, 1])

    with c2:
        st.write("")
        st.write("")
        infinite = st.checkbox(
            label = "∞",
            key="maxiter_inf",
            value=True
        )

    with c1:
        max_iterations = st.number_input(
            "Max Iterations",
            min_value=1,
            max_value=1000,
            value=10,
            step=1,
            key="input_max_iterations",
            disabled=infinite,
            help="Upper limit on the number of quering iterations (infinite = until converged)"
        )

    if infinite:
        max_iterations = float("inf")

if mode == "CSV upload":
    if st.session_state.data_submitted:
        x_valid = st.session_state.get("X_columns") and len(st.session_state["X_columns"]) >= 1
        y_valid = st.session_state.get("Y_columns") and len(st.session_state["Y_columns"]) >= 1

        uploaded_df = st.session_state.uploaded_df

        disabled = not (x_valid and y_valid)

        st.write("")
        st.write("")

        if st.button("🚀 START PROCESSING", type="primary", width="stretch", disabled=disabled):
            submitted_x = [[] for _ in x_selected]
            submitted_y = [[] for _ in y_selected]
            submitted_queries = [[] for _ in x_selected]

            for _, row in uploaded_df.iterrows():
                x_vals = [row[c] for c in x_selected]
                y_vals = [row[c] for c in y_selected]

                x_all_valid = all(v is not None and not pd.isna(v) for v in x_vals)
                y_all_valid = all(v is not None and not pd.isna(v) for v in y_vals)
                y_all_none  = all(v is None or pd.isna(v) for v in y_vals)

                if x_all_valid and y_all_valid:

                    for i, v in enumerate(x_vals):
                        submitted_x[i].append(str(v))

                    for i, v in enumerate(y_vals):
                        submitted_y[i].append(str(v))

                elif x_all_valid and y_all_none:

                    for i, v in enumerate(x_vals):
                        submitted_queries[i].append(str(v))
                        
            st.session_state.submitted_x = submitted_x
            st.session_state.submitted_y = submitted_y
            st.session_state.submitted_queries = submitted_queries

            st.session_state.num_x_cols = len(x_selected)
            st.session_state.num_y_cols = len(y_selected)
            st.session_state.num_training_rows = len(submitted_x[0])
            st.session_state.num_query_rows = len(submitted_queries[0])
            st.session_state.start_processing = True
            st.session_state.results_available = False
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
                st.session_state.results_available = False
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
        # cleaned_x_lists     = x_lists
        # cleaned_y_lists     = y_lists
        # tokenized_querries  = query_x_lists
        tau = st.session_state.input_tau

        ##Init of Algorithm

        # TODO: initialize it once at the beginning
        config = get_default_vertica_config()
        qf = QueryFactory(config)


        ##Logic
        # print("Start 1")
        # z = qf.find_xy_candidates(cleaned_x_lists, cleaned_y_lists, tau)        ###Warum werden die denn in einer unterschiedlichen Reihenfolge je nach itteration angezeigt. 
        # print("Finish")                                                         ###Ist der Prozess nicht deterministisch? Liegt das am Kernel und Parallel Processing? 
        # erg = qf.stable_row_val(z, cleaned_x_lists, cleaned_y_lists, tau)
        # print("Done with erg")
        # st.subheader("All Direct Tables")
        # st.write(erg)
        # answers = qf.stable_get_y(next(iter(erg)), tokenized_querries) ###Erwartet Tuple, keine Liste. Erg ist eine Liste von Tuplen! 

        ranking_config = get_default_ranking_config()
        ranking_config.table_prior = table_prior
        ranking_config.epsilon = epsilon
        ranking_config.max_iterations = max_iterations
        ranker = WebTableRanker(ranking_config, tau)

        # TODO: generator to print out the current believe (and stop button)
        print("Starting EM algorithm!")
        answers = ranker.expectation_maximization(cleaned_x_lists, cleaned_y_lists, tokenized_querries)

        if answers:

            rows = []
            for idx, query in enumerate(zip(*tokenized_querries), 1):
                 
                    row_dict = {}
                    
                    row_dict['Query #'] = idx

                    # TODO: do it with real x values (not tokenized)
                    num_x = st.session_state.num_x_cols
                    for i in range(num_x):
                        row_dict[f'X{i+1}'] = query[i]

                    candidates = answers[query]

                    candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)

                    candidates_sorted = [candidate for candidate in candidates_sorted if candidate[1] >= 0.01]

                    row_dict["Y_candidates"] = [f"{candidate[0]} ({candidate[1]:.2f})" for candidate in candidates_sorted]

                    if candidates_sorted: 
                        row_dict["Y_selected"] = f"{candidates_sorted[0][0]} ({candidates_sorted[0][1]:.2f})"
                    else:
                        row_dict["Y_selected"] = None
                    
                    rows.append(row_dict)

            df_results = pd.DataFrame(rows)

            st.session_state.df_results = df_results

            st.session_state.results_available = True
            st.success("Processing Complete!")
            st.rerun()

        else:
            st.warning("No answers found")

        


        ##Erster Querry für Multi-Hop. Collected alle T_x 
        
        # multi_z = qf.find_xy_candidates(cleaned_x_lists, None, tau, True)
        # st.subheader("Multi-Hop Tables")
        # st.write(multi_z)

        # erg_set = set(Index[0] for Index in erg)

        # clean = list()
        # for Multi_IDX in multi_z: 
        #     Table_ID, *_ = Multi_IDX
        #     if Table_ID not in erg_set: 
        #         clean.append(Table_ID) #####jaja, verlust von Col_ids, aber das hier ist nur ein Test 
        
        # st.subheader("T_X <- T_X/T_E")
        # st.write(clean)


        ####Das hier muss alles überarbeitet werden, muss in eigene Datei "Logic" oder "Pipeline"
        # EM, Multihop etc. fehlen noch hier 


        # ####Hier EM und itterativ. Multi erst hier nach? Erst starten, wenn normales nicht für alle x in Q 
        # ####ein Ergebnis gebracht hat? 



        ###Visualisation of Results

if 'results_available' in st.session_state and st.session_state.results_available:
    st.write("---")
    st.subheader("Results")

    df_results = st.session_state.df_results
    num_x_cols = st.session_state.num_x_cols
    num_y_cols = st.session_state.num_y_cols

    submitted = False
    
    if not df_results.empty:
        new_rows = []
        header_cols = st.columns([1] * num_x_cols + [(num_x_cols + 1)])

        for j in range(num_x_cols):
            with header_cols[j]:
                st.markdown(f"**X{j+1}**")

        with header_cols[-1]:
            st.markdown("**Answer selection** (Answer, Probability)")
        with st.form("results_table"):

            for i, row in df_results.iterrows():
                cols = st.columns([1] * num_x_cols + [(num_x_cols+1)])

                for j in range(num_x_cols):
                    with cols[j]:
                        st.write(row[f"X{j+1}"])

                with cols[-1]:
                    options = row["Y_candidates"] + [None]
                    selected = st.selectbox(
                        label="Y Selection",
                        label_visibility="collapsed",
                        options=options,
                        index=options.index(row["Y_selected"]) if row["Y_selected"] in options else None,
                        key=f"y_select_{i}",
                    )
            submitted = st.form_submit_button("Update selections")
    else:
        st.warning("No results to display")

    if submitted:
        for i, row in df_results.iterrows():
            x_vals = []
            for j in range(num_x_cols):
                x_vals.append(row[f"X{j+1}"])
            selected_answer = st.session_state[f"y_select_{i}"]
            if selected_answer:
                selected_answer = selected_answer.rsplit(" (", 1)[0]
                selected_answer = list(ast.literal_eval(selected_answer))
            else:
                selected_answer = [None] * num_y_cols
            new_rows.append(x_vals + selected_answer)

        updated_df = pd.DataFrame(new_rows)
        csv = updated_df.to_csv(index=False, header=False)

        if mode == "CSV upload":
            x_selected = st.session_state.X_columns
            y_selected = st.session_state.Y_columns

            selected = x_selected + y_selected

            uploaded_df = st.session_state.uploaded_df

            rename_dict = dict()
            for i, column in enumerate(selected):
                rename_dict[i] = column

            updated_df.rename(columns=rename_dict, inplace=True)

            for col in selected:
                updated_df[col] = updated_df[col].astype(uploaded_df[col].dtype)


            updated_df = uploaded_df.merge(
                updated_df,
                on=x_selected,
                how="left",
                suffixes=("", "_new"),
            )

            for y in y_selected:
                updated_df[str(y)] = updated_df[str(y)].fillna(updated_df[f"{y}_new"])
                updated_df.drop(columns=[f"{y}_new"], inplace=True)

            csv = updated_df.to_csv(index=False, header=st.session_state["csv_has_header"])

        st.success("Selected Answers Saved!")

        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="query_results.csv",
            mime="text/csv"
        )


###Example Bill Gates Microsoft line 21 (unser Stemmer stemmed ihn leider als bill gate, token ist aber gates) Mark Zuckerberg Facebook (passt) Steve Ballmer Microsoft 175 Kevin Mitnick Hacker 113
###Table 30,440,039

#Q: Bob Frankston? 26 Biz Stone 25

##Todo: Check if all Values are Filled in 
##Todo: Demo Mode with Predefined Values 
#streamlit run /Users/christophhalberstadt/Documents/GitHub/ProjectDataXFormer/src/Frontend.py  