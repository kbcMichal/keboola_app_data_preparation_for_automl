import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from kbcstorage.client import Client
from kbcstorage.tables import Tables
from kbcstorage.buckets import Buckets
import csv
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import h2o
from h2o.automl import H2OAutoML

# Function to handle missing values based on selected action
def solve_missing(data, missing_action_concat):
    if missing_action_concat == 'NoneAll':
        st.write('Not solving any columns.')
        return data
    
    if missing_action_concat == 'replaceAll':
        st.write('Replacing missing values in all columns:')
        for col in data.columns:
            if data[col].isna().sum() > 0:
                if data[col].dtype == 'object':
                    data[col].fillna('REPLACED-Undefined', inplace=True)
                else:
                    data[col].fillna(data[col].mean(), inplace=True)
                st.write(col)
                
    elif "replaceNumeric" in missing_action_concat:
        st.write('Replacing missing values in NUMERIC columns:')
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].isna().sum() > 0:
                data[col].fillna(data[col].mean(), inplace=True)            
                st.write(col)
            
    elif "replaceCategorical" in missing_action_concat:
        st.write('Replacing missing values in CATEGORICAL columns:')
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].isna().sum() > 0:
                data[col].fillna('REPLACED-Undefined', inplace=True)
                st.write(col)
    
    elif "replace" in missing_action_concat:
        st.write('Replacing missing values in selected columns.')
        cols_to_replace = eval(missing_action_concat.replace("replace", ""))
        for col in cols_to_replace:
            if data[col].dtype == 'object':
                if data[col].isna().sum() > 0:
                    data[col].fillna('REPLACED-Undefined', inplace=True)
                    st.write(col)
            else:
                if data[col].isna().sum() > 0:
                    data[col].fillna(data[col].mean(), inplace=True)            
                    st.write(col)
                        
    if missing_action_concat == 'dropAll':
        st.write('Dropping missing values in all columns.')
        data.dropna(inplace=True)
            
    elif "drop" in missing_action_concat[:4]:
        st.write('Dropping missing values in selected columns.')
        cols_to_drop = eval(missing_action_concat.replace("drop", ""))
        data.dropna(subset=cols_to_drop, inplace=True)
        st.write(cols_to_drop)
    
    return data

@st.cache_data(ttl=7200)
def get_dataframe(_client, table_name):
    table_detail = _client.tables.detail(table_name)
    _client.tables.export_to_file(table_id=table_name, path_name='')
    with open('./' + table_detail['name'], mode='rt', encoding='utf-8') as in_file:
        lazy_lines = (line.replace('\0', '') for line in in_file)
        reader = csv.reader(lazy_lines, lineterminator='\n')
    if os.path.exists((table_detail['name'] + '.csv')):
        os.remove(table_detail['name'] + '.csv')
    else:
        print("The file does not exist")
    os.rename(table_detail['name'], table_detail['name'] + '.csv')
    df = pd.read_csv(table_detail['name'] + '.csv')
    return df

def load_datasets():
    train_file = st.session_state.train_file_path
    test_file = st.session_state.test_file_path
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

def gini_coefficient(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    gini = 2 * roc_auc_score(y_true, y_prob) - 1
    return gini

st.set_page_config(layout="wide")
st.title("Data Preparation for Machine Learning")

st.sidebar.header("Select Dataset")
url_options = [
    "https://connection.keboola.com/",
    "https://connection.eu-central-1.keboola.com/",
    "https://connection.north-europe.azure.keboola.com/"
]
selected_url = st.sidebar.selectbox("Select Keboola Connection URL", url_options, key="url_selectbox")
token = st.sidebar.text_input("Enter your API Token", type="password")

if 'connected' not in st.session_state:
    st.session_state.connected = False

if st.sidebar.button("Connect"):
    with st.spinner("Connecting..."):
        if token:
            st.session_state.client = Client(selected_url, token)
            st.session_state.buckets = Buckets(selected_url, token)
            st.session_state.bucket_list = [bucket['id'] for bucket in st.session_state.buckets.list()]
            st.session_state.connected = True
        else:
            st.sidebar.error("API Token is required to connect.")

if st.session_state.connected:
    selected_bucket = st.sidebar.selectbox("Select Bucket", st.session_state.bucket_list, key="bucket_selectbox")
    bucket_tables = st.session_state.buckets.list_tables(selected_bucket)
    table_list = [table['id'] for table in bucket_tables]
    selected_table = st.sidebar.selectbox("Select Table", table_list, key="table_selectbox")
    
    if st.sidebar.button("Load Dataset"):
        with st.spinner("Fetching data from Storage, please wait..."):
            data = get_dataframe(st.session_state.client, selected_table)
            st.session_state.data = data
            

if 'data' in st.session_state:
    data = st.session_state.data.copy()
    tab_names = ["Data Profiling", "Redundant Columns", "Duplicates", "Missing Values", "Encoding", "Scaling", "Dimensionality Reduction", "Split & Save", "Correlation Treatment", "Feature Selection", "AutoML"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.subheader("Data Profiling")

        if st.button("Display Dataset"):
            with st.spinner("Displaying dataset..."):
                st.session_state.display_data = data.head()

        if "display_data" in st.session_state:
            st.write(st.session_state.display_data)

        if st.button("Generate Profile"):
            with st.spinner("Generating profile..."):
                profile = ProfileReport(data, title="Data Profile Report", explorative=True)
                profile_html = profile.to_html()
                st.session_state.profile_html = profile_html

        if "profile_html" in st.session_state:
            components.html(st.session_state.profile_html, height=1000, scrolling=True)


    with tabs[1]:
        st.subheader("Removing Redundant Columns")
        redundant_columns = [col for col in data.columns if data[col].nunique() <= 1]
        if redundant_columns:
            st.write(f"The following columns have only one unique value and can be considered redundant: {redundant_columns}")
            drop_redundant = st.radio("Drop redundant columns?", ('Yes', 'No'), key="redundant_radio")
            if st.button("Remove Redundant Columns"):
                with st.spinner("Removing redundant columns..."):
                    if drop_redundant == 'Yes':
                        data.drop(columns=redundant_columns, inplace=True)
                        st.write(f"Dropped columns: {redundant_columns}")
        else:
            st.write("No redundant columns found.")
        st.write(data.head())
        st.session_state.data = data

    with tabs[2]:
        st.subheader("Solving Duplicate Rows")
        # Find duplicate rows including the first occurrence
        duplicate_mask = data.duplicated(keep=False)
        duplicate_rows = data[duplicate_mask]
        num_duplicate_rows = duplicate_rows.shape[0]
        num_unique_duplicates = duplicate_rows.drop_duplicates().shape[0]
        
        if num_duplicate_rows > 0:
            st.write(f"There are {num_duplicate_rows} duplicate rows in the dataset.")
            st.write("Duplicate rows:")
            st.write(duplicate_rows)
            
            drop_duplicates = st.radio("Drop duplicate rows? This option will keep one observation of each duplicate item.", ('Yes', 'No'), key="duplicates_radio")
            if st.button("Remove Duplicates"):
                with st.spinner("Removing duplicate rows..."):
                    if drop_duplicates == 'Yes':
                        data = data.drop_duplicates(keep='first')
                        num_dropped_rows = num_duplicate_rows - num_unique_duplicates
                        st.write(f"Dropped {num_dropped_rows} duplicate rows, keeping one unique item for each observed duplicity.")
                        st.session_state.data = data
        else:
            st.write("No duplicate rows found.")
            
        st.write(data.head())

    with tabs[3]:
        st.subheader("Solve Missing Values")
        st.markdown("""
        **Choose an action for handling missing values.**
        You can apply different actions to individual columns by running this step multiple times.
        """)

        def get_missing(data):
            missing_cnt = data.isna().sum().sum()
            missing_pct = missing_cnt / (len(data.columns) * len(data))     
            missing_out = data.isna().sum()
            
            st.write(f'Total missing cells: {missing_cnt}')
            st.write(f'Percentage of missing cells: {missing_pct:.2%}')
            st.write('Count of missing cells per column:')
            st.write(missing_out)

        get_missing(data)
        if data.isna().sum().sum() > 0:
            missing_action = st.radio('Action:', ('None', 'drop', 'replace', 'replaceNumeric', 'replaceCategorical'), key="missing_radio")
            columns_action = st.multiselect('Columns:', ['ALL COLUMNS'] + list(data.columns), key="missing_multiselect")
            st.markdown("""
                        Options:
                            - **None**: Do nothing with missing values.
                            - **Drop**: Remove rows containing missing values.
                            - **Replace**: Replace missing values with the mean for numeric columns and 'Undefined' for categorical columns.
                            - **Replace Numeric**: Replace missing values in numeric columns with the mean value.
                            - **Replace Categorical**: Replace missing values in categorical columns with 'Undefined'.
                        """)
            if st.button("Apply Missing Values Action"):
                with st.spinner("Applying missing values action..."):
                    if 'ALL COLUMNS' in columns_action:
                        missing_action_concat = missing_action + 'All'
                    else:
                        missing_action_concat = missing_action + str(columns_action)
                    data = solve_missing(data, missing_action_concat)
                    st.write(data.head())
                    st.session_state.data = data

    with tabs[4]:
        st.subheader("Encoding Categorical Variables")
        st.markdown("""
        **Select an encoding method for each categorical variable.**
        
        Options:
        - **One-Hot Encoding**: Creates a new binary column for each unique category.
        - **Label Encoding**: Converts each category to a unique integer.
        """)
        
        categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
        encoding_methods = ['Label Encoding', 'One-Hot Encoding']
        encoding_dropdowns = {col: st.selectbox(f'{col}:', encoding_methods, key=f"encoding_{col}") for col in categorical_columns}
        
        if st.button("Apply Encoding"):
            with st.spinner("Applying encoding..."):
                for col, method in encoding_dropdowns.items():
                    if method == 'One-Hot Encoding':
                        data = pd.get_dummies(data, columns=[col], drop_first=True)
                    elif method == 'Label Encoding':
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col])
                st.write("Encoding applied. Here is the updated dataset:")
                st.write(data.head())
                st.session_state.data = data

    with tabs[5]:
        st.subheader("Feature Scaling")
        st.markdown("""
        **Feature scaling ensures that all numerical features are on the same scale.**
        
        This helps improve the performance of many machine learning algorithms that are sensitive to the scale of input data.
        """)
        target_column = st.selectbox('Select Target Column:', data.columns, key="scaling_target_column")
        if st.button("Perform Feature Scaling"):
            with st.spinner("Performing feature scaling..."):
                scaler = StandardScaler()
                numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                numerical_columns = [col for col in numerical_columns if col != target_column]
                data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
                st.write("Feature scaling applied to numerical columns.")
                st.write(data.head())
                st.session_state.data = data

    with tabs[6]:
        st.subheader("Dimensionality Reduction")
        st.markdown("""
        **Dimensionality reduction reduces the number of features while retaining most of the important information.**
        
        This helps to reduce computational cost and can improve model performance.
        
        - **PCA (Principal Component Analysis)**: Retains a specified amount of variance in the data.
        """)
        target_column = st.selectbox('Select Target Column:', data.columns, key="dimensionality_target_column")
        if st.button("Perform Dimensionality Reduction"):
            with st.spinner("Performing dimensionality reduction..."):
                pca = PCA(n_components=0.95)
                numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                numerical_columns = [col for col in numerical_columns if col != target_column]
                X_reduced = pca.fit_transform(data[numerical_columns])
                X_pca = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(X_reduced.shape[1])])
                non_numerical_data = data.drop(columns=numerical_columns)
                data = pd.concat([non_numerical_data.reset_index(drop=True), X_pca.reset_index(drop=True)], axis=1)
                st.write("Dimensionality reduction applied (if applicable).")
                st.write(data.head())
                st.session_state.data = data

    with tabs[7]:
        st.subheader("Splitting and Saving the Data")
        st.markdown("""
        **Split the dataset into training and testing sets and save the transformed dataset to Keboola.**
        
        - Select the target column which is the variable you aim to predict.
        - Choose the size of the test set.
        """)

        target_column = st.selectbox('Select Target Column:', data.columns, key="split_target_column")
        test_size = st.slider('Test Size:', 0.1, 0.5, 0.2, 0.1, key="split_test_size")
        selected_bucket = st.selectbox("Select Bucket to Save Data", st.session_state.bucket_list, key="save_bucket")
        train_table_name = st.text_input("Enter Train Table Name", "train_dataset", key="save_train_table")
        test_table_name = st.text_input("Enter Test Table Name", "test_dataset", key="save_test_table")

        if selected_bucket and train_table_name and test_table_name:
            if st.button("Split and Save Dataset"):
                with st.spinner("Splitting and saving the dataset..."):
                    X = data.drop(columns=[target_column])
                    y = data[target_column]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    st.write(f"Data split into training and testing sets with test size = {test_size}.")

                    train_file_path = 'train_dataset.csv'
                    test_file_path = 'test_dataset.csv'

                    # Save the split train and test datasets locally
                    train_dataset = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
                    test_dataset = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

                    train_dataset.to_csv(train_file_path, index=False)
                    test_dataset.to_csv(test_file_path, index=False)

                    client = st.session_state.client
                    train_table_id = f"{selected_bucket}.{train_table_name}"
                    test_table_id = f"{selected_bucket}.{test_table_name}"

                    # Try to upload datasets to Keboola, if it fails, create the table
                    try:
                        client.tables.load(table_id=train_table_id, file_path=train_file_path, is_incremental=False)
                        st.write(f"Dataset {train_table_id} loaded to Keboola...")
                    except Exception as e:
                        st.write(f"Failed to load training dataset: {e}")
                        st.write(f"Creating table {train_table_id} in Keboola...")
                        if 'Some columns are missing in the csv file' in str(e):
                            # Drop table first
                            st.write(f"The table already exists with different set of columns. Dropping the table and creating it again...")
                            client.tables.delete(table_id=train_table_id)
                            # Then create it again
                            client.tables.create(name=train_table_name, bucket_id=selected_bucket, file_path=train_file_path)
                            st.write(f"Training dataset created as '{train_table_id}'.")
                        else:
                            client.tables.create(name=train_table_name, bucket_id=selected_bucket, file_path=train_file_path)
                            st.write(f"Training dataset created as '{train_table_id}'.")
                            
                    try:
                        client.tables.load(table_id=test_table_id, file_path=test_file_path, is_incremental=False)
                        st.write(f"Dataset {test_table_id} loaded to Keboola...")
                    except Exception as e:
                        st.write(f"Failed to load testing dataset: {e}")
                        st.write(f"Creating table {test_table_id} in Keboola...")
                        if 'Some columns are missing in the csv file' in str(e):
                            # Drop table first
                            st.write(f"The table already exists with different set of columns. Dropping the table and creating it again...")
                            client.tables.delete(table_id=test_table_id)
                            # Then create it again
                            client.tables.create(name=test_table_name, bucket_id=selected_bucket, file_path=test_file_path)
                            st.write(f"Testing dataset created as '{test_table_id}'.")
                        else:
                            client.tables.create(name=test_table_name, bucket_id=selected_bucket, file_path=test_file_path)
                            st.write(f"Testing dataset created as '{test_table_id}'.")
                            
                        

                    st.session_state.train_file_path = train_file_path
                    st.session_state.test_file_path = test_file_path
                    st.session_state.split_target_column_name = target_column

    with tabs[8]:
        st.subheader("Correlation Treatment")
        
        if 'train_file_path' in st.session_state and 'test_file_path' in st.session_state:
            train_data, test_data = load_datasets()
            target_column = st.session_state.split_target_column_name

            st.write("Identifying and handling highly correlated features. Please select the correlation threshold.")
            correlation_threshold = st.slider('Correlation Threshold:', 0.5, 1.0, 0.9, 0.05)
            
            corr_matrix = train_data.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
            st.write(f"Highly correlated features (threshold = {correlation_threshold}): {to_drop}")

            drop_corr_features = st.radio("Drop highly correlated features?", ('Yes', 'No'), key="drop_corr_features")
            if st.button("Apply Correlation Treatment"):
                with st.spinner("Applying correlation treatment..."):
                    if drop_corr_features == 'Yes':
                        train_data.drop(columns=to_drop, inplace=True)
                        test_data.drop(columns=to_drop, inplace=True)
                        st.write(f"Dropped features: {to_drop}")
                        st.session_state.train_data_selected = train_data
                        st.session_state.test_data_selected = test_data
                    else:
                        st.write("No features were dropped.")
        else:
            st.write("Please perform the 'Split & Save' step first.")

    with tabs[9]:
        st.subheader("Feature Selection")
        
        if 'train_file_path' in st.session_state and 'test_file_path' in st.session_state:
            train_data = st.session_state.get('train_data_selected', load_datasets()[0])
            test_data = st.session_state.get('test_data_selected', load_datasets()[1])
            target_column = st.session_state.split_target_column_name

            if st.button("Calculate Feature Importance and Performance"):
                with st.spinner("Calculating feature importance and performance..."):
                    st.write("Calculating GINI on train and test datasets.")
                    
                    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                    model.fit(train_data.drop(columns=[target_column]), train_data[target_column])

                    train_pred = model.predict_proba(train_data.drop(columns=[target_column]))[:, 1]
                    test_pred = model.predict_proba(test_data.drop(columns=[target_column]))[:, 1]

                    train_gini = gini_coefficient(train_data[target_column], train_pred)
                    test_gini = gini_coefficient(test_data[target_column], test_pred)

                    st.write(f"Train GINI: {train_gini}")
                    st.write(f"Test GINI: {test_gini}")

                    st.write("Calculating feature importance using XGBoost and L1 regularized logistic regression.")
                    
                    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                    xgb_model.fit(train_data.drop(columns=[target_column]), train_data[target_column])
                    xgb_feature_importance = pd.Series(xgb_model.feature_importances_, index=train_data.drop(columns=[target_column]).columns)

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(train_data.drop(columns=[target_column]))

                    l1_model = LogisticRegression(penalty='l1', solver='liblinear')
                    l1_model.fit(X_scaled, train_data[target_column])
                    l1_feature_importance = pd.Series(np.abs(l1_model.coef_[0]), index=train_data.drop(columns=[target_column]).columns)

                    st.write("XGBoost Feature Importance:")
                    st.table(xgb_feature_importance.sort_values(ascending=False).head(10))

                    st.write("L1 Regularized Logistic Regression Feature Importance:")
                    st.table(l1_feature_importance.sort_values(ascending=False).head(10))

                    st.write("Based on the feature importance analysis, you may choose to use all features or select a subset of important features.")
                    
                    important_features_xgb = xgb_feature_importance.sort_values(ascending=False).head(10).index.tolist()
                    important_features_l1 = l1_feature_importance.sort_values(ascending=False).head(10).index.tolist()

                    suggested_features = list(set(important_features_xgb + important_features_l1))
                    st.write(f"Suggested important features based on analysis: {suggested_features}")

                    st.session_state.suggested_features = suggested_features

            if 'suggested_features' in st.session_state:
                use_important_features = st.radio("Select features to use:", ('All Features', 'Important Features'), key="use_important_features")
                if st.button("Apply Feature Selection"):
                    with st.spinner("Applying feature selection..."):
                        if use_important_features == 'Important Features':
                            st.session_state.train_data_selected = train_data[st.session_state.suggested_features + [target_column]]
                            st.session_state.test_data_selected = test_data[st.session_state.suggested_features + [target_column]]
                            st.write("Using only important features for training.")
                        else:
                            st.session_state.train_data_selected = train_data
                            st.session_state.test_data_selected = test_data
                            st.write("Using all features for training.")
        else:
            st.write("Please perform the 'Split & Save' step first.")

    with tabs[10]:
        st.subheader("H2O AutoML")
        
        if 'train_file_path' in st.session_state and 'test_file_path' in st.session_state:
            train_data = st.session_state.get('train_data_selected', load_datasets()[0])
            test_data = st.session_state.get('test_data_selected', load_datasets()[1])
            target_column = st.session_state.split_target_column_name

            predictor_columns = [col for col in train_data.columns if col != target_column]
            
            st.write(f"Target variable: {target_column}")
            st.write(f"Predictor variables: {predictor_columns}")

            max_models = st.number_input("Max Models", min_value=1, max_value=100, value=20, step=1)
            seed = st.number_input("Seed", value=1, step=1)

            if st.button("Initialize H2O and Run AutoML"):
                with st.spinner("Initializing H2O and running AutoML..."):
                    h2o.init()
                    h2o_train = h2o.H2OFrame(train_data)
                    h2o_test = h2o.H2OFrame(test_data)

                    if train_data[target_column].nunique() == 2:
                        h2o_train[target_column] = h2o_train[target_column].asfactor()
                        h2o_test[target_column] = h2o_test[target_column].asfactor()
                        st.write(f"Target variable '{target_column}' converted to categorical for binary classification.")

                    x = h2o_train.columns
                    y = target_column
                    x.remove(y)

                    aml = H2OAutoML(max_models=max_models, seed=seed)
                    aml.train(x=x, y=y, training_frame=h2o_train)

                    lb = aml.leaderboard.as_data_frame()
                    st.write("H2O AutoML Leaderboard:")
                    st.table(lb.head())
                    
                    st.session_state.aml = aml
                    st.session_state.lb = lb

            if 'aml' in st.session_state:
                selected_model = st.selectbox("Select Model to Save", st.session_state.lb['model_id'])
                if st.button("Save Selected Model"):
                    with st.spinner("Saving selected model..."):
                        model = h2o.get_model(selected_model)
                        model_path = h2o.save_model(model=model, path="", force=True)
                        st.write(f"Model saved to {model_path}")
        else:
            st.write("Please perform the 'Split & Save' step first.")
