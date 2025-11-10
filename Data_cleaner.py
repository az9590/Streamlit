import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import traceback
import warnings
from datetime import datetime

# CUSTOM ERROR HANDLING

class DataValidationError(Exception):
    """Custom exception for data validation failures"""
    def __init__(self, message, context=None):
        self.message = message
        self.context = context or {}
        super().__init__(f"Data Validation Error: {message}")
        
    def to_dict(self):
        return {
            "error_type": "validation",
            "message": self.message,
            "context": self.context,
            "stack_trace": traceback.format_exc() if hasattr(traceback, 'format_exc') else ""
        }

class ProcessingError(Exception):
    """Custom exception for data processing failures"""
    def __init__(self, message, context=None):
        self.message = message
        self.context = context or {}
        super().__init__(f"Data Processing Error: {message}")
        
    def to_dict(self):
        return {
            "error_type": "processing",
            "message": self.message,
            "context": self.context,
            "stack_trace": traceback.format_exc() if hasattr(traceback, 'format_exc') else ""
        }

# SESSION STATE INITIALIZATION

if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = {}
if 'processing_errors' not in st.session_state:
    st.session_state.processing_errors = []

# HELPER FUNCTIONS

def validate_file(file):
    """Validate uploaded file with comprehensive checks"""
    try:
        # Check file size
        if file.size > 100 * 1024 * 1024:  # 100MB
            raise DataValidationError("File size exceeds 100MB limit", {"max_size": 100 * 1024 * 1024})
        
        # Check file type
        file_name = file.name
        if file_name.lower().endswith('.csv'):
            df = pd.read_csv(file)
        elif file_name.lower().endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            raise DataValidationError("Unsupported file format", {"supported": ["csv", "xlsx"]})
        
        # Check for empty data
        if df.empty:
            raise DataValidationError("File contains no data", {})
        
        return df
    
    except Exception as e:
        raise DataValidationError(f"File validation failed: {str(e)}", {"error": str(e)})

def run_data_validation(df):
    """Perform comprehensive data validation checks"""
    results = {}
    try:
        # Check for missing columns
        required_columns = ['id', 'timestamp', 'value']  # Example required columns
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            results['missing_columns'] = missing
        
        # Check for data quality issues
        null_counts = df.isnull().sum()
        if null_counts.any():
            results['null_counts'] = null_counts.to_dict()
        
        # Check for duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            results['duplicate_rows'] = dup_count
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                results[f'outliers_{col}'] = len(outliers)
        
        # Check for data types
        dtypes = df.dtypes
        for col, dtype in dtypes.items():
            if dtype == object:
                results['object_columns'] = col
        
        # Check for time series format
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
            except:
                results['timestamp_format'] = "Invalid datetime format"
        
        return results
    
    except Exception as e:
        st.session_state.processing_errors.append({
            "timestamp": datetime.now().isoformat(),
            "message": f"Validation failed: {str(e)}",
            "stack_trace": traceback.format_exc()
        })
        return {}

def run_cleaning_operations(df):
    """Apply comprehensive data cleaning operations"""
    try:
        # 1. Handle missing values
        df = df.copy()
        df = df.dropna(axis=1, how='all')  # Drop columns with all NaN
        df = df.fillna(method='ffill', limit=1)  # Forward fill for first missing
        
        # 2. Handle duplicates
        df = df.drop_duplicates()
        
        # 3. Standardize data types
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # 4. Handle outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 5. Add timestamp if missing
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime('now')
        
        return df
    
    except Exception as e:
        st.session_state.processing_errors.append({
            "timestamp": datetime.now().isoformat(),
            "message": f"Cleaning operation failed: {str(e)}",
            "stack_trace": traceback.format_exc()
        })
        raise ProcessingError(f"Cleaning failed: {str(e)}", {"error": str(e)})

# STREAMLIT APP

st.set_page_config(
    page_title="Data Cleaner Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR - USER CONFIGURATION

with st.sidebar:
    st.title("Data Cleaning Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"],
        help="Upload your data file for cleaning"
    )
    
    # Processing options
    st.subheader("Processing Options")
    clean_missing = st.checkbox("Clean missing values", value=True)
    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
    standardize_types = st.checkbox("Standardize data types", value=True)
    handle_outliers = st.checkbox("Handle outliers", value=True)
    
    # Reset button
    if st.button("Reset to Original Data", type="primary"):
        st.session_state.current_data = st.session_state.original_data.copy()
        st.rerun()

# MAIN CONTENT AREA

st.title("📊 Data Cleaner Pro")
st.markdown("""
This application provides a robust data cleaning and processing solution with comprehensive validation and error handling.
""")

# FILE UPLOAD AND VALIDATION

if uploaded_file:
    try:
        # Validate file
        with st.spinner("Validating file..."):
            df = validate_file(uploaded_file)
        
        # Store original data
        st.session_state.original_data = df.copy()
        st.session_state.current_data = df.copy()
        
        st.success("✅ File loaded successfully!")
        
    except DataValidationError as e:
        with st.expander("❌ File Validation Error"):
            st.error(f"File validation failed: {e.message}")
            st.code(e.context, language="json")
        st.stop()

# DATA VALIDATION SECTION

st.subheader("🔍 Data Validation Results")
if st.session_state.original_data is not None:
    try:
        # Run validation
        validation_results = run_data_validation(st.session_state.original_data)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(st.session_state.original_data))
        with col2:
            st.metric("Columns", len(st.session_state.original_data.columns))
        
        # Show validation metrics
        for metric, value in validation_results.items():
            if isinstance(value, list):
                st.metric(f"{metric}", f"{len(value)} columns")
            elif isinstance(value, dict):
                st.metric(f"{metric}", f"{value}")
            else:
                st.metric(f"{metric}", f"{value}")
        
        # Show detailed errors
        if st.session_state.processing_errors:
            st.warning("⚠️ Validation errors found")
            for error in st.session_state.processing_errors:
                st.error(f"{error['timestamp']}: {error['message']}")
    
    except Exception as e:
        st.error(f"Data validation failed: {str(e)}")

# DATA CLEANING SECTION

st.subheader("🛠️ Data Cleaning Operations")
if st.session_state.current_data is not None:
    try:
        # Run cleaning operations
        with st.spinner("Applying data cleaning operations..."):
            if clean_missing:
                st.session_state.current_data = st.session_state.current_data.dropna(axis=1, how='all')
            if remove_duplicates:
                st.session_state.current_data = st.session_state.current_data.drop_duplicates()
            if standardize_types:
                for col in st.session_state.current_data.columns:
                    if st.session_state.current_data[col].dtype == object:
                        try:
                            st.session_state.current_data[col] = pd.to_numeric(st.session_state.current_data[col], errors='coerce')
                        except:
                            pass
            if handle_outliers:
                numeric_cols = st.session_state.current_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    q1 = st.session_state.current_data[col].quantile(0.25)
                    q3 = st.session_state.current_data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    st.session_state.current_data[col] = st.session_state.current_data[col].clip(lower=lower_bound, upper=upper_bound)
        
        st.success("✅ Data cleaning completed successfully!")
    
    except ProcessingError as e:
        with st.expander("❌ Data Cleaning Error"):
            st.error(f"Data cleaning failed: {e.message}")
            st.code(e.context, language="json")
        st.stop()

# FINAL DATA DISPLAY

st.subheader("✅ Final Data")
if st.session_state.current_data is not None:
    try:
        # Display sample data
        with st.expander("View sample data (first 10 rows)"):
            st.dataframe(st.session_state.current_data.head(10))
        
        # Display data quality metrics
        st.subheader("Data Quality Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(st.session_state.current_data))
        with col2:
            st.metric("Columns", len(st.session_state.current_data.columns))
        
        # Download button
        csv = st.session_state.current_data.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Data",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"Failed to display final data: {str(e)}")

# ERROR HANDLING

if st.session_state.processing_errors:
    st.subheader("⚠️ Processing Errors")
    for error in st.session_state.processing_errors:
        st.error(f"{error['timestamp']}: {error['message']}")
        st.code(error['stack_trace'], language="python")





