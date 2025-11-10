import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Main title
st.title("📊 Exploratory Data Analysis")

# Sidebar for controls
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.success("Data loaded successfully!")
    else:
        st.info("Upload a CSV file to begin analysis")

# Main content area
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Data Overview
    with st.container():
        st.subheader("1. Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{df.shape[0]}")
        with col2:
            st.metric("Columns", f"{df.shape[1]}")
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage().sum()/1024:.2f} KB")
    
    #Data Preview
    with st.container():
        st.subheader("2. Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    #Basic Statistics
    with st.container():
        st.subheader("3. Basic Statistics")
        st.dataframe(df.describe().T, use_container_width=True)
    
    # Missing Values
    with st.container():
        st.subheader("4. Missing Values")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Total Missing Values", f"{missing.sum()} entries")
        with col_m2:
            st.metric("Missing %", f"{missing_pct.mean():.2f}%")
        
        st.write("Columns with missing values:")
        st.dataframe(missing.sort_values(ascending=False).head(10), use_container_width=True)
    
    # Duplicates
    with st.container():
        st.subheader("5. Duplicates")
        dup_count = df.duplicated().sum()
        st.metric("Duplicate Rows", f"{dup_count}")
        
        if dup_count > 0:
            st.warning("Found duplicate rows. Consider removing duplicates.")
    
    # Correlation Matrix
    with st.container():
        st.subheader("6. Correlation Matrix")
        
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                corr, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm',
                center=0,
                square=True,
                ax=ax,
                linewidths=0.5
            )
            plt.title('Correlation Matrix', fontsize=16)
            st.pyplot(fig)
        else:
            st.info("No numeric columns found for correlation analysis")
    
    #Interactive Visualization
    with st.container():
        st.subheader("6. Interactive Visualizations")
        
        # Create columns for visualization controls
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            x_col = st.selectbox("X-axis", options=df.columns, index=0)
            
            # Handle numeric vs categorical
            if pd.api.types.is_numeric_dtype(df[x_col]):
                x_type = "Numeric"
            else:
                x_type = "Categorical"
                
            st.caption(f"X-axis type: {x_type}")
        
        with col_v2:
            y_col = st.selectbox("Y-axis", options=df.columns, index=1)
            
            # Handle numeric vs categorical
            if pd.api.types.is_numeric_dtype(df[y_col]):
                y_type = "Numeric"
            else:
                y_type = "Categorical"
                
            st.caption(f"Y-axis type: {y_type}")
        
        # Plot type selection
        plot_type = st.selectbox(
            "Plot Type",
            options=[
                "Scatter Plot", 
                "Line Chart", 
                "Bar Chart", 
                "Box Plot", 
                "Histogram"
            ],
            index=0
        )
        
        # Generate plot
        if st.button("Generate Visualization", type="primary"):
            try:
                if plot_type == "Scatter Plot":
                    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                        plt.title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=16)
                        st.pyplot(fig)
                    else:
                        st.error("Both X and Y axes must be numeric columns")
                
                elif plot_type == "Line Chart":
                    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
                        plt.title(f'Line Chart: {x_col} vs {y_col}', fontsize=16)
                        st.pyplot(fig)
                    else:
                        st.error("Both X and Y axes must be numeric columns")
                
                elif plot_type == "Bar Chart":
                    if pd.api.types.is_numeric_dtype(df[y_col]):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
                        plt.title(f'Bar Chart: {x_col} vs {y_col}', fontsize=16)
                        st.pyplot(fig)
                    else:
                        st.error("Y-axis must be numeric")
                
                elif plot_type == "Box Plot":
                    if pd.api.types.is_object_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
                        plt.title(f'Box Plot: {x_col} vs {y_col}', fontsize=16)
                        st.pyplot(fig)
                    else:
                        st.error("X-axis must be categorical and Y-axis numeric")
                
                elif plot_type == "Histogram":
                    if pd.api.types.is_numeric_dtype(df[y_col]):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(df[y_col], kde=True, ax=ax)
                        plt.title(f'Histogram: {y_col}', fontsize=16)
                        st.pyplot(fig)
                    else:
                        st.error("Y-axis must be numeric")
                
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")
        
        st.caption("Select X/Y axes and plot type above to generate visualizations")
    with st.container():
        st.subheader("3D Scatter Plot")
    
    # Get numeric columns from the current dataframe
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Check if we have enough numeric columns
    if len(numeric_cols) < 3:
        st.info("No enough numeric columns for 3D scatter plot. Please add more numeric columns.")
    else:
        # Create dropdowns for 3D axes
        x_col = st.selectbox("X-axis", numeric_cols, key="x_3d")
        y_col = st.selectbox("Y-axis", numeric_cols, key="y_3d")
        z_col = st.selectbox("Z-axis", numeric_cols, key="z_3d")
        
        # Generate 3D plot button
        if st.button("Generate 3D Scatter Plot", key="3d_button"):
            try:
                # Create 3D plot
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(df[x_col], df[y_col], df[z_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_zlabel(z_col)
                plt.title(f'3D Scatter Plot: {x_col} vs {y_col} vs {z_col}')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating 3D scatter plot: {str(e)}")   
   
    
else:
    st.info("Please upload a CSV file to begin analysis")

# Footer
st.caption("Exploratory Data Analysis | Created with Streamlit🖤🐧")


    
   














 










       






