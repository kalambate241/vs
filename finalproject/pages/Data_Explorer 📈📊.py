import streamlit as st
import pandas as pd
import plotly.express as px

# Streamlit Page Configuration
st.set_page_config(page_title="Interactive Data Explorer", page_icon="📊", layout="wide")

st.title("📊 Interactive Data Explorer")
st.markdown("Upload a CSV or Excel file to explore data with automatic visualizations and filters.")

# Sidebar Information
st.sidebar.header("📌 How It Works")
st.sidebar.write("1. Upload a CSV or Excel file.")
st.sidebar.write("2. View a preview and summary of the data.")
st.sidebar.write("3. Select columns to generate visualizations.")
st.sidebar.write("4. Apply filters to analyze specific data segments.")
st.sidebar.write("This tool helps you explore and visualize datasets effortlessly!")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Load data
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.success("✅ File uploaded successfully!")
    
    # Display Data Preview
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())
    
    # Display Basic Data Information
    st.subheader("📊 Data Summary")
    st.write(df.describe())
    
    # Select columns for visualization
    st.subheader("📈 Create Visualizations")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_columns:
        x_axis = st.selectbox("Select X-axis", options=numeric_columns)
        y_axis = st.selectbox("Select Y-axis", options=numeric_columns)
        chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Line Chart", "Bar Chart"])
        
        if chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{chart_type} of {x_axis} vs {y_axis}")
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{chart_type} of {x_axis} vs {y_axis}")
        else:
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{chart_type} of {x_axis} vs {y_axis}")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numerical columns available for visualization.")
    
    # Data Filtering
    st.subheader("🔍 Data Filters")
    if categorical_columns:
        selected_column = st.selectbox("Select a column to filter", options=categorical_columns)
        unique_values = df[selected_column].unique()
        selected_value = st.selectbox("Select a value", options=unique_values)
        filtered_data = df[df[selected_column] == selected_value]
        
        st.write("Filtered Data:")
        st.dataframe(filtered_data)
    else:
        st.warning("No categorical columns available for filtering.")
