import streamlit as st
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt
import io
import os
import traceback
from dotenv import load_dotenv
import numpy as np
import warnings     
import altair as alt # Added for interactive charting

# --- Page Configuration ---
st.set_page_config(
    page_title="Query Your CSV with AI",
    page_icon="📊",
    layout="wide",
)

load_dotenv()
# Replace with your actual API key or set it as an environment variable
# The key in the prompt has been removed for security
client = OpenAI(
  api_key= os.getenv("OPENAI_API_KEY")
)

# --- App Title and Description ---
st.title("📊 Query Your CSV Data with AI")
st.markdown("""
Welcome to your personal data analyst! Upload your CSV file, and ask questions in plain English. 
The AI will generate insights and visualizations for you.
""")

# --- Helper Functions ---

def get_llm_response(query, df):
    """Gets the response from the LLM to generate Python code."""
    try:        
        # Prepare the prompt with information about the dataframe
        data_info = f"The DataFrame `df` has columns: {list(df.columns)}\n"
        data_info += f"Here are the first 3 rows:\n{df.head(3).to_string()}\n\n"

        prompt = f"""
        You are a helpful data analyst that generates Python code to be run in a Streamlit app.
        You are given a pandas DataFrame named `df`.
        Your task is to write Python code to answer the user's query.
        
        Here is information about the DataFrame:
        {data_info}

        The user's query is: "{query}"

        Based on this query, write Python code that may use pandas and/or Altair and/or streamlit to generate the answer.
        - The DataFrame is already available as a variable `df`.
        - The streamlit library is imported as `st`.
        - If the user asks for a chart, use the Altair library to create an interactive chart. The code should define the chart using `alt.Chart(...)` and include tooltips. Finally, display the chart using `st.altair_chart(chart)`. Make the chart visually appealing with clear labels and titles. **ABSOLUTELY DO NOT call `plt.show()` or `st.pyplot()`. The main application code will display the plot automatically.**
        - If the user asks for data, a number, or any textual answer, use `st.write()` to display the result. For example: `st.write(df.head())` or `st.write(f"The average profit is {{avg_profit}}")`.
        - The code should be a single block of executable Python.
        - Do not include any explanation, just the code.
        - Do not include the python markdown tag.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst that generates python code for displaying text or data or beautiful charts for a Streamlit app as per the demand of user only."},
                {"role": "user", "content": prompt}
            ], 
            max_tokens=500,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the OpenAI API: {e}")
        st.code(traceback.format_exc())
        return None

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("🔗 Connections")
    # openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    st.markdown("---")
    st.header("Instructions")
    st.markdown("""
        **Upload CSV**: Upload the CSV file you want to analyze.
    """)


# --- Main Application Logic ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        num_rows = len(df)
        print(f"Number of rows (using len()): {num_rows}")


        if df is not None and not df.empty:
            st.success("Successfully loaded CSV file!")
            
            st.subheader("Data Preview")
            st.write(df.head())

            st.markdown("---")
            st.header("💬 Ask a question about your data")
            query = st.text_input("e.g., 'Plot a bar chart of total sales by region' or 'What is the average profit?'")

            if st.button("Get Answer"):
                if query:
                    with st.spinner("AI is thinking..."):
                        generated_code = get_llm_response(query, df)

                        if generated_code:
                            # st.subheader("Generated Code")
                            # st.code(generated_code, language="python")

                            st.subheader("Result")
                            try:
                                # Prepare the execution environment
                                local_vars = {"df": df, "pd": pd, "plt": plt, "st": st, "alt": alt}
                                # Remove markdown code block markers if present
                                cleaned_code = generated_code.replace("```python", "").replace("```", "").strip()
                                # Execute the code
                                exec(cleaned_code, {}, local_vars)
                                
                                # Capture plot if generated
                                # fig = None
                                fig = local_vars.get("plt").gcf()
                                if fig and len(fig.get_axes()) > 0:
                                    st.pyplot(fig)
                                    
                            except Exception as e:
                                st.error("An error occurred while executing the generated code:")
                                st.code(traceback.format_exc())
                else:
                    st.warning("Please enter a question.")
    except Exception as e:
        st.error("An error occurred in the main application flow:")
        st.code(traceback.format_exc())
else:
    st.info("Please provide your API Key and upload a CSV file to get started.")