import streamlit as st
from pipeline import get_predictions , load_model
import time


model, tokenizer = load_model()

st.title("Skills Extraction from Job post")

# Input text area for user to input description
user_input = st.text_area("Enter the job description:", "")

# Button to trigger predictions
if st.button("Get Predictions"):
    # Assuming 'model' and 'tokenizer' are already defined
    start_time = time.time()
    predicted_skills = get_predictions(user_input, model, tokenizer, threshold=0.65)
    execution_time = time.time() - start_time

    st.write("Predicted Skills:", predicted_skills)
    st.write(f"Execution Time: {execution_time:.4f} seconds")
