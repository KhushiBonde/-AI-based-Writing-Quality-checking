
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    gbr_model = joblib.load('gbr_model.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop() # Stop the app if the model can't be loaded

st.title('Essay Score Prediction App')
st.write('Enter the features below to predict the essay score.')

# Input fields for features
word_count = st.number_input('Word Count', min_value=0, value=100)
sentence_count = st.number_input('Sentence Count', min_value=1, value=5)
avg_sentence_length = st.number_input('Average Sentence Length', min_value=0.0, value=20.0)
flesch_reading_ease = st.number_input('Flesch Reading Ease', min_value=0.0, value=60.0)
vocabulary_diversity = st.number_input('Vocabulary Diversity (0.0 to 1.0)', min_value=0.0, max_value=1.0, value=0.75)

# Create a DataFrame for prediction
input_data = pd.DataFrame([[word_count, sentence_count, avg_sentence_length, flesch_reading_ease, vocabulary_diversity]],
                           columns=['word_count', 'sentence_count', 'avg_sentence_length', 'flesch_reading_ease', 'vocabulary_diversity'])

if st.button('Predict Score'):
    prediction = gbr_model.predict(input_data)[0]
    st.success(f'Predicted Essay Score: {prediction:.2f}')
