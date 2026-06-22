import os
import re
import joblib
import pandas as pd
import numpy as np
import nltk
import textstat
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Initialize FastAPI App
app = FastAPI(title="Writing Quality Checker API")

# Enable CORS for React Frontend (typically running on localhost:5173 or * for ease of local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download necessary NLTK corpora at startup
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Warning: NLTK downloading error: {e}")

# Load the trained model using absolute path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'gbr_model.pkl')

try:
    gbr_model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Critical Error: Failed to load model from {model_path}: {e}")
    gbr_model = None

# Input schemas
class EssayRequest(BaseModel):
    text: str

# Helper functions for preprocessing and feature extraction
def clean_text(text: str) -> str:
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_remove_stopwords(text: str):
    try:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return filtered_tokens
    except Exception:
        # Fallback split if NLTK fails
        words = text.split()
        return [w for w in words if len(w) > 2]

def get_sentence_and_avg_word_length(text: str):
    try:
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)
        
        total_words = 0
        for sent in sentences:
            total_words += len(word_tokenize(sent))
            
        avg_sentence_length = total_words / sentence_count if sentence_count > 0 else 0
        return sentence_count, avg_sentence_length
    except Exception:
        # Fallback if sentence tokenization fails
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        sentence_count = max(1, len(sentences))
        words = text.split()
        avg_sentence_length = len(words) / sentence_count
        return sentence_count, avg_sentence_length

def calculate_vocabulary_diversity(tokens):
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)

@app.post("/api/evaluate")
async def evaluate_text(payload: EssayRequest):
    if not gbr_model:
        raise HTTPException(status_code=500, detail="Prediction model is not loaded on the server.")
        
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    # --- 1. NATURAL / DISPLAY METRICS (For UI rendering) ---
    # Total word count in raw text
    display_words = text.split()
    display_word_count = len(display_words)
    
    # Total sentence count in raw text (using NLTK sent_tokenize on raw text to preserve periods)
    try:
        display_sentences = sent_tokenize(text)
        display_sentence_count = len(display_sentences)
    except Exception:
        # Fallback split
        display_sentences = [s.strip() for s in text.split('.') if s.strip()]
        display_sentence_count = max(1, len(display_sentences))
        
    display_avg_sentence_length = display_word_count / display_sentence_count if display_sentence_count > 0 else 0.0
    
    # Vocabulary diversity in raw text (lexical diversity of lowercased alphanumeric words)
    display_cleaned_words = [re.sub(r'[^a-zA-Z0-9]', '', w.lower()) for w in display_words]
    display_cleaned_words = [w for w in display_cleaned_words if w]
    display_vocabulary_diversity = (len(set(display_cleaned_words)) / len(display_cleaned_words)) if display_cleaned_words else 0.0
    
    # --- 2. MODEL METRICS (For GBR prediction, aligned with Colab training bugs) ---
    # clean_text strips all punctuation, including periods.
    cleaned_for_model = clean_text(text)
    tokens_for_model = tokenize_and_remove_stopwords(cleaned_for_model)
    
    model_word_count = len(tokens_for_model)
    model_sentence_count, model_avg_sentence_length = get_sentence_and_avg_word_length(cleaned_for_model)
    model_flesch_reading_ease = textstat.flesch_reading_ease(text)
    model_vocabulary_diversity = calculate_vocabulary_diversity(tokens_for_model)
    
    # Pack features into DataFrame matching exact training feature structure
    input_data = pd.DataFrame(
        [[model_word_count, model_sentence_count, model_avg_sentence_length, model_flesch_reading_ease, model_vocabulary_diversity]],
        columns=['word_count', 'sentence_count', 'avg_sentence_length', 'flesch_reading_ease', 'vocabulary_diversity']
    )
    
    # Predict score
    try:
        prediction = gbr_model.predict(input_data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
        
    # --- 3. CONSTRUCTIVE FEEDBACK (Based on natural metrics) ---
    feedback = []
    
    # Sentence Length Feedback
    if display_avg_sentence_length > 25:
        feedback.append({
            "type": "sentence_length",
            "impact": "negative",
            "message": f"Your average sentence length is high ({display_avg_sentence_length:.1f} words). Consider breaking down long, complex sentences to improve readability."
        })
    elif 10 <= display_avg_sentence_length <= 20:
        feedback.append({
            "type": "sentence_length",
            "impact": "positive",
            "message": "Excellent average sentence length! Your sentences are well-paced and easy to follow."
        })
        
    # Readability Feedback
    if model_flesch_reading_ease < 30:
        feedback.append({
            "type": "readability",
            "impact": "negative",
            "message": f"Readability score ({model_flesch_reading_ease:.1f}) is very low. The text might feel extremely dense or academic. Try using simpler phrasing."
        })
    elif model_flesch_reading_ease < 50:
        feedback.append({
            "type": "readability",
            "impact": "neutral",
            "message": f"Readability score ({model_flesch_reading_ease:.1f}) is fairly low (difficult to read). Suitable for college-level audiences, but could be simplified."
        })
    elif 60 <= model_flesch_reading_ease <= 80:
        feedback.append({
            "type": "readability",
            "impact": "positive",
            "message": f"Great readability score ({model_flesch_reading_ease:.1f})! The text is clear and readable for a general audience."
        })

    # Vocabulary Diversity Feedback
    if display_word_count > 30:
        if display_vocabulary_diversity < 0.45:
            feedback.append({
                "type": "vocabulary",
                "impact": "negative",
                "message": f"Vocabulary diversity is low ({display_vocabulary_diversity:.2f}). You are repeating the same words frequently. Try using synonyms."
            })
        elif display_vocabulary_diversity > 0.7:
            feedback.append({
                "type": "vocabulary",
                "impact": "positive",
                "message": f"Exceptional lexical diversity ({display_vocabulary_diversity:.2f})! You use a rich, varied set of words."
            })

    # Length Feedback
    if display_word_count < 100:
        feedback.append({
            "type": "length",
            "impact": "neutral",
            "message": "The essay is quite short. To build a stronger and more convincing argument, consider expanding your points and details."
        })
        
    # Final output (Return natural metrics for UI display)
    return {
        "score": max(0.0, round(float(prediction), 2)),
        "metrics": {
            "word_count": int(display_word_count),
            "sentence_count": int(display_sentence_count),
            "avg_sentence_length": round(float(display_avg_sentence_length), 2),
            "flesch_reading_ease": round(float(model_flesch_reading_ease), 2),
            "vocabulary_diversity": round(float(display_vocabulary_diversity), 3)
        },
        "feedback": feedback
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
