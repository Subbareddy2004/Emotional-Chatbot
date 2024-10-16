import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
import torch
from google.api_core import exceptions as google_exceptions
import time
from textblob import TextBlob
import subprocess
import sys

def install_rust():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rustup-init"])
        subprocess.check_call(["rustup-init", "-y"])
        # Add Rust to PATH
        import os
        os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.cargo/bin")
    except Exception as e:
        print(f"Failed to install Rust: {e}")

install_rust()

# Set your Gemini API key
genai.configure(api_key='AIzaSyCv6jdoUC3YslqkNj42YNZEhjtWmbBkYEM')

# Load pre-trained emotion detection model and tokenizer
@st.cache_resource
def load_emotion_model():
    tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
    return tokenizer, model

emotion_tokenizer, emotion_model = load_emotion_model()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to detect emotion
def detect_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = emotion_model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    emotion_classes = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    top_emotion_idx = torch.argmax(probs, dim=-1).item()
    top_emotion = emotion_classes[top_emotion_idx]
    
    return top_emotion

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Returns polarity (-1 to 1) and subjectivity (0 to 1)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

# Function to generate a response using Gemini API
def generate_response(emotion, sentiment, user_input, chat_history):
    chat_context = "\n".join([f"{'User' if i%2==0 else 'Bot'}: {msg}" for i, msg in enumerate(chat_history[-5:])])  # Last 5 messages for brevity
    sentiment_polarity, sentiment_subjectivity = sentiment
    
    prompt = f"""Chat history:
{chat_context}

User's current emotion: {emotion}
User's sentiment: Polarity {sentiment_polarity:.2f}, Subjectivity {sentiment_subjectivity:.2f}
User's current message: {user_input}

Respond to the user's message, taking into account their current emotion, sentiment, and the chat history. 
Be empathetic, supportive, and maintain conversational coherence. Provide relevant emotional support.
"""
    
    model = genai.GenerativeModel('gemini-pro')
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            # Here you could add a function to evaluate the response for empathy and relevance
            return response.text
        except google_exceptions.InternalServerError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return "I'm sorry, but I'm having trouble responding right now. Please try again in a moment."
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Callback function to handle sending messages
def send_message():
    if st.session_state.user_input:
        user_message = st.session_state.user_input
        
        emotion = detect_emotion(user_message)
        sentiment = analyze_sentiment(user_message)
        response = generate_response(emotion, sentiment, user_message, [msg for _, msg in st.session_state.chat_history])
        
        st.session_state.chat_history.append(("user", user_message))
        st.session_state.chat_history.append(("bot", response))
        st.session_state.user_input = ""  # Clear the input

        if response.startswith("An error occurred") or response.startswith("I'm sorry, but I'm having trouble"):
            st.error(response)

# Custom CSS for a clean, professional UI
st.markdown("""
<style>
body {
    font-family: Arial, sans-serif;
    background-color: #ffffff;
    color: #333333;
}

.main {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    color: #2c3e50;
    text-align: center;
    padding: 20px 0;
    font-size: 2.5em;
    border-bottom: 2px solid #3498db;
}

.chat-container {
    margin-top: 20px;
}

.message {
    padding: 10px 15px;
    border-radius: 20px;
    margin: 10px 0;
    max-width: 80%;
    line-height: 1.4;
}

.user-message {
    background-color: #e8f5e9;
    color: #1b5e20;
    margin-left: auto;
    text-align: right;
}

.bot-message {
    background-color: #e3f2fd;
    color: #0d47a1;
}

.input-area {
    display: flex;
    margin-top: 20px;
}

.stTextInput > div > div > input {
    border: 1px solid #3498db;
    border-radius: 20px;
    padding: 10px 15px;
    font-size: 16px;
}

.stButton > button {
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 20px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.stButton > button:hover {
    background-color: #2980b9;
    color: white;
}

.emotion-detection {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 15px;
    margin-top: 20px;
    text-align: center;
}

.emotion-detection h3 {
    color: #2c3e50;
    margin-bottom: 10px;
}

.emotion-detection p {
    font-size: 1.2em;
    font-weight: bold;
    color: #e74c3c;
}
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Emotion-Aware Chatbot")

# Create a container for chat history
chat_container = st.container()

# Display chat history
with chat_container:
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'<div class="message user-message"><strong>You:</strong> {message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message bot-message"><strong>Bot:</strong> {message}</div>', unsafe_allow_html=True)

# Input area and send button
col1, col2 = st.columns([5, 1])
with col1:
    st.text_input("Type your message here:", key="user_input", on_change=send_message, label_visibility="collapsed")
with col2:
    st.button("Send", on_click=send_message)

# Display emotion in sidebar
if st.session_state.chat_history:
    last_user_message = [msg for role, msg in st.session_state.chat_history if role == "user"][-1]
    detected_emotion = detect_emotion(last_user_message)
    st.sidebar.markdown(
        f"""
        <div class="emotion-detection">
            <h3>Detected Emotion</h3>
            <p>{detected_emotion}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
