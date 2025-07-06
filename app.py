import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Monkey patch to fix legacy keras path issue ---
import sys
import tensorflow.keras.preprocessing.text as tf_text
sys.modules['keras.preprocessing.text'] = tf_text

# --- Load model and tokenizer ---
model = load_model('lstm_next_word_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# --- Prediction Function ---
def predict_next_word(model, tokenizer, text, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) > max_sequence_length:
        token_list = token_list[-(max_sequence_length - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return "No match found"

# --- Page Configuration ---
st.set_page_config(page_title="Next Word Predictor", page_icon="🧠", layout="centered")

# --- Custom CSS for styling ---
st.markdown("""
    <style>

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- App Layout ---
st.title("🧠 Next Word Prediction App")
st.markdown("#### ✨ Powered by LSTM & Streamlit")
st.write("Type a phrase or sentence and let the AI guess what comes next!")

with st.container():
    input_text = st.text_input("🔤 Enter your sentence below", placeholder="e.g. Once upon a", key="input")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Predict Next Word"):
            if input_text.strip() == "":
                st.warning("Please enter a valid input to predict.")
            else:
                max_sequence_length = model.input_shape[1] + 1
                predicted_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
                st.success(f"✨ **Predicted Next Word:** `{predicted_word}`")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using `TensorFlow` and `Streamlit`")
