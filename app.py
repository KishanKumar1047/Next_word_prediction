from flask import Flask, request, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Monkey patch to fix legacy keras path issue ---
import sys
import tensorflow.keras.preprocessing.text as tf_text
sys.modules['keras.preprocessing.text'] = tf_text

# --- Initialize Flask app ---
app = Flask(__name__)

# --- Load model and tokenizer ---
model = load_model('lstm_next_word_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# --- Prediction function ---
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

# --- Flask routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        input_text = request.form['text']
        if input_text.strip() != "":
            max_sequence_length = model.input_shape[1] + 1
            prediction = predict_next_word(model, tokenizer, input_text, max_sequence_length)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
