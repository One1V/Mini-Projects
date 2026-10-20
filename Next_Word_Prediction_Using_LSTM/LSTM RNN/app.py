
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Hamlet Next-Word Prediction")
st.write("Loading your trained LSTM model...")

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("hamlet_model.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_trained_model()
    tokenizer = load_tokenizer()
    st.success("Model and tokenizer loaded successfully!")

    st.write("Model input shape:", model.input_shape)
    st.write("Tokenizer vocabulary:", len(tokenizer.word_index))

    input_text = st.text_input(
        "Enter a sequence of words:",
        value="To be or not to"
    )

    if st.button("Predict Next Word"):
        if not input_text.strip():
            st.warning("Please enter some text.")
        else:
            sequence = tokenizer.texts_to_sequences([input_text])[0]

            if not sequence:
                st.warning("No recognized words. Try a phrase from Hamlet.")
            else:
                max_sequence_len = model.input_shape[1] + 1

                sequence = sequence[-(max_sequence_len - 1):]

                padded = pad_sequences(
                    [sequence],
                    maxlen=max_sequence_len - 1,
                    padding="pre"
                )

                predictions = model.predict(padded, verbose=0)
                predicted_id = int(np.argmax(predictions[0]))
                predicted_word = tokenizer.index_word.get(
                    predicted_id, "Unknown"
                )

                st.success(f"Predicted next word: **{predicted_word}**")

except Exception as e:
    st.error("The application encountered an error.")
    st.exception(e)
