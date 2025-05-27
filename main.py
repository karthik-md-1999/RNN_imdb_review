import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

##Load the IMDB dataset word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key , value in word_index.items()}

##Load the pre_trained model with ReLU activation
model=load_model('simple_rnn_imdb.h5')

##Step2:Helper functions
#Function to decode reviews
def decode_review(enocded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in enocded_review])

#Function to preprocess user innput
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review



##streamlit app

st.title("IMDB movie review sentiment analysis")
st.write('Enter a movie review to classify it as postive or negative.')

#user input
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)
    
    ##Make predicition
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    ##Display the result
    st.write(f"sentiment:{sentiment}")
    st.write(f"Prediction_score:{prediction[0][0]}")
else:
    st.write("Please enter a movie review")    
