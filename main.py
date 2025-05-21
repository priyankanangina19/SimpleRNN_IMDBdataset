import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense

model = load_model('simplernn_imdb.h5')
word_index = imdb.get_word_index()

##step2 : Helper Functionw
#function to decode reviews

def decode_reviews(encoded_review):
    return ' '.join([inverted_word_index.get(i-3, '?') for i in encoded_review])

##function to preprocess text 
def preprocess_text(text):
    words = text.lower().split()
    encoded_review =[word_index.get(word, 2)+3 for word in words ]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


##step3 - prediction  function
def predict_sentiment(review):
   preproceeses_input= preprocess_text(review)

   prediction = model.predict(preproceeses_input)

   sentimemt = 'positive' if prediction[0][0] >0.5 else 'negative'

   return sentimemt, prediction[0][0]

##streamlit app
import streamlit as st
st.title('IMDB Movie review sentiment analysis')
st.write('Enter a movie review to classify it as positive or negative.')

#userinput
user_input = st.text_area('Moview Review')

if st.button('classify'):
    preprocess_input = preprocess_text(user_input)


    ##make prediction
    prediction = model.predict(preprocess_input)
    sentiment ='positive' if prediction[0][0]>0.5 else 'Negative'

    st.write(f'Sentiment :{sentiment}')
    st.write(f'Prediction score: {prediction[0][0]}')
else:
    st.write('Please enter a moview review')
