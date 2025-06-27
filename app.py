import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('next_word_lstm.keras')

with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)

#function to predict next word
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_next_word(model, tokenizer, text, max_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

   
    if len(token_list) >= max_len:
        token_list = token_list[-(max_len - 1):]
    
   
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')

  
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]  

    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None  

# streamlit app
st.title('Next Word Prediction With LSTM')
input_text = st.text_input('enter the sequence of words','To be or not to')
if st.button('Predict Next word'):
    max_len = model.input_shape[1] + 1
    next_word = predict_next_word(model,tokenizer,input_text,max_len)
    st.write(f'Next Word: {next_word}')
