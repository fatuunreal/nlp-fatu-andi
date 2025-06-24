import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# load saved model
model_spam = pickle.load(open('model_spam.sav', 'rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))

# judul
st.title("NATURAL LANGUAGE PROCESSING A11.4617")
st.title("Klasifikasi SMS Spam")
st.write("FATU RAHMAT A11.2022.14831")
st.write("ANDI LAKSONO A11.2022.14839")

clean_teks = st.text_input('Masukkan Teks SMS')

spam_detection = ''

if st.button('Cek SMS'):
    predict_spam = model_spam.predict(loaded_vec.fit_transform([clean_teks]))

    if(predict_spam == 0):
        spam_detection = 'SMS Spam'
    else:
        spam_detection = 'SMS Normal'

st.success(spam_detection)
