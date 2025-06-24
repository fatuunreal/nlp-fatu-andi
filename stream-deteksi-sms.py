import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved models
model_nb = pickle.load(open('model_nb.pkl', 'rb'))
model_rf = pickle.load(open('model_rf.pkl', 'rb'))

# Load TF-IDF vectorizer (use pre-fitted)
loaded_vec = pickle.load(open("feature_tf-idf.sav", "rb"))

# Judul
st.title("Deteksi SMS Spam")

# Input teks SMS
clean_teks = st.text_input('Masukkan Teks SMS')

spam_detection_nb = ''
spam_detection_rf = ''

if st.button('Cek SMS'):
    teks_transformed = loaded_vec.transform([clean_teks])

    # Prediksi dengan Naive Bayes
    predict_nb = model_nb.predict(teks_transformed)
    spam_detection_nb = 'SMS Normal' if predict_nb[0] else 'SMS Spam'

    # Prediksi dengan Random Forest
    predict_rf = model_rf.predict(teks_transformed)
    spam_detection_rf = 'SMS Normal' if predict_rf[0] else 'SMS Spam'

    st.success(f"Hasil Naive Bayes: {spam_detection_nb}")
    st.success(f"Hasil Random Forest: {spam_detection_rf}")
