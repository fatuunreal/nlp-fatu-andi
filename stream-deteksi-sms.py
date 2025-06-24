import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
model_spam_rf = pickle.load(open('model_rf.sav', 'rb'))
model_spam_nb = pickle.load(open('model_nb.sav', 'rb'))

# Load TF-IDF vocabulary
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))

# judul
st.title("NATURAL LANGUAGE PROCESSING A11.4617")
st.title("Deteksi SMS Spam")
st.write("FATU RAHMAT A11.2022.14831")
st.write("ANDI LAKSONO A11.2022.14839")

# Input teks SMS
clean_teks = st.text_input('Masukkan Teks SMS')

spam_detection_rf = ''
spam_detection_nb = ''

if st.button('Cek SMS'):
    teks_transformed = loaded_vec.transform([clean_teks])

    # Prediksi dengan Random Forest
    predict_rf = model_spam_rf.predict(teks_transformed)
    spam_detection_rf = 'SMS Normal' if predict_rf[0] else 'SMS Spam'

    # Prediksi dengan Naive Bayes
    predict_nb = model_spam_nb.predict(teks_transformed)
    spam_detection_nb = 'SMS Normal' if predict_nb[0] else 'SMS Spam'

    st.success(f"Hasil Random Forest: {spam_detection_rf}")
    st.success(f"Hasil Naive Bayes: {spam_detection_nb}")
