import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# --- Konfigurasi ---
img_height, img_width = 180, 180

# ⚠️ Urutan harus sama seperti saat training (default: alfabetis dari nama folder)
# Kalau folder = ["anjing", "kucing"], maka index 0=anjing, 1=kucing
class_names = ["Anjing 🐶", "Kucing 🐱"]

# --- Load Model dari Hugging Face ---
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="zahratalitha/anjingkucing",  # ganti sesuai repo kamu
        filename="best_model.h5"              # ganti sesuai nama file model
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

# --- Fungsi Preprocessing ---
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")   # pastikan 3 channel
    img = img.resize((img_width, img_height))        # resize sesuai training
    img_array = tf.keras.utils.img_to_array(img)     
    img_array = img_array / 255.0                    # normalisasi (0-1)
    img_array = np.expand_dims(img_array, axis=0)    # tambah batch dimensi
    return img_array

# --- Streamlit UI ---
st.title("🐶🐱 Klasifikasi Anjing vs Kucing")
st.write("Upload gambar, lalu model akan memprediksi apakah itu **anjing** atau **kucing**.")

uploaded_file = st.file_uploader("Upload gambar:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # tampilkan gambar
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)

    # preprocessing
    img_array = preprocess_image(uploaded_file)

    # prediksi
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]).numpy()

    # hasil
    label = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.subheader("📌 Hasil Prediksi")
    st.write(f"Model memprediksi gambar ini adalah **{label}** dengan probabilitas **{confidence:.2f}%**")

    st.write("🔎 Probabilitas masing-masing kelas:")
    for i, class_name in enumerate(class_names):
        st.write(f"- {class_name}: {100 * score[i]:.2f}%")
