import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

img_height, img_width = 180, 180
class_names = ["Kucing 🐱", "Anjing 🐶"] 

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="zahratalitha/anjingkucing",  
        filename="best_model.h5"            
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")   
    img = img.resize((img_width, img_height))        
    img_array = tf.keras.utils.img_to_array(img)     
    img_array = img_array / 255.0                  
    img_array = np.expand_dims(img_array, axis=0)   
    return img_array

# --- Streamlit UI ---
st.title("🐶🐱 Klasifikasi Anjing vs Kucing")
st.write("Upload gambar:")

uploaded_file = st.file_uploader("Upload gambar:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar yang diupload", use_column_width=True)

    img_array = preprocess_image(uploaded_file)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    label = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.subheader("📌 Hasil Prediksi")
    st.write(f"Model memprediksi gambar ini adalah **{label}** dengan probabilitas **{confidence:.2f}%**")
