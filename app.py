import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Memuat model yang sudah dilatih
loaded_model = load_model('model.h5', compile=False)

# Fungsi untuk memprediksi kelas gambar
def predict_image(image):
    # Memuat dan mengubah ukuran gambar
    img = load_img(image, target_size=(128, 128))

    # Mengonversi gambar menjadi array dan menormalisasi
    img_array = img_to_array(img) / 255.0  # Normalisasi nilai piksel ke rentang [0, 1]

    # Menambahkan dimensi tambahan untuk batch
    img_array = np.expand_dims(img_array, axis=0)

    # Melakukan prediksi
    predictions = loaded_model.predict(img_array)

    # Mendapatkan indeks kelas dengan probabilitas tertinggi
    predicted_class_index = np.argmax(predictions[0])

    # Mapping indeks kelas ke label
    class_mapping = {0: 'Healthy', 1: 'Powdery', 2:'Rust'}
    predicted_class_label = class_mapping[predicted_class_index]

    return predicted_class_label, img

# Membuat antarmuka pengguna Streamlit
st.title('Plant Disease Detection App')
st.write('Unggah gambar daun tanaman untuk diklasifikasikan.')

# Menampilkan contoh gambar dari masing-masing kelas
st.subheader('Contoh Gambar:')
col1, col2, col3 = st.columns(3)

with col1:
    st.image('871bbbd18a4560e7.jpg', caption='Healthy', use_column_width=True)
with col2:
    st.image('fff094a1d85289a5.jpg', caption='Powdery', use_column_width=True)
with col3:
    st.image('ffb7de908858884a.jpg', caption='Rust', use_column_width=True)

# Widget untuk mengunggah gambar
uploaded_file = st.file_uploader('Pilih gambar...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    st.image(uploaded_file, caption='Gambar yang diunggah', use_column_width=True)

    # Prediksi kelas gambar
    label, img = predict_image(uploaded_file)

    # Tampilkan hasil prediksi
    st.write(f'Gambar diprediksi sebagai: **{label}**')
