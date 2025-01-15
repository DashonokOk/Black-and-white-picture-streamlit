import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_file):
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return image

def compress_image(U, S, Vt, k):
    compressed_image = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    return compressed_image

def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

st.title('Сжатие изображения с использованием SVD')

uploaded_file = st.file_uploader("Выберите изображение...", type="jpg")

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.write("Оригинальное изображение")
    st.image(image, use_container_width=True)

    U, S, Vt = np.linalg.svd(image, full_matrices=False)

    k = st.slider('Выберите количество сингулярных чисел (k)', 1, min(image.shape), 50)
    compressed_image = compress_image(U, S, Vt, k)
    normalized_compressed_image = normalize_image(compressed_image)

    st.write(f"Сжатое изображение с k={k}")
    st.image(normalized_compressed_image, use_container_width=True, clamp=True)
