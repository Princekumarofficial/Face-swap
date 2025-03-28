import streamlit as st
import numpy as np
from PIL import Image
from model2 import main_face_swap

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

# Streamlit app
st.title("Face Swap App")

st.write("Upload two images to perform face swapping.")

# Upload images
uploaded_file1 = st.file_uploader("Choose the first image", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Choose the second image", type=["jpg", "jpeg", "png"])

if uploaded_file1 and uploaded_file2:
    # Load images
    img1 = load_image(uploaded_file1)
    img2 = load_image(uploaded_file2)

    # Display uploaded images
    st.image(img1, caption="First Image", use_column_width=True)
    st.image(img2, caption="Second Image", use_column_width=True)

    # Perform face swap
    result = main_face_swap(img1, img2)

    # Display result
    st.image(result, caption="Face Swapped Image", use_column_width=True)