# app.py
import streamlit as st
import matplotlib.pyplot as plt
from utils import load_model, get_digit_images
import torch

# Set page
st.set_page_config(page_title="Digit Generator", layout="centered")
st.title("🧠 Handwritten Digit Image Generator")

# Select digit
digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))

# Load model
model = load_model("mnist_cnn.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if st.button("Generate Images"):
    st.markdown(f"### Generated images of digit {digit}")
    images = get_digit_images(digit, count=5)

    # Display images in a row
    cols = st.columns(5)
    for i in range(5):
        img = images[i].squeeze(0).numpy()
        cols[i].image(img, caption=f"Sample {i+1}", width=100, clamp=True)
