import openai
import streamlit as st
import os
import requests
from PIL import Image
import io
import streamlit as st
from io import BytesIO
import base64


# Function to inject custom CSS
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Main page styles */
        .main {
            background-color: #FFFAF0;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        p {
         font-size: 1.0em;   
        }
        h1 {
            color: #FF6347;
            text-align: center;
            font-size: 3em;
        }
        .stButton>button {
            background-color: #ffcccb;
            color: black;
            border-radius: 10px;
            font-size: 1em;
            padding: 10px 20px;
        }
        .stFileUploader>label {
            color: #4682B4;
            font-size: 1.0em;
        }

        /* Sidebar styles */
        .sidebar .sidebar-content {
            background-color: #FFFACD; /* Light golden background */
            color: #FF6347; /* Tomato red text color */
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        .sidebar .sidebar-content h2 {
            color: #FF4500; /* OrangeRed color for headers */
        }
        .sidebar .sidebar-content label {
            font-size: 1.0em;
            color: #4682B4;
        }
        .sidebar .sidebar-content .stRadio>div {
            background-color: #FFD700; /* Bright gold radio buttons */
            border-radius: 10px;
            padding: 10px;
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def encode_image(image):
    # Convert the PIL image to bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # or "JPEG" depending on your image type
    img_bytes = buffered.getvalue()

    # Encode the bytes as base64
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    return base64_image