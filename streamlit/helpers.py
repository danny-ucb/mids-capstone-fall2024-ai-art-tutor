import openai
import streamlit as st
import os
import requests
from PIL import Image
import io
import streamlit as st
from io import BytesIO
import base64

# Set up OpenAI API key
openai.api_key = "sk-xMls_bdlW2zesbQwEg1eM_VZII3XtTt9t4NCYOqTPxT3BlbkFJcgWtdu7E4cW1BAXm8KIav9XmjmTiTSSEC7aXl9GJQA"
## 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
## langchain api key = "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"
# Define a function to get a response from GPT-4

def login():
    USERNAME = "abbyp"
    PASSWORD = "capstone2024"
    """A simple login page."""
    st.title("Login")
    
    # Username input
    username = st.text_input("Username")
    
    # Password input
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state['logged_in'] = True
            st.success("Successfully logged in!")
        else:
            st.error("Invalid username or password")

def generate_response(prompt, image_url):
    try:
        # Use the OpenAI API to create chat completion
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Or "gpt-3.5-turbo" if you're using GPT-3.5
        messages=[
            {"role": "system", "content": "You are a helpful assistant for kids interested in drawing and art."},
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
                },
            ],
            }
        ],
        max_tokens=150,
        )            
        
        # Return the response message
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"



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
         font-size: 1.3em;   
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
            font-size: 1.4em;
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
            font-size: 1.2em;
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