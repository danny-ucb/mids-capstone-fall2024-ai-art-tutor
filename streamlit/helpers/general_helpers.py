#general_helpers.py
# Standard library imports
import os
import io
from io import BytesIO
import json
import random
import shutil
from datetime import datetime, timedelta
from typing import List, Dict

# Third-party imports
import boto3
import bcrypt
import pytz
import openai
import pandas as pd
import tiktoken
import streamlit as st
from PIL import Image as PilImage
from streamlit_drawable_canvas import st_canvas

# Database
import chromadb

# LangChain imports
from langchain.chat_models import ChatOpenAI  # Removed duplicate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Local imports
from helpers.memory_utils import *
from helpers.image_helpers import * 
from helpers.api_keys import * 
from helpers.consent_utils import * 

# Compression
import zipfile

# Function to inject custom CSS
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Main page styles with default black text */
        .main {
            background-color: #FFFAF0;
            font-family: 'Comic Sans MS', cursive, sans-serif;
            color: black !important;
        }

        /* Set all text elements to black by default */
        h1, h2, h3, h4, h5, h6, p, span, div, label, text {
            color: black !important;
        }

        /* Specific heading styles */
        .css-10trblm {  /* Streamlit main title class */
            color: black !important;
        }
        
        .css-1629p8f h1 {  /* Streamlit header class */
            color: black !important;
        }
        
        .css-1629p8f h2 {
            color: black !important;
        }

        /* Specific style for subheader */
        .css-10trblm.e16nr0p31 {
            color: black !important;
        }

        /* Style for all text elements in streamlit */
        .stMarkdown, .stText {
            color: black !important;
        }

        /* Button styling */
        .stButton>button,
        .stDownloadButton>button,
        button[kind="primary"],
        button[kind="secondary"],
        .stButton button,
        div[data-testid="stDownloadButton"] button {
            background-color: #ffcccb !important;
            color: black !important;
            border-radius: 15px !important;
            font-size: 1em !important;
            padding: 15px !important;
            width: 100% !important;
            margin: 5px 0 !important;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1) !important;
            border: none !important;
            font-family: 'Comic Sans MS', cursive, sans-serif !important;
            transition: all 0.3s ease !important;
        }

        /* Button hover effects */
        .stButton>button:hover,
        .stDownloadButton>button:hover,
        button[kind="primary"]:hover,
        button[kind="secondary"]:hover,
        .stButton button:hover,
        div[data-testid="stDownloadButton"] button:hover {
            background-color: #ffb6b5 !important;
            cursor: pointer !important;
            transform: translateY(-2px) !important;
        }

        /* Button active state */
        .stButton>button:active,
        .stDownloadButton>button:active,
        button[kind="primary"]:active,
        button[kind="secondary"]:active,
        .stButton button:active,
        div[data-testid="stDownloadButton"] button:active {
            background-color: #ffa19f !important;
            transform: translateY(0px) !important;
        }

        /* Input field container styling */
        .stTextInput>div,
        .stTextArea>div,
        .stNumberInput>div {
            background-color: transparent !important;
        }

        /* Input fields - more specific selectors */
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stNumberInput>div>div>input,
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {
            background-color: white !important;
            color: black !important;
            border: 1px solid #cccccc !important;
            border-radius: 10px !important;
            padding: 10px !important;
            font-size: 1em !important;
            font-family: 'Comic Sans MS', cursive, sans-serif !important;
            width: 100% !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        }

        /* Base web overrides */
        div[data-baseweb="base-input"],
        div[data-baseweb="textarea"],
        div[data-baseweb="input"] {
            background-color: white !important;
        }

        /* Focus state */
        .stTextInput>div>div>input:focus,
        .stTextArea>div>div>textarea:focus,
        .stNumberInput>div>div>input:focus,
        div[data-baseweb="input"] input:focus,
        div[data-baseweb="textarea"] textarea:focus {
            border-color: #ffcccb !important;
            box-shadow: 0 0 0 2px rgba(255,204,203,0.2) !important;
            outline: none !important;
        }


        /* File uploader styling - more specific selectors */
        .stFileUploader,
        div[data-testid="stFileUploader"] {
            background-color: #f0f2f6 !important;
        }


        /* File uploader container - more specific */
        .stFileUploader > div:first-child,
        div[data-testid="stFileUploader"] > div:first-child,
        div[data-testid="stFileUploadDropzone"] {
            background-color: #f0f2f6 !important;
            border-radius: 10px !important;
            padding: 20px !important;
            border: 2px dashed #cccccc !important;
        }

        /* File uploader text - more specific */
        .stFileUploader label,
        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploadDropzone"] span,
        div[data-testid="stFileUploadDropzone"] p {
            color: black !important;
        }


        /* Browse files button */
        .stFileUploader button,
        div[data-testid="stFileUploader"] button,
        button[data-testid="stFileUploaderUploadButton"],
        div[data-testid="stFileUploadDropzone"] button {
            background-color: #ffcccb !important;
            color: black !important;
            padding: 8px 16px !important;
            font-size: 0.9em !important;
            border-radius: 10px !important;
            border: none !important;
            margin: 4px !important;
            width: auto !important;
            min-width: 120px !important;
            height: auto !important;
            font-family: 'Comic Sans MS', cursive, sans-serif !important;
        }
        
        /* Hover state for browse button - more specific */
        .stFileUploader button:hover,
        div[data-testid="stFileUploader"] button:hover,
        button[data-testid="stFileUploaderUploadButton"]:hover,
        div[data-testid="stFileUploadDropzone"] button:hover {
            background-color: #ffb6b5 !important;
            cursor: pointer !important;
        }
        
        /* Drop zone specific styling */
        div[data-testid="stFileUploadDropzone"] {
            background-color: #f0f2f6 !important;
            border: 2px dashed #cccccc !important;
            border-radius: 10px !important;
        }

        /* Sidebar text */
        .css-1d391kg, .css-1lycqwe {  /* Streamlit sidebar classes */
            color: black !important;
        }

        /* Tab labels */
        .stTabs [data-baseweb="tab"] {
            color: black !important;
        }

        /* File uploader text */
        .stFileUploader label {
            color: black !important;
        }

        /* Ensure all other text elements are black */
        [class*="st"] {
            color: black !important;
        }
        
        /* Exception for specific elements that need different colors */
        .css-1dp5vir a, .css-1dp5vir a:visited {  /* Keep links a different color */
            color: #0066cc !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

