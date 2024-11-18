# Standard library imports
import os
import io
import json
import uuid
import random
import shutil
import base64
from datetime import datetime, timedelta
from typing import List, Dict
from io import BytesIO

# Third-party imports
import boto3
import bcrypt
import pytz
import openai
import requests
import pandas as pd
import tiktoken
import streamlit as st
from PIL import Image, Image as PilImage
from streamlit_drawable_canvas import st_canvas
from botocore.exceptions import ClientError

# Database
import chromadb
from chromadb.utils import embedding_functions

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Local imports
from helpers.memory_utils import *
from helpers.general_helpers import *
from helpers.consent_utils import * 


# Compression
import zipfile

def encode_image(image):
    # Convert the PIL image to bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # or "JPEG" depending on your image type
    img_bytes = buffered.getvalue()

    # Encode the bytes as base64
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    return base64_image


def organize_upload_directory(base_dir="uploaded_images"):
    """Create and organize the upload directory structure."""
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    # Create user directory if it doesn't exist
    user_dir = os.path.join(base_dir, st.session_state['username'])
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    return user_dir

def save_uploaded_image(uploaded_file):
    """Save uploaded image based on consent preferences."""
    if uploaded_file is None:
        return None
        
    try:
        # Check for image collection consent
        if not st.session_state.consent_settings["image_collection"]:
            # Create temp directory if it doesn't exist
            temp_dir = "temp_session_files"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            # Create a temp file path
            temp_path = os.path.join(temp_dir, f"temp_{st.session_state['username']}_{uploaded_file.name}")
            
            # Save to temp location
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.info("Image collection consent not given. Image will only be available for this session.")
            return temp_path
        
        # If consent is given, save to permanent storage
        base_dir = "uploaded_images"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        user_dir = os.path.join(base_dir, st.session_state['username'])
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        date_dir = os.path.join(user_dir, current_date)
        if not os.path.exists(date_dir):
            os.makedirs(date_dir)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%H-%M-%S')
        name, ext = os.path.splitext(uploaded_file.name)
        base_filename = f"{timestamp}_{name}{ext}"
        save_path = os.path.join(date_dir, base_filename)
        
        # Handle duplicates
        counter = 1
        while os.path.exists(save_path):
            new_filename = f"{timestamp}_{name}({counter}){ext}"
            save_path = os.path.join(date_dir, new_filename)
            counter += 1
        
        # Save the file
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return save_path
        
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

def get_all_user_images():
    """Get all images for the current user with their metadata."""
    user_dir = os.path.join("uploaded_images", st.session_state['username'])
    if not os.path.exists(user_dir):
        return []
        
    image_files = []
    # Walk through all subdirectories
    for root, dirs, files in os.walk(user_dir):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                filepath = os.path.join(root, filename)
                # Get folder name (date) from path
                date_folder = os.path.basename(os.path.dirname(filepath))
                
                # Get file creation/modification time
                timestamp = os.path.getctime(filepath)
                image_files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'size': os.path.getsize(filepath),
                    'date_folder': date_folder
                })
    
    return image_files
def format_file_size(size_in_bytes):
    """Convert file size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} GB"

def create_image_archive(image_files):
    """Create a ZIP archive of all images"""
    import io
    import zipfile
    
    # Create a BytesIO object to store the ZIP file
    zip_buffer = io.BytesIO()
    
    # Create the ZIP file
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for image_info in image_files:
            # Add each image to the ZIP file
            zip_file.write(
                image_info['filepath'], 
                arcname=image_info['filename']  # Use original filename in the ZIP
            )
    
    # Reset buffer position
    zip_buffer.seek(0)
    return zip_buffer

def cleanup_temp_files():
    """Clean up temporary session files."""
    try:
        temp_dir = "temp_session_files"
        if os.path.exists(temp_dir):
            # Only remove files for current user
            user_pattern = f"temp_{st.session_state['username']}_"
            for filename in os.listdir(temp_dir):
                if filename.startswith(user_pattern):
                    try:
                        os.remove(os.path.join(temp_dir, filename))
                    except Exception as e:
                        print(f"Error removing temp file {filename}: {str(e)}")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

# Initialize a session using your credentials
session = boto3.Session(
    aws_access_key_id='AKIA6IY35YCLBJCCEZX6',
    aws_secret_access_key='nNxRSmCPsUHjBF5Te9oPXOX7G2dNFQ7UuIGSvU4y',
)

# Initialize S3 resource
s3 = session.resource('s3')

# Specify the bucket name
bucket_name = 'artbuddy-image-bucket'

def upload_image_and_get_url(image_path, username):
    """
    Upload an image to S3 with a filename containing username and timestamp.
    
    :param image_path: Path to the local image file.
    :param username: Username of the person uploading the image.
    :return: The public URL of the uploaded image if successful, None otherwise.
    """
    try:
        # Ensure the image file exists
        if not os.path.isfile(image_path):
            print(f'Image {image_path} does not exist.')
            return None
        
        # Get file extension from original file
        file_extension = os.path.splitext(image_path)[1].lower()
        
        # Create timestamp in format: YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create sanitized filename: username_timestamp.extension
        sanitized_username = ''.join(c for c in username if c.isalnum() or c in '-_')
        file_name = f"user_uploads/{sanitized_username}/{timestamp}{file_extension}"
        
        # Upload the image
        s3.Bucket(bucket_name).upload_file(
            image_path, 
            file_name,
            ExtraArgs={'ContentType': f'image/{file_extension[1:]}'}  # Set proper content type
        )
        print(f'Successfully uploaded {image_path} as {file_name}')
        
        # Construct the URL
        bucket_location = s3.meta.client.get_bucket_location(Bucket=bucket_name)
        region = bucket_location['LocationConstraint'] or 'us-east-1'
        if region == 'us-east-1':
            url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
        else:
            url = f"https://{bucket_name}.s3-{region}.amazonaws.com/{file_name}"
        
        return url
    except NoCredentialsError:
        print('Credentials not available')
        return None
    except PartialCredentialsError:
        print('Incomplete credentials provided')
        return None
    except Exception as e:
        print(f'Error: {e}')
        return None


# def download_image_requests(url, file_name):
#     response = requests.get(url)
#     if response.status_code == 200:
#         with open(file_name, 'wb') as file:
#             file.write(response.content)
#     else:
#         pass
