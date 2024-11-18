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
from helpers.general_helpers import *
from helpers.image_helpers import * 
from helpers.api_keys import * 

# Compression
import zipfile

def check_consent(username: str, consent_type: str) -> bool:
    """
    Check if user has given consent for a specific type of data storage.
    
    Args:
        username: The username to check consent for
        consent_type: Type of consent to check ('memory_collection', 'image_collection', 
                     'email_updates')
    Returns:
        bool: True if consent is given, False otherwise
    """
    consent_file = f'user_consents/{username}_consent.json'
    try:
        if os.path.exists(consent_file):
            with open(consent_file, 'r') as f:
                consent_data = json.load(f)
                return consent_data.get(consent_type, False)
        return False
    except Exception as e:
        print(f"Error checking consent: {str(e)}")
        return False

def load_user_consent(username):
    """
    Load user's consent settings from a JSON file.
    Includes backwards compatibility for existing consent files.
    """
    default_consent = {
        "memory_collection": True,  # Default consents
        "image_collection": True,
        "session_summaries": True,
        "email_updates": True,
        "consent_date": None,
        "last_modified": None
    }
    
    consent_file = f'user_consents/{username}_consent.json'
    if os.path.exists(consent_file):
        try:
            with open(consent_file, 'r') as f:
                existing_consent = json.load(f)
            
            # Update existing consent with any missing fields
            updated_consent = default_consent.copy()
            updated_consent.update(existing_consent)
            
            # Save the updated consent if it was modified
            if len(updated_consent) != len(existing_consent):
                with open(consent_file, 'w') as f:
                    json.dump(updated_consent, f)
            
            return updated_consent
        except Exception as e:
            print(f"Error loading consent file: {str(e)}")
            return default_consent
    
    return default_consent

def save_user_consent(username, consent_data):
    """Save user's consent settings to a JSON file"""
    if not os.path.exists('user_consents'):
        os.makedirs('user_consents')
    
    consent_file = f'user_consents/{username}_consent.json'
    consent_data['last_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(consent_file, 'w') as f:
        json.dump(consent_data, f)

def revoke_all_data(username):
    """Handle complete data removal when consent is revoked"""
    try:
        # Delete uploaded images
        user_image_dir = os.path.join("uploaded_images", username)
        if os.path.exists(user_image_dir):
            shutil.rmtree(user_image_dir)
            
        # Delete session data
        session_file = get_user_session_file(username)
        if os.path.exists(session_file):
            os.remove(session_file)
            
        # Delete memories
        delete_user_memories(username)
        
        # Delete consent file
        consent_file = f'user_consents/{username}_consent.json'
        if os.path.exists(consent_file):
            os.remove(consent_file)
            
        return True
    except Exception as e:
        print(f"Error revoking data: {str(e)}")
        return False

