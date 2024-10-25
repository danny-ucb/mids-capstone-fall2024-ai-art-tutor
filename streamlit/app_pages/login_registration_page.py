import openai
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image as PilImage
import os
import bcrypt
import json
import random
from helpers import * 
from datetime import datetime, timedelta

# Function to load and save credentials
def load_credentials():
    try:
        with open('credentials.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_credentials(credentials):
    with open('credentials.json', 'w') as f:
        json.dump(credentials, f)

# Password Hashing Function
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)


def registration_page():
    st.title("Create an Account")
    
    # Initialize session state variables
    if 'verification_code' not in st.session_state:
        st.session_state.verification_code = None
    if 'registration_step' not in st.session_state:
        st.session_state.registration_step = 'initial'
    
    # Form inputs
    username = st.text_input("Enter a Username")
    email = st.text_input("Enter your Email")
    password = st.text_input("Enter a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Register"):
        if username and email and password and password == confirm_password:
            credentials = load_credentials()
            if username in credentials:
                st.warning("Username already exists. Please choose another one.")
            else:
                # Generate verification code only if not already generated
                if st.session_state.verification_code is None:
                    st.session_state.verification_code = random.randint(10000, 99999)
                    st.session_state.registration_step = 'verify'
                    st.session_state.temp_username = username
                    st.session_state.temp_password = password
                    
                    # Send verification email
                    try:
                        send_verification_email(
                            sender="aiartbuddy@gmail.com",
                            recipient=email,
                            subject="Verify your AI Art Buddy account",
                            verification_code=st.session_state.verification_code
                        )
                        st.success("Please check your email for verification code.")
                    except Exception as e:
                        st.error(f"Error sending verification email: {str(e)}")
                        st.session_state.registration_step = 'initial'
                        st.session_state.verification_code = None
        
        elif password != confirm_password:
            st.error("Passwords do not match.")
    
    # Show verification input only if we're in the verify step
    if st.session_state.registration_step == 'verify':
        input_code = st.text_input("Enter the verification code")
        if st.button("Verify Code"):
            if len(input_code) != 5:
                st.write("Code must be 5 digits")
            else:
                if int(input_code) == st.session_state.verification_code:
                    credentials = load_credentials()
                    hashed_pw = hash_password(st.session_state.temp_password)
                    credentials[st.session_state.temp_username] = hashed_pw.decode('utf-8')
                    save_credentials(credentials)
                    st.success("Account created successfully! Please login.")
                    
                    # Reset the registration state
                    # st.session_state.verification_code = None
                    # st.session_state.registration_step = 'initial'
                    # st.session_state.temp_username = None
                    # st.session_state.temp_password = None
                    # st.rerun()
                else:
                    st.error("Verification code does not match!")

    # Add a cancel button to reset the registration process
    if st.session_state.registration_step == 'verify':
        if st.button("Cancel Registration"):
            st.session_state.verification_code = None
            st.session_state.registration_step = 'initial'
            st.session_state.temp_username = None
            st.session_state.temp_password = None
            st.rerun()


# Function to create the login form
def login_page():
    st.title("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        credentials = load_credentials()
        if username in credentials:
            hashed_password = credentials[username]
            if check_password(password, hashed_password.encode('utf-8')):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Login successful!")
            else:
                st.error("Incorrect password.")
        else:
            st.error("Username does not exist. Please register.")