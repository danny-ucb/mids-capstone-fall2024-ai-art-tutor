import openai
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image as PilImage
import os
import bcrypt
import json


def parental_controls_page():
    st.header("Parental Controls")

    # Parental Authentication (Optional)
    with st.expander("Enter Parental Passcode"):
        passcode = st.text_input("Passcode", type="password")
        if passcode != "parent123":  # Example passcode
            st.warning("Incorrect Passcode")
            return

    # Control Options
    st.subheader("Manage Child’s Progress")
    progress_view = st.checkbox("Allow viewing of child’s progress")
    
    st.subheader("Set Time Limits")
    time_limit = st.slider("Set maximum usage time (hours per day)", 0, 5, 1)

    st.subheader("Block or Allow Features")
    block_story_feature = st.checkbox("Block Story Feature")
    block_critique_feature = st.checkbox("Block Critique Feature")

    # Display summary
    st.subheader("Parental Control Summary")
    st.write(f"Time Limit: {time_limit} hours/day")
    st.write("Features Blocked: ", 
             "Story" if block_story_feature else "None", 
             ", Critique" if block_critique_feature else "")

    # Save parental settings (you can use session state, a database, etc.)
    if st.button("Save Settings"):
        # Example of saving parental settings in session state (can also be saved in a DB)
        st.session_state['parental_settings'] = {
            "time_limit": time_limit,
            "block_story_feature": block_story_feature,
            "block_critique_feature": block_critique_feature,
            "progress_view": progress_view
        }
        st.success("Settings saved!")

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

# Function to create the registration form
def registration_page():
    st.title("Create an Account")
    
    username = st.text_input("Enter a Username")
    password = st.text_input("Enter a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if username and password and password == confirm_password:
            credentials = load_credentials()
            if username in credentials:
                st.warning("Username already exists. Please choose another one.")
            else:
                hashed_pw = hash_password(password)
                credentials[username] = hashed_pw.decode('utf-8')
                save_credentials(credentials)
                st.success("Account created successfully! Please login.")
        elif password != confirm_password:
            st.error("Passwords do not match.")

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