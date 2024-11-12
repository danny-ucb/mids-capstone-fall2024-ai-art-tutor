# Standard library imports
import os
import json
import random
from datetime import datetime, timedelta

# Third-party imports
import bcrypt
import openai
import streamlit as st
from PIL import Image as PilImage
from streamlit_drawable_canvas import st_canvas

# Local imports
from helpers.general_helpers import *
from helpers.api_keys import * 

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

# def check_password(password, hashed):
#     return bcrypt.checkpw(password.encode(), hashed)

def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)


def registration_page():
    inject_custom_css()
    st.title("Create an Account")
    st.write("To create an account, you’ll need a parent’s permission. We require a parent’s email address to verify your account and get their consent for you to use our application.")
    
    # Initialize session state variables
    if 'verification_code' not in st.session_state:
        st.session_state.verification_code = None
    if 'registration_step' not in st.session_state:
        st.session_state.registration_step = 'initial'

    # Form inputs
    username = st.text_input("Enter a Username")
    email = st.text_input("Enter a Parent's Email")
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
                    st.session_state.parent_email = email
                    
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

                    credentials[st.session_state.temp_username] = {
                        "password": hashed_pw.decode('utf-8'),
                        "parent_email": st.session_state.parent_email
                    }

                    save_credentials(credentials)
                    st.success("Account created successfully! Please login.")

                else:
                    st.error("Verification code does not match!")

    # Add a cancel button to reset the registration process
    if st.session_state.registration_step == 'verify':
        if st.button("Cancel Registration"):
            st.session_state.verification_code = None
            st.session_state.registration_step = 'initial'
            st.session_state.temp_username = None
            st.session_state.temp_password = None
            st.session_state.parent_email = None
            st.rerun()


# Function to create the login form
def login_page():
    inject_custom_css()
    st.title("Login")
    
    # Initialize session state variables if they don't exist
    if 'login_attempt' not in st.session_state:
        st.session_state.login_attempt = False
        
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login") or st.session_state.login_attempt:
        st.session_state.login_attempt = True
        
        if username and password:  # Only proceed if both fields are filled
            credentials = load_credentials()
            if username in credentials:
                hashed_password = credentials[username]['password']
                if check_password(password, hashed_password.encode('utf-8')):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.session_state['parent_email'] = credentials[username]['parent_email']
                    st.session_state.login_attempt = False  # Reset login attempt
                    st.rerun()  # Force a rerun to update the page
                else:
                    st.error("Incorrect password.")
                    st.session_state.login_attempt = False
            else:
                st.error("Username does not exist. Please register.")
                st.session_state.login_attempt = False
        else:
            st.warning("Please enter both username and password.")
            st.session_state.login_attempt = False


def send_verification_email(sender, recipient, subject, verification_code, body_text=None, body_html=None):
    """
    Send an email using AWS SES with a verification code for AI Art Buddy.
    
    Parameters:
    sender (str): Email address of the sender
    recipient (str or list): Email address(es) of recipient(s)
    subject (str): Subject line of the email
    verification_code (str): Verification code to include in the email
    body_text (str): Optional plain text version of the email
    body_html (str): Optional HTML version of the email
    
    Returns:
    dict: Response from AWS SES
    """
    # Specify your AWS region (example: 'us-east-1')
    region = 'us-east-1'  # Change to your region
    
    # Create a new SES resource with the specified region
    ses_client = boto3.client('ses', region_name=region)
    
    # Default email bodies with the verification code and theme
    default_body_text = f"Your verification code for AI Art Buddy is: {verification_code}\n\n"
    # default_body_html = f"""
    default_body_html = f"""
    <html>
        <head>
            <style>
                /* General styling */
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    color: #333333;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    width: 100%;
                    padding: 20px;
                    background-color: #ffffff;
                    border-radius: 8px;
                    box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    background-color: #3b5998;
                    color: white;
                    padding: 10px;
                    text-align: center;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                }}
                .header h1 {{
                    margin: 0;
                }}
                .content {{
                    padding: 20px;
                    text-align: center;
                }}
                .verification-code {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #ff6347;
                    margin: 10px 0;
                }}
                .footer {{
                    margin-top: 30px;
                    font-size: 12px;
                    color: #999999;
                    text-align: center;
                }}
                .coppa-info {{
                    text-align: left;
                    font-size: 14px;
                    color: #555555;
                    margin-top: 20px;
                    padding-top: 20px;
                    border-top: 1px solid #dddddd;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to AI Art Buddy!</h1>
                </div>
                <div class="content">
                    <p>Thank you for signing up!</p>
                    <p>Your verification code is:</p>
                    <p class="verification-code">{verification_code}</p>
                    <p>Please review the parental consent information below. By entering the verification code, you confirm your consent to complete the sign-up process.</p>
                </div>
                <div class="coppa-info">
                    <h2>Parental Consent Required</h2>
                    <p>In compliance with the Children's Online Privacy Protection Act (COPPA), we require parental consent for children under 13 to use our services. As part of the registration process, we collect the following information:</p>
                    <ul>
                        <li><strong>Account Information:</strong> Your child’s username and age, used to personalize their experience.</li>
                        <li><strong>Artwork Uploads:</strong> Images uploaded by your child, solely for the purpose of providing art feedback.</li>
                        <li><strong>Parental Contact:</strong> This email address, used to communicate important information regarding your child’s account and consent verification.</li>
                    </ul>
                    <p>As a parent, you have the right to review and delete your child’s information at any time. If you have questions or wish to modify your consent, please contact us at aiartbuddy@gmail.com.</p>
                </div>
                <div class="footer">
                    <p>&copy; 2024 AI ArtBuddy. All rights reserved.</p>
                </div>
            </div>
        </body>
    </html>
    """
    
    # Use provided body if available, otherwise use defaults
    body_text = body_text or default_body_text
    body_html = body_html or default_body_html
    
    # Prepare the email body
    body = {
        'Text': {
            'Data': body_text,
            'Charset': 'UTF-8'
        }
    }
    
    # Add HTML body if provided
    if body_html:
        body['Html'] = {
            'Data': body_html,
            'Charset': 'UTF-8'
        }
    
    try:
        response = ses_client.send_email(
            Source=sender,
            Destination={
                'ToAddresses': [recipient] if isinstance(recipient, str) else recipient
            },
            Message={
                'Subject': {
                    'Data': subject,
                    'Charset': 'UTF-8'
                },
                'Body': body
            }
        )
        return response
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def intro_page():
    inject_custom_css()
    st.title("Welcome to AI Art Buddy!🎨")

    
    st.write("""
        **AI Art Buddy** is a unique art tutoring application designed specifically for children ages 8-10.
        Our app offers personalized feedback and guidance to help young artists explore their creativity.
    """)

    st.subheader("Parental Consent Required")
    st.write("""
        In compliance with the **Children's Online Privacy Protection Act (COPPA)**, we require parental consent for children under 13 to use our services.
        Parents will receive a verification email during account setup to confirm consent for their child to use the app.
    """)

    st.subheader("Session Reports for Parents")
    st.write("""
        To keep you informed, **session summaries** will be emailed to the registered parent after each session, 
        including feedback provided to your child and any progress they make within the app.
    """)

    # Navigation buttons for Login or Register
    st.write("To get started, please proceed to create an account or log in.")
