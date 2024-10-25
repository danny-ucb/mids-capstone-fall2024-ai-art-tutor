import openai
import streamlit as st
import os
import requests
from PIL import Image
import io
import streamlit as st
from io import BytesIO
import base64
import boto3
from botocore.exceptions import ClientError

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
                    <p>Please enter this code to complete your sign-up process.</p>
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