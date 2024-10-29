import openai
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image as PilImage
import os
import bcrypt
import json
import random
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from helpers import * 
from datetime import datetime, timedelta
from datetime import datetime
import pandas as pd
import pytz
import boto3

def transform_messages_to_conversation(messages):
    conversation = []
    for message in messages:
        # Map "role" to "sender" and "content" remains the same
        sender = "AIArtBuddy" if message["role"] == "assistant" else "user"
        conversation.append({
            "sender": sender,
            "content": message["content"]
        })
    return conversation

def summarize_conversation(formatted_conversation):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt_template = """Summarize the following conversation between the user and AI ArtBuddy. Use the following summary report format:
    
    Conversation:
    {conversation}
    
    Summary:

    Also, if the user mentions anything inappropriate, quote that within the summary using the user's words. Otherwise, leave the summary as is.

    """
    prompt = PromptTemplate(
        input_variables=["conversation"],
        template=prompt_template,
    )
    
    # Create the chain with the prompt and language model
    summarization_chain = LLMChain(llm=llm, prompt=prompt)
    summary = summarization_chain.run(conversation=formatted_conversation)

    # Extract the summary part from the full output
    summary_start = summary.find("Summary:")
    if summary_start != -1:
        summary = summary[summary_start + len("Summary:"):].strip()
    else:
        summary = summary.strip()
    return summary


def record_session_data(user_message, assistant_response):
    if "session_data" not in st.session_state:
        st.session_state["session_data"] = []
        timezone = pytz.timezone("America/New_York")

    # Each entry contains a timestamp, user message, assistant response, and duration placeholder
    session_entry = {
        "timestamp": datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S"),
        "user_message": user_message,
        "assistant_response": assistant_response,
        "duration": None  # Update duration if tracking is implemented
    }
    st.session_state["session_data"].append(session_entry)


def get_user_session_file(username):
    """Get the path to the user-specific session data file"""
    if not os.path.exists('user_sessions'):
        os.makedirs('user_sessions')
    return f'user_sessions/session_data_{username}.json'


def load_session_data(username):
    """Load existing session data from a JSON file for a specific user"""
    file_path = get_user_session_file(username)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return []

def save_session_data(username, session_data):
    """Save session data to a JSON file for a specific user"""
    file_path = get_user_session_file(username)
    with open(file_path, 'w') as f:
        json.dump(session_data, f)


def save_session_summary():
    """Save the current session summary for a specific user"""
    
    session_data = load_session_data(st.session_state["username"])
    messages = st.session_state["messages"]

    # Calculate session duration
    if st.session_state["messages"]:
        start_time = datetime.strptime(messages[0].get('timestamp', ''), "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(messages[-1].get('timestamp', ''), "%Y-%m-%d %H:%M:%S")
        duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes
    else:
        duration = 0

    session_summary = {
        'date': datetime.now().strftime("%Y-%m-%d"), 
        'start_time': start_time.strftime("%H:%M:%S"), 
        'duration_minutes': round(duration), 
        'message_count': len(st.session_state["messages"]), 
        'summary': summarize_conversation(transform_messages_to_conversation(st.session_state["messages"]))
    }

    session_data.append(session_summary)
    save_session_data(st.session_state['username'], session_data)
    send_session_summary_email(sender = "aiartbuddy@gmail.com" ,
                               recipient = st.session_state['parent_email'],
                                subject = f"AI ArtBuddy Session Report for {st.session_state['username']}", 
                                session_details = session_summary)
    st.success(f"Session email sent to {st.session_state['parent_email']}")


def parental_controls_page():
    st.title("Parental Controls Dashboard")
    st.subheader("Session Summaries")

    sessions = load_session_data(st.session_state["username"])
    if sessions:
        for session in sessions:
            
            st.markdown(
                f"""
                <div style="
                    padding: 1rem; 
                    margin-bottom: 1rem; 
                    border-radius: 8px; 
                    background-color: #f9f9f9;
                    border: 1px solid #e1e1e1;
                    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
                ">
                    <h4 style="margin: 0;">🗓️ Date: {session["date"]}</h4>
                    <p style="margin: 0;"><strong>Session Start Time:</strong> {session["start_time"]}</p>
                    <p style="margin: 0.5rem 0;"><strong>Duration:</strong> {session['duration_minutes']} minutes</p>
                    <p style="margin: 0;"><strong>Number of Messages:</strong> {session['message_count']}</p>
                    <p style="margin: 0;"><strong>Summary:</strong> {session['summary'].replace("Summary: ", "")}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.write("No saved session information yet!")

def send_session_summary_email(sender, recipient, subject, session_details, body_text=None, body_html=None):
    """
    Send an email using AWS SES with session details for AI Art Buddy.
    
    Parameters:
    sender (str): Email address of the sender
    recipient (str or list): Email address(es) of recipient(s)
    subject (str): Subject line of the email
    session_details (dict): Dictionary containing session details such as timestamp, duration, summary, and message count
    body_text (str): Optional plain text version of the email
    body_html (str): Optional HTML version of the email
    
    Returns:
    dict: Response from AWS SES
    """
    # Specify your AWS region (example: 'us-east-1')
    region = 'us-east-1'  # Change to your region
    
    # Create a new SES resource with the specified region
    ses_client = boto3.client('ses', region_name=region)
    
    # Default email bodies with session details
    default_body_text = (
        f"Session Summary for AI Art Buddy:\n\n"
        f"Date: {session_details['date']}\n"
        f"Duration: {session_details['duration_minutes']} minutes\n"
        f"Summary: {session_details['summary']}\n\n"
        f"For more details, please log in to your AI Art Buddy account.\n\n"
        f"Thank you for using AI Art Buddy!"
    )
    
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
                .session-info {{
                    font-size: 16px;
                    color: #333333;
                    margin-top: 10px;
                }}
                .footer {{
                    margin-top: 30px;
                    font-size: 12px;
                    color: #999999;
                    text-align: center;
                }}
                .login-suggestion {{
                    margin-top: 20px;
                    color: #3b5998;
                    font-size: 14px;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Session Summary for AI Art Buddy</h1>
                </div>
                <div class="content">
                    <p class="session-info"><strong>Date:</strong> {session_details['date']}</p>
                    <p class="session-info"><strong>Duration:</strong> {session_details['duration_minutes']} minutes</p>
                    <p class="session-info"><strong>Summary:</strong> {session_details['summary']}</p>
                    <p class="login-suggestion">For more details, please log in to your AI Art Buddy account.</p>
                    <p>Thank you for using AI Art Buddy!</p>
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
