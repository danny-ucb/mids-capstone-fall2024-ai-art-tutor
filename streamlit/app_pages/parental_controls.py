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
from helpers.multi_agent import * 
from typing import List, Dict
import chromadb

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

def list_all_memories(username: str = None) -> List[Dict]:
    """List all memories in the vector store, optionally filtered by username.
    
    Args:
        username (str, optional): If provided, only show memories for this user
        
    Returns:
        List[Dict]: List of dictionaries containing memory details
    """
    collection = get_vector_store()
    
    # Query parameters
    where_filter = {"username": username} if username else None
    
    # Get all memories
    results = collection.get(
        where=where_filter,
        include=['metadatas', 'documents', 'embeddings']
    )
    
    # Format results
    memories = []
    for i in range(len(results['ids'])):
        memories.append({
            'id': results['ids'][i],
            'content': results['documents'][i],
            'metadata': results['metadatas'][i]
        })
    
    return memories

def update_memory(memory_id: str, new_content: str) -> bool:
    """Update the content of a specific memory.
    
    Args:
        memory_id (str): The ID of the memory to update
        new_content (str): The new content for the memory
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        collection = get_vector_store()
        
        # Get existing metadata
        existing = collection.get(
            ids=[memory_id],
            include=['metadatas']
        )
        
        if not existing['ids']:
            return False
            
        # Update the memory
        collection.update(
            ids=[memory_id],
            documents=[new_content],
            metadatas=[existing['metadatas'][0]]
        )
        return True
    except Exception as e:
        print(f"Error updating memory: {str(e)}")
        return False

def delete_memory(memory_id: str) -> bool:
    """Delete a specific memory.
    
    Args:
        memory_id (str): The ID of the memory to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        collection = get_vector_store()
        collection.delete(ids=[memory_id])
        return True
    except Exception as e:
        print(f"Error deleting memory: {str(e)}")
        return False

def delete_user_memories(username: str) -> bool:
    """Delete all memories for a specific user.
    
    Args:
        username (str): The username whose memories should be deleted
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        collection = get_vector_store()
        collection.delete(where={"username": username})
        return True
    except Exception as e:
        print(f"Error deleting user memories: {str(e)}")
        return False

def search_memories(query: str, username: str = None, limit: int = 5) -> List[Dict]:
    """Search memories based on semantic similarity.
    
    Args:
        query (str): The search query
        username (str, optional): If provided, only search memories for this user
        limit (int): Maximum number of results to return
        
    Returns:
        List[Dict]: List of relevant memories
    """
    collection = get_vector_store()
    
    # Query parameters
    where_filter = {"username": username} if username else None
    
    results = collection.query(
        query_texts=[query],
        n_results=limit,
        where=where_filter
    )
    
    # Format results
    memories = []
    for i in range(len(results['ids'][0])):
        memories.append({
            'id': results['ids'][0][i],
            'content': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        })
    
    return memories

def parental_controls_page():
    st.title("Parental Controls Dashboard")

    tab1, tab2 = st.tabs(["Session Summaries", "Memory Management"])
    
    with tab1:
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
            
    with tab2:
        st.subheader("Memory Management")
        with st.expander("ℹ️ About Memory Management"):
            st.markdown("""
            ### Managing AI ArtBuddy's Memories
            
            - **Search**: Use the search bar to find specific memories by content
            - **View**: All memories are displayed in a list format
            - **Edit**: Click the edit button to modify a memory
            - **Delete**: Remove individual memories or use 'Delete All' to start fresh
            
            Memories help AI ArtBuddy provide more personalized assistance by remembering past interactions and preferences.
            """)

        # Search functionality
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("🔍 Search memories", placeholder="Enter keywords to search...")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("🗑️ Delete All Memories", type="secondary", help="This will delete all memories for this user"):
                if delete_user_memories(st.session_state["username"]):
                    st.success("All memories have been deleted successfully!")
                    st.rerun()
                else:
                    st.error("Failed to delete all memories")

        # Get memories based on search or show all
        if search_query:
            memories = search_memories(search_query, st.session_state["username"])
        else:
            memories = list_all_memories(st.session_state["username"])

        if memories:
            st.markdown(
                """
                <style>
                    .memory-list-item {
                        background-color: white;
                        padding: 1rem;
                        margin: 0.5rem 0;
                        border-radius: 8px;
                        border: 1px solid #e1e1e1;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    }
                    .memory-metadata {
                        color: #666;
                        font-size: 0.85em;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Initialize session state for edit mode if not exists
            if 'editing_memory' not in st.session_state:
                st.session_state.editing_memory = None

            for memory in memories:
                with st.container():
                    # Display memory as a list item
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(
                            f"""
                            <div class="memory-list-item">
                                <p>{memory['content']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        # Edit button to enter edit mode
                        if st.button("✍️ Edit", key=f"edit_{memory['id']}", 
                                   help="Edit this memory"):
                            st.session_state.editing_memory = memory['id']
                        
                        if st.button("🗑️ Delete", key=f"delete_{memory['id']}", 
                                   help="Delete this memory permanently"):
                            if delete_memory(memory['id']):
                                st.success("Memory deleted!")
                                st.rerun()
                            else:
                                st.error("Deletion failed")
                    
                    # Show edit form if this memory is being edited
                    if st.session_state.editing_memory == memory['id']:
                        with st.container():
                            st.markdown("### Edit Memory")
                            new_content = st.text_area(
                                "Edit memory content:",
                                value=memory['content'],
                                key=f"memory_{memory['id']}",
                                height=100
                            )
                            
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                if st.button("💾 Save", key=f"save_{memory['id']}", 
                                           help="Save changes"):
                                    if update_memory(memory['id'], new_content):
                                        st.success("Memory updated!")
                                        st.session_state.editing_memory = None
                                        st.rerun()
                                    else:
                                        st.error("Update failed")
                            with col2:
                                if st.button("❌ Cancel", key=f"cancel_{memory['id']}", 
                                           help="Cancel editing"):
                                    st.session_state.editing_memory = None
                                    st.rerun()

                    # Add a visual separator between memories
                    st.markdown("---")
        
        else:
            st.info("🤔 No memories found! Memories will be created as you interact with AI ArtBuddy.")

        # Help information


# def parental_controls_page():
#     st.title("Parental Controls Dashboard")

#     tab1, tab2 = st.tabs(["Session Summaries", "Memory Management"])
    
#     with tab1:
#         st.subheader("Session Summaries")
#         sessions = load_session_data(st.session_state["username"])
#         if sessions:
#             for session in sessions:
#                 st.markdown(
#                     f"""
#                     <div style="
#                         padding: 1rem; 
#                         margin-bottom: 1rem; 
#                         border-radius: 8px; 
#                         background-color: #f9f9f9;
#                         border: 1px solid #e1e1e1;
#                         box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
#                     ">
#                         <h4 style="margin: 0;">🗓️ Date: {session["date"]}</h4>
#                         <p style="margin: 0;"><strong>Session Start Time:</strong> {session["start_time"]}</p>
#                         <p style="margin: 0.5rem 0;"><strong>Duration:</strong> {session['duration_minutes']} minutes</p>
#                         <p style="margin: 0;"><strong>Number of Messages:</strong> {session['message_count']}</p>
#                         <p style="margin: 0;"><strong>Summary:</strong> {session['summary'].replace("Summary: ", "")}</p>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
#         else:
#             st.write("No saved session information yet!")
            
#     with tab2:
#         st.subheader("Memory Management")

#         st.write("### Debug Information")
#         try:
#             # Check if username exists in session state
#             if 'username' not in st.session_state:
#                 st.error("No username found in session state!")
#                 return
                
#             st.success(f"Current user: {st.session_state['username']}")
            
#             # Try to get vector store
#             collection = get_vector_store()
#             st.success("Successfully connected to vector store")
            
#             # Get all memories for debugging
#             memories = list_all_memories(st.session_state["username"])
#             st.write(f"Number of memories found: {len(memories)}")
            
#             if not memories:
#                 st.warning("No memories found for this user. This could be because:")
#                 st.write("1. No memories have been saved yet")
#                 st.write("2. The memory store connection isn't working")
#                 st.write("3. The username filter isn't working correctly")
                
#                 # Check the total number of memories in the collection
#                 total_results = collection.get()
#                 st.write(f"Total memories in collection: {len(total_results['ids'])}")
                
#                 if len(total_results['ids']) > 0:
#                     st.write("Sample metadata from collection:")
#                     st.json(total_results['metadatas'][:5])
            
#         except Exception as e:
#             st.error(f"Error accessing memories: {str(e)}")
#             return



        
#         # Help information
#         with st.expander("ℹ️ About Memory Management"):
#             st.markdown("""
#             ### Managing AI ArtBuddy's Memories
            
#             - **Search**: Use the search bar to find specific memories by content
#             - **View**: All memories are displayed in a list format
#             - **Edit**: Click the edit button to modify a memory
#             - **Delete**: Remove individual memories or use 'Delete All' to start fresh
            
#             Memories help AI ArtBuddy provide more personalized assistance by remembering past interactions and preferences.
#             """)
#         # Search functionality
#         col1, col2 = st.columns([3, 1])
#         with col1:
#             search_query = st.text_input("🔍 Search memories", placeholder="Enter keywords to search...")
#         with col2:
#             st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
#             if st.button("🗑️ Delete All Memories", type="secondary", help="This will delete all memories for this user"):
#                 if delete_user_memories(st.session_state["username"]):
#                     st.success("All memories have been deleted successfully!")
#                     st.rerun()
#                 else:
#                     st.error("Failed to delete all memories")

#         # Get memories based on search or show all
#         if search_query:
#             memories = search_memories(search_query, st.session_state["username"])
#         else:
#             memories = list_all_memories(st.session_state["username"])

#         if memories:
#             st.markdown(
#                 """
#                 <style>
#                     .memory-list-item {
#                         background-color: white;
#                         padding: 1rem;
#                         margin: 0.5rem 0;
#                         border-radius: 8px;
#                         border: 1px solid #e1e1e1;
#                         box-shadow: 0 2px 4px rgba(0,0,0,0.05);
#                     }
#                     .memory-metadata {
#                         color: #666;
#                         font-size: 0.85em;
#                     }
#                 </style>
#                 """,
#                 unsafe_allow_html=True
#             )

#             # Initialize session state for edit mode if not exists
#             if 'editing_memory' not in st.session_state:
#                 st.session_state.editing_memory = None

#             for memory in memories:
#                 with st.container():
#                     # Display memory as a list item
#                     col1, col2 = st.columns([4, 1])
                    
#                     with col1:
#                         st.markdown(
#                             f"""
#                             <div class="memory-list-item">
#                                 <p>{memory['content']}</p>
#                                 <p class="memory-metadata">Created: {memory['metadata'].get('date', 'N/A')}</p>
#                             </div>
#                             """,
#                             unsafe_allow_html=True
#                         )
                    
#                     with col2:
#                         # Edit button to enter edit mode
#                         if st.button("✍️ Edit", key=f"edit_{memory['id']}", 
#                                    help="Edit this memory"):
#                             st.session_state.editing_memory = memory['id']
                        
#                         if st.button("🗑️ Delete", key=f"delete_{memory['id']}", 
#                                    help="Delete this memory permanently"):
#                             if delete_memory(memory['id']):
#                                 st.success("Memory deleted!")
#                                 st.rerun()
#                             else:
#                                 st.error("Deletion failed")
                    
#                     # Show edit form if this memory is being edited
#                     if st.session_state.editing_memory == memory['id']:
#                         with st.container():
#                             st.markdown("### Edit Memory")
#                             new_content = st.text_area(
#                                 "Edit memory content:",
#                                 value=memory['content'],
#                                 key=f"memory_{memory['id']}",
#                                 height=100
#                             )
                            
#                             col1, col2 = st.columns([1, 4])
#                             with col1:
#                                 if st.button("💾 Save", key=f"save_{memory['id']}", 
#                                            help="Save changes"):
#                                     if update_memory(memory['id'], new_content):
#                                         st.success("Memory updated!")
#                                         st.session_state.editing_memory = None
#                                         st.rerun()
#                                     else:
#                                         st.error("Update failed")
#                             with col2:
#                                 if st.button("❌ Cancel", key=f"cancel_{memory['id']}", 
#                                            help="Cancel editing"):
#                                     st.session_state.editing_memory = None
#                                     st.rerun()

#                     # Add a visual separator between memories
#                     st.markdown("---")
        
#         else:
#             st.info("🤔 No memories found! Memories will be created as you interact with AI ArtBuddy.")



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
