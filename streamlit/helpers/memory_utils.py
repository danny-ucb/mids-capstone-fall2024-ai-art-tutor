## memory_utils.py
# Standard library imports
import os
import io
import json
import uuid
import random
import shutil
from datetime import datetime, timedelta
from typing import List, Dict

# Third-party imports
import boto3
import bcrypt
import pytz
import pandas as pd
import tiktoken
import streamlit as st
from PIL import Image as PilImage
from streamlit_drawable_canvas import st_canvas

# Database
import chromadb
from chromadb.utils import embedding_functions

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Local imports
from helpers.api_keys import * 
from helpers.general_helpers import *
from helpers.image_helpers import * 
from helpers.consent_utils import * 

# Compression
import zipfile

def get_vector_store():
    """Get or create vector store instance"""
    persist_directory = '/home/ubuntu/workspace/mids-capstone-fall2024-ai-art-tutor/streamlit'
    
    try:
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize the embedding function
        emb_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-ada-002"
        )
        
        # Initialize the persistent client
        client = chromadb.PersistentClient(path=persist_directory)
        
        try:
            # Try to get existing collection
            collection = client.get_collection(
                name="recall_vector_store",
                embedding_function=emb_fn
            )
            print("Successfully connected to existing collection")
            
        except Exception as e:
            # If collection doesn't exist, create new collection
            print(f"Creating new collection due to: {str(e)}")
            collection = client.create_collection(
                name="recall_vector_store",
                embedding_function=emb_fn
            )
            print("Successfully created new collection")
        
        return collection
    
    except Exception as e:
        print(f"Critical error getting vector store: {str(e)}")
        raise

def get_memory_clusters(memories: List[Dict], similarity_threshold: float = 0.85) -> List[List[Dict]]:
    """Group related memories together based on embedding similarity."""
    collection = get_vector_store()
    clusters = []
    used_memories = set()

    for i, memory in enumerate(memories):
        if memory['id'] in used_memories:
            continue

        # Create a cluster starting with this memory
        cluster = [memory]
        used_memories.add(memory['id'])

        # Find similar memories using the vector store
        similar_results = collection.query(
            query_texts=[memory['content']],
            n_results=len(memories),  # Get all potential matches
            where={"username": memory['metadata']['username']}
        )

        # Add similar memories to cluster
        for j, (similar_id, distance) in enumerate(zip(similar_results['ids'][0], similar_results['distances'][0])):
            if similar_id not in used_memories and distance <= similarity_threshold:
                similar_memory = next((m for m in memories if m['id'] == similar_id), None)
                if similar_memory:
                    cluster.append(similar_memory)
                    used_memories.add(similar_id)

        clusters.append(cluster)

    return clusters

def summarize_memory_cluster(cluster: List[Dict]) -> str:
    """Summarize a cluster of related memories."""
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # Using 3.5 to save tokens
    
    prompt_template = """
    Below are several related memories from interactions with a user. 
    Please create a concise summary that preserves the key information and patterns while reducing length.
    Focus on recurring themes, preferences, and important details.

    Memories:
    {memories}

    Summary (be concise):
    """
    
    prompt = PromptTemplate(
        input_variables=["memories"],
        template=prompt_template,
    )
    
    # Prepare memories text
    memories_text = "\n---\n".join([f"Memory {i+1}: {m['content']}" 
                                   for i, m in enumerate(cluster)])
    
    # Create and run the chain
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(memories=memories_text)
    
    return summary.strip()

def consolidate_memories(username: str, max_token_target: int = 6000) -> None:
    """Consolidate memories when they exceed token limit."""
    collection = get_vector_store()
    
    # Get all memories for user
    memories = list_all_memories(username)
    
    # Calculate total tokens
    total_tokens = sum(count_tokens(m['content']) for m in memories)
    
    if total_tokens > max_token_target:
        # Group related memories
        clusters = get_memory_clusters(memories)
        
        # Process each cluster
        for cluster in clusters:
            if len(cluster) > 1:  # Only summarize clusters with multiple memories
                # Create summary
                summary = summarize_memory_cluster(cluster)
                
                # Delete old memories
                old_ids = [m['id'] for m in cluster]
                collection.delete(ids=old_ids)
                
                # Save consolidated memory
                collection.add(
                    documents=[summary],
                    metadatas=[{
                        "username": username,
                        "type": "consolidated",
                        "original_count": len(cluster),
                        "consolidated_date": datetime.now().isoformat(),
                        "oldest_memory_date": min(m['metadata'].get('date', datetime.now().isoformat()) 
                                                for m in cluster)
                    }],
                    ids=[f"{username}_consolidated_{str(uuid.uuid4())}"]
                )


def save_session_summary():
    """Save session summary based on consent preferences."""
    messages = st.session_state["messages"]
    
    if not messages:
        return
        
    # Calculate session details
    start_time = datetime.strptime(messages[0].get('timestamp', ''), "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(messages[-1].get('timestamp', ''), "%Y-%m-%d %H:%M:%S")
    duration = (end_time - start_time).total_seconds() / 60
    
    # Create session summary
    session_summary = {
        'date': datetime.now().strftime("%Y-%m-%d"), 
        'start_time': start_time.strftime("%H:%M:%S"), 
        'duration_minutes': round(duration), 
        'message_count': len(messages), 
        'summary': summarize_conversation(transform_messages_to_conversation(messages))
    }
    
    # Save session data if consent is given
    if st.session_state.consent_settings["session_summaries"]:
        session_data = load_session_data(st.session_state["username"])
        session_data.append(session_summary)
        save_session_data(st.session_state['username'], session_data)
    
    # Send email if email updates consent is given
    if st.session_state.consent_settings["email_updates"]:
        send_session_summary_email(
            sender="aiartbuddy@gmail.com",
            recipient=st.session_state['parent_email'],
            subject=f"AI ArtBuddy Session Report for {st.session_state['username']}", 
            session_details=session_summary
        )
        st.success(f"Session summary sent to {st.session_state['parent_email']}")
    
    # Clear messages after processing
    st.session_state['messages'] = []
    
        
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


def count_tokens(text: str, model: str = "gpt-4-0125-preview") -> int:
    """Count the number of tokens in a text string."""
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))

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
