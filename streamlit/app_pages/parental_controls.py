# parental_controls.py
# Standard library imports
import os
import io
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
from helpers.consent_utils import * 

# Compression
import zipfile
        
def parental_controls_page():
    st.title("Parental Controls Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(["Session Summaries", "Memory Management", "Uploaded Images", "Consent Management"])
    
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

    with tab3:
        st.subheader("Uploaded Images History")
        
        # Help information
        with st.expander("ℹ️ About Image History"):
            st.markdown("""
            ### Managing Uploaded Images
            
            This section shows all images that have been uploaded during art sessions:
            - Images are organized by date
            - Each image includes upload time and file details
            - Images can be downloaded individually or as a group
            - Images can be previewed directly in the dashboard
            
            All images are stored securely in user-specific folders.
            """)
        
        # Get list of uploaded images
        image_files = get_all_user_images()
        
        if image_files:
            # Group images by date folder
            from itertools import groupby
            sorted_images = sorted(image_files, key=lambda x: x['date_folder'], reverse=True)
            grouped_images = groupby(sorted_images, key=lambda x: x['date_folder'])
            
            # Add download all button
            st.download_button(
                label="📥 Download All Images",
                data=create_image_archive(image_files),
                file_name="all_artwork.zip",
                mime="application/zip",
                help="Download all uploaded images as a ZIP file"
            )
            
            # Display images grouped by date
            for date, group in grouped_images:
                st.markdown(f"### 📅 {date}")
                
                # Convert group to list since groupby iterator can only be used once
                group_images = list(group)
                
                # Create columns for images
                cols = st.columns(3)
                for idx, image_info in enumerate(group_images):
                    with cols[idx % 3]:
                        try:
                            # Create container for each image
                            with st.container():
                                # Display image
                                img = PilImage.open(image_info['filepath'])
                                st.image(img, caption=image_info['filename'], use_column_width=True)
                                
                                # Display metadata
                                st.markdown(f"""
                                <div style='font-size: 0.9em; color: #666;'>
                                    🕒 {datetime.fromtimestamp(os.path.getctime(image_info['filepath'])).strftime('%H:%M:%S')}
                                    <br>
                                    📁 {format_file_size(image_info['size'])}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Download and delete buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    with open(image_info['filepath'], 'rb') as file:
                                        st.download_button(
                                            label="📥",
                                            data=file,
                                            file_name=image_info['filename'],
                                            mime=f"image/{image_info['filename'].split('.')[-1].lower()}",
                                            help="Download this image"
                                        )
                                with col2:
                                    if st.button("🗑️", key=f"delete_{image_info['filename']}", 
                                               help="Delete this image"):
                                        try:
                                            os.remove(image_info['filepath'])
                                            # Remove empty directories
                                            parent_dir = os.path.dirname(image_info['filepath'])
                                            if not os.listdir(parent_dir):
                                                os.rmdir(parent_dir)
                                            st.success("Image deleted!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting image: {str(e)}")
                                
                                # Add spacing between images
                                st.markdown("<br>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error loading image {image_info['filename']}: {str(e)}")
                
                # Add separator between dates
                st.markdown("---")
        else:
            st.info("No images have been uploaded yet!")

    with tab4:
        st.subheader("Consent Management")
        
        with st.expander("ℹ️ About Consent Management"):
            st.markdown("""
            ### Managing Your Child's Data and Privacy
            
            Control how AI ArtBuddy stores and uses information:
            
            - **Memory Collection**: Allows AI ArtBuddy to remember your child's art preferences, interests, and previous 
              discussions to provide more personalized guidance and suggestions.
              
            - **Image Collection**: Enables saving of your child's uploaded artwork and AI-generated images, allowing them 
              to revisit their work in future sessions.
              
            - **Session Summaries**: Saves records of art sessions, including conversations and learning progress, 
              which you can review in the parent dashboard.
              
            - **Email Updates**: Receive session summaries by email, including what they learned and created, 
              helping you stay involved in their artistic journey.
            
            You can modify these settings at any time or request complete data deletion.
            """)
        
        # Load current consent settings
        consent_data = load_user_consent(st.session_state["username"])
        
        st.markdown("### Current Consent Settings")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Memory Collection Consent
            memory_collection = st.checkbox(
                "Memory Collection",
                value=consent_data["memory_collection"],
                help="Allow AI ArtBuddy to remember your child's art interests and previous conversations"
            )
            
            # Image Collection Consent
            image_collection = st.checkbox(
                "Image Collection",
                value=consent_data["image_collection"],
                help="Save your child's artwork and AI-generated images for future sessions"
            )
            
            # Session Summaries Consent
            session_summaries = st.checkbox(
                "Session Summaries",
                value=consent_data["session_summaries"],
                help="Save session records for review in the parent dashboard"
            )
            
            # Email Updates Consent
            email_updates = st.checkbox(
                "Email Updates",
                value=consent_data["email_updates"],
                help="Receive email summaries about your child's art sessions"
            )

        # Save changes button
        if st.button("Save Consent Changes"):
            new_consent_data = {
                "memory_collection": memory_collection,
                "image_collection": image_collection,
                "session_summaries": session_summaries,
                "email_updates": email_updates,
                "consent_date": consent_data.get("consent_date") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            save_user_consent(st.session_state["username"], new_consent_data)
            st.success("Consent settings updated successfully!")
            
            # Show what changes mean
            st.info("""
            🔄 Your changes have been saved. Here's what this means:
            
            {}
            {}
            {}
            {}
            """.format(
                "• AI ArtBuddy will remember past conversations and preferences" if memory_collection else "• Past conversations won't be saved between sessions",
                "• Artwork will be saved for future sessions" if image_collection else "• Artwork will only be available during the current session",
                "• Session records will be saved in your dashboard" if session_summaries else "• Session records won't be kept",
                "• You'll receive session summaries by email" if email_updates else "• You won't receive session summaries by email"
            ))




