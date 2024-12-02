#Streamlit_app.py
# Standard library imports
import os
import uuid
from datetime import datetime

# Third-party imports
import openai
import pytz
import streamlit as st
from PIL import Image as PilImage
from streamlit_drawable_canvas import st_canvas

# Database
import chromadb
from chromadb.utils import embedding_functions

# Local application imports
from helpers.api_keys import get_keys
from helpers.multi_agent import *
from helpers.general_helpers import *
from helpers.image_helpers import * 
from helpers.memory_utils import * 
from app_pages.parental_controls import *
from app_pages.login_registration_page import *
from app_pages.beta_feedback import * 

timezone = pytz.timezone("America/New_York")
     

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
        

if not st.session_state['logged_in']:
    intro_page()

    # Button to proceed to login/registration
    if st.button("Proceed to Login / Register"):
        st.session_state['show_login_registration'] = True

    # Check if the user chose to proceed to login/registration
    if 'show_login_registration' in st.session_state and st.session_state['show_login_registration']:
        # Display login and registration tabs
        login, registration = st.tabs(["Login", "Register"])
        with login:
            login_page()
        with registration:
            registration_page()

else:
    # Main application for logged-in users
    openai_key = get_keys()
    inject_custom_css()

    # Initialize hide_safety_message in session state if it doesn't exist
    if 'hide_safety_message' not in st.session_state:
        st.session_state['hide_safety_message'] = False

    
    main_page, parental_controls, beta_feedback = st.tabs(["Home", "Parental Controls", "Beta Testing Feedback"])
    
    if 'graph' not in st.session_state:
        st.session_state['graph'] = create_nodes(openai_key)

    if 'thread_counter' not in st.session_state:
        st.session_state['thread_counter'] = 0


            # Load current consent settings
    consent_data = load_user_consent(st.session_state["username"])
    
#     # Initialize consent settings in session state if not exists
    if 'consent_settings' not in st.session_state:
        st.session_state.consent_settings = {
            "memory_collection": consent_data["memory_collection"],
            "image_collection": consent_data["image_collection"],
            "session_summaries": consent_data["session_summaries"],
            "email_updates": consent_data["email_updates"]
        }
        
    with main_page:

        # App title and description
        st.title("🎨 AIArtBuddy")
        # st.write(st.session_state)
        # st.write(st.session_state["consent_settings"]["memory_collection"])
        # Safety message with hide button
        if not st.session_state['hide_safety_message']:
            col1, col2 = st.columns([20, 1])
            with col1:
                st.markdown("""
                <div style="
                    padding: 20px;
                    border-radius: 10px;
                    background-color: #E8F5E9;
                    margin-bottom: 20px;
                    border: 2px solid #81C784;
                    font-family: 'Comic Sans MS', cursive, sans-serif;
                ">
                    <p style="
                        color: #2E7D32;
                        margin: 0;
                        font-size: 1.1em;
                    ">
                        👋 Hi! We want to make sure you're having fun and staying safe. To help with that, your parent or guardian will be able to see the art you create and our conversations, so they know what you're doing and can help if you need it!
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("✕", help="Hide message"):
                    st.session_state['hide_safety_message'] = True
                    # persist_username()
                    st.rerun()
        
        st.subheader(f"Let's Draw {st.session_state["username"]}!")
        st.write("Let's make something amazing together! I can help you:")
        st.write("""
        - 📝 Create fun stories and turn them into art
        - 🖼️ Give friendly tips about your drawings
        - 🌈 Learn cool new art techniques
        - 👩‍🎨 Explore the wonderful world of art
        - 🎨 Generate example images to inspire your creativity
        - ✏️ Practice drawing alongside AI-generated examples
        
        Just tell me what sounds fun to you! You can ask me to generate example images to help you practice, or show me your own drawings for friendly feedback. Remember, every great artist started just like you - with imagination and curiosity! 🌟
        
        Try saying things like:
        - "Can you help me improve my drawing?"
        - "Generate an example of a magical forest I can practice drawing"
        - "Can you show me how to draw a friendly dragon?"
        - "What are the steps to draw a cartoon animal?"
        """)
        # Add a visual separator here
        st.markdown("---")  # This creates a horizontal line
        st.markdown("<br>", unsafe_allow_html=True)  # This adds some spacing
        # Initialize session state variables for chat
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        if 'relevant_messages' not in st.session_state:
            st.session_state['relevant_messages'] = []
        if 'current_image' not in st.session_state:
            st.session_state['current_image'] = None
        if 'chat_active' not in st.session_state:
            st.session_state['chat_active'] = False
        
        # File uploader for image input
        uploaded_image = st.file_uploader("Upload your artwork (optional):", type=["png", "jpg", "jpeg"])
                
        if uploaded_image is not None:
            # Check if this is a new upload
            if 'last_uploaded_image' not in st.session_state or st.session_state['last_uploaded_image'] != uploaded_image.name:
                save_path = save_uploaded_image(uploaded_image)
                if save_path:
                    st.session_state['current_image'] = save_path
                    st.session_state['last_uploaded_image'] = uploaded_image.name  # Store the filename to track uploads
                    st.image(uploaded_image, caption="Your uploaded artwork", width=200)
            else:
                # If it's the same image, just display it
                st.image(uploaded_image, caption="Your uploaded artwork", width=200)       
                
        # Start Chat button
        if not st.session_state['chat_active']:
            if st.button("💬 Start Chat!"):
                st.session_state['chat_active'] = True
                st.experimental_rerun()             
                
        # Chat interface
        if st.session_state['chat_active']:
            # Create containers for different parts of the chat interface
            chat_history = st.container()
            input_container = st.container()
          
            # Display chat history in the first container
            with chat_history:
                for msg in st.session_state['messages']:
                    if isinstance(msg, dict):
                        role = msg.get('role', '')
                        content = msg.get('content', '')
                        # Display the message
                        if role == 'user':
                            st.write(f'👤 **You:** {content}')
                        elif role == 'assistant':    
                            if content.startswith("https"):
                                img_file_path = f"produced_images/AI_generated_image_{generate_random_string(10)}.png"
                                download_image_requests(url=content, file_name=img_file_path)
                                st.write(f'🎨 **AIArtBuddy:** Here\'s what I drew for you:')
                                st.image(img_file_path, width = 300) 
                            else:
                                if len(content) > 0: 
                                    st.write(f'🎨 **AIArtBuddy:** {content}')

            
            # Create a spacer
            st.markdown("<br>", unsafe_allow_html=True)

            # Input area in the second container
            with input_container:

                # Initialize session state variables
                if "submit_pressed" not in st.session_state:
                    st.session_state.submit_pressed = False
                if 'temp_input' not in st.session_state:
                    st.session_state.temp_input = ""
                if 'input_key' not in st.session_state:
                    st.session_state.input_key = 0
                    
                # Initialize the current dynamic input key in session state
                current_key = f"user_input_{st.session_state.input_key}"
                if current_key not in st.session_state:
                    st.session_state[current_key] = ""

                def handle_submit():
                    if st.session_state[current_key]:
                        # persist_username()
                        st.session_state.submit_pressed = True
                        st.session_state.temp_input = st.session_state[current_key]
                        
                user_input = st.text_input(
                    "Type your message here:",
                    key=current_key, 
                    on_change = handle_submit
                )
               
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    clear_pressed = st.button("Clear Chat")
                    
                with col2:
                    end_session_pressed = st.button("End Session")


                # Download button
                with col3:
                    if st.session_state['messages']:
                        chat_text = "\n".join([
                            f"{'You' if msg.get('role') == 'user' else 'AIArtBuddy'}: {msg.get('content', '')}"
                            for msg in st.session_state['messages']
                            if isinstance(msg, dict)
                        ])
                        st.download_button(
                            label="Download Chat History",
                            data=chat_text,
                            file_name="art_buddy_chat.txt",
                            mime="text/plain"
                        )

                if (st.session_state.submit_pressed and st.session_state.temp_input):                    
                    try:
                        current_input = st.session_state.temp_input
                        
                        # Reset states
                        st.session_state.submit_pressed = False
                        st.session_state.temp_input = ""
                        st.session_state.input_key += 1
                
                        # Add user message to history
                        existing_messages = st.session_state["messages"]
                        is_duplicate = any(
                            msg.get('content') == current_input and 
                            msg.get('role') == 'user'
                            for msg in existing_messages[-3:]                    
                        )
                        
                        if not is_duplicate:      
                            st.session_state['messages'].append({
                                "role": "user",
                                "content": current_input, 
                                "timestamp": datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
                            })
                
                            # Get AI response
                            thread_config = {
                                "configurable": {
                                    "username": st.session_state['username'], 
                                    "thread_id": "1", 
                                    "consent_settings": st.session_state.consent_settings
                                }
                            }
                            response = None
                        
                            if st.session_state['current_image'] and (
                                'last_image_used' not in st.session_state or 
                                st.session_state['last_image_used'] != st.session_state['current_image']
                            ):
    
                                response = stream_messages(
                                    st.session_state['graph'],
                                    text=current_input,
                                    thread=thread_config,
                                    image_path=st.session_state['current_image']
                                )
                                st.session_state["relevant_messages"] = response
                                # Mark this image as used
                                st.session_state['last_image_used'] = st.session_state['current_image']
                            
                            else:
        
                                response = stream_messages(
                                    st.session_state['graph'],
                                    text=current_input,
                                    thread=thread_config
                                )
                                st.session_state["relevant_messages"] = response
                                
                
                            # Add all messages to history
                            if response:
                                for msg in response:
                                    is_duplicate_response = any(
                                        existing_msg.get('content') == msg['content'] and 
                                        existing_msg.get('role') == msg['role']
                                        for existing_msg in st.session_state['messages'][-3:]
                                    )
                                    
                                    if not is_duplicate_response:
                                        st.session_state['messages'].append({
                                            "role": msg["role"],
                                            "content": msg["content"],
                                            "timestamp": datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
                                        })
                
                            # Prepare for next input
                            st.session_state['thread_counter'] += 1
                            next_key = f"user_input_{st.session_state.input_key}"
                            if next_key not in st.session_state:
                                st.session_state[next_key] = ""
                                
                            st.experimental_rerun()
            

                    except Exception as e:
                        error_msg = str(e)
                        if "array too long" in error_msg.lower() or "context length" in error_msg.lower():
                            st.error("You've reached the conversation limit for our beta testing. Please click 'Clear Chat' or 'End Session' to restart the conversation.")
                        else:
                            st.error(f"An error occurred: {error_msg}")
                        st.error(f"An error occurred: {str(e)}")                        

                
                if clear_pressed:
                    st.session_state['messages'] = []
                    st.session_state['graph'] = create_nodes(openai_key)
                    st.experimental_rerun()
                    # persist_username()


                if end_session_pressed:
                    save_session_summary()
                    cleanup_temp_files()  
                    st.session_state['messages'] = []
                    st.session_state['graph'] = create_nodes(openai_key)
                    st.experimental_rerun()
                    persist_username()
                    

    with parental_controls:
        parental_controls_page()

    with beta_feedback:
        beta_feedback_page()