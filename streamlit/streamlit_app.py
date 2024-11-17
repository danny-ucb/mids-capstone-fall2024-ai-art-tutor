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
    
    # page = st.sidebar.selectbox("Select a page", ["Home", "Parental Controls"])
    main_page, parental_controls = st.tabs(["Home", "Parental Controls"])
    
    if 'graph' not in st.session_state:
        st.session_state['graph'] = create_nodes(openai_key)

            # Load current consent settings
    consent_data = load_user_consent(st.session_state["username"])
    
    # Initialize consent settings in session state if not exists
    if 'consent_settings' not in st.session_state:
        st.session_state.consent_settings = {
            "memory_collection": consent_data["memory_collection"],
            "image_collection": consent_data["image_collection"],
            "session_summaries": consent_data["session_summaries"],
            "email_updates": consent_data["email_updates"]
        }
        
    with main_page:

        # App title and description
        st.title("🎨 AI ArtBuddy")
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
                    st.rerun()
        
        st.subheader(f"Let's Draw {st.session_state["username"]}!")
        st.write("Let's make something amazing together! I can help you:")
        st.write("""
        - 📝 Create fun stories and turn them into art
        - 🖼️ Give friendly tips about your drawings
        - 🌈 Learn cool new art techniques
        - 👩‍🎨 Explore the wonderful world of art
        
        Just tell me what sounds fun to you! Remember, every great artist started just like you - with imagination and curiosity! 🌟
        """)

        
        # Initialize session state variables for chat
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        
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
                                st.write(f'🎨 **ArtBuddy:** Here\'s what I drew for you:')
                                st.image(img_file_path, width = 200) 
                            else:
                                st.write(f'🎨 **ArtBuddy:** {content}')

            
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
                    if st.session_state[current_key]:  # Now we can safely check this
                        st.session_state.submit_pressed = True
                        st.session_state.temp_input = st.session_state[current_key]
                        
                user_input = st.text_input(
                    "Type your message here:",
                    key=current_key,
                    on_change=handle_submit
                )
               
                col1, col2, col3 = st.columns([1, 1, 1])
                col = st.columns([1]) 
                # Clear chat button
                with col1:
                    clear_pressed = st.button("Clear Chat")
                    
                with col2:
                    end_session_pressed = st.button("End Session")

                # Download button
                with col3:
                    if st.session_state['messages']:
                        chat_text = "\n".join([
                            f"{'You' if msg.get('role') == 'user' else 'ArtBuddy'}: {msg.get('content', '')}"
                            for msg in st.session_state['messages']
                            if isinstance(msg, dict)
                        ])
                        st.download_button(
                            label="Download Chat History",
                            data=chat_text,
                            file_name="art_buddy_chat.txt",
                            mime="text/plain"
                        )
            
                # Process input when either Enter is pressed or Send button is clicked
                if st.session_state.submit_pressed and st.session_state.temp_input:
                    current_input = st.session_state.temp_input
                    
                    # Reset states
                    st.session_state.submit_pressed = False
                    st.session_state.temp_input = ""
                    st.session_state.input_key += 1

                    
                    # Add user message to history
                    st.session_state['messages'].append({
                        "role": "user",
                        "content": current_input, 
                        "timestamp": datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
                    })


                    # Get AI response
                    thread_config = {"configurable": {"username": st.session_state['username'], 
                                                      "thread_id": "1", 
                                                     "consent_settings": st.session_state.consent_settings}}
                    
                    # try:
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
                        # Mark this image as used
                        st.session_state['last_image_used'] = st.session_state['current_image']
                    
                    else:

                        response = stream_messages(
                            st.session_state['graph'],
                            text=current_input,
                            thread=thread_config
                        )
                    
                    # Extract the message content
                    if response:
                        content = None
                        if isinstance(response, dict):
                            # Check for moderator response first
                            if 'moderator' in response:
                                content = response['moderator'].get('moderator_response', '')
                            # Check for messages from different agents
                            elif 'messages' in response:
                                messages = response['messages']
                                if messages and len(messages) > 0:
                                    content = messages[0].content
                            # Check for specific agent responses
                            else:
                                for node_key in ['visual_artist', 'critic', 'storyteller', 'silly']:
                                    if node_key in response and 'messages' in response[node_key]:
                                        messages = response[node_key]['messages']
                                        if messages and len(messages) > 0:
                                            content = messages[0].content
                                            break
                        
                        if content:
                            st.session_state['messages'].append({
                                "role": "assistant",
                                "content": content, 
                                "timestamp": datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                    # Initialize next key before rerun
                    next_key = f"user_input_{st.session_state.input_key}"
                    if next_key not in st.session_state:
                        st.session_state[next_key] = ""
                    st.experimental_rerun()
                        
                    # except Exception as e:
                    #     st.error(f"An error occurred: {str(e)}")
                
                if clear_pressed:
                    st.session_state['messages'] = []
                    st.experimental_rerun()

                if end_session_pressed:
                    save_session_summary()
                    cleanup_temp_files()  
                    st.session_state['messages'] = []
                    st.experimental_rerun()
                    

    with parental_controls:
        
        # Call the function to display the parental controls page
        parental_controls_page()
