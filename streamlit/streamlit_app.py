import openai
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image as PilImage
import os
from helpers.api_keys import get_keys
from helpers.multi_agent import *
from app_pages.helpers import inject_custom_css
from app_pages.parental_controls import * 
from app_pages.login_registration_page import * 
from datetime import datetime
import pytz
import chromadb
from chromadb.utils import embedding_functions
import uuid
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
    
    # page = st.sidebar.selectbox("Select a page", ["Home", "Parental Controls"])
    main_page, parental_controls = st.tabs(["Home", "Parental Controls"])
    
    if 'graph' not in st.session_state:
        st.session_state['graph'] = create_nodes(openai_key)
    with main_page:

        # App title and description
        st.title("🎨 AI ArtBuddy")
        st.subheader("Let's Draw!")
        st.write("Ask me anything about art and drawing! I'm here to help you learn and have fun. 😊")
        
        # Initialize session state variables for chat
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        if 'current_image' not in st.session_state:
            st.session_state['current_image'] = None
        if 'chat_active' not in st.session_state:
            st.session_state['chat_active'] = False
        
        # File uploader for image input
        uploaded_image = st.file_uploader("Upload your artwork (optional):", type=["png", "jpg", "jpeg"])
        
        # Handle image upload
        if uploaded_image is not None:
            # Save image and update session state
            save_dir = "uploaded_images"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            save_path = os.path.join(save_dir, uploaded_image.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            st.session_state['current_image'] = save_path
            st.image(uploaded_image, caption="Your uploaded artwork", width=200)
            # image_moderator_result = image_moderation(openai_key, image_path=save_path)
        
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
                                st.image(img_file_path, width = 300) 
                            else:
                                st.write(f'🎨 **ArtBuddy:** {content}')

            
            # Create a spacer
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Input area in the second container
            with input_container:
                # Chat input
                user_input = st.text_input("Type your message here:", key="user_input", value = "")
                
                # Buttons in columns
                
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                col = st.columns([1]) 
                # Send button
                with col1:
                    send_pressed = st.button("Send", key="send_button")
                
                # Clear chat button
                with col2:
                    clear_pressed = st.button("Clear Chat")
                    
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

                with col4:
                    end_session_pressed = st.button("End Session")

                
                if send_pressed and user_input:
                    
                    # Add user message to history
                    st.session_state['messages'].append({
                        "role": "user",
                        "content": user_input, 
                        "timestamp": datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")
                    })

                    # Get AI response
                    thread_config = {"configurable": {"username": st.session_state['username'], "thread_id": "1"}}
                    
                    try:
                        response = None
                        if st.session_state['current_image']:
                            response = stream_messages(
                                st.session_state['graph'],
                                text=user_input,
                                thread=thread_config,
                                image_path=st.session_state['current_image']
                            )
                        else:
                            response = stream_messages(
                                st.session_state['graph'],
                                text=user_input,
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
                        
                        # Rerun to update the chat display
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                
                if clear_pressed:
                    st.session_state['messages'] = []
                    st.experimental_rerun()

                if end_session_pressed:
                    save_session_summary()
                    st.session_state['messages'] = []
                    st.experimental_rerun()
                    

    
    # elif page == "Parental Controls":
    with parental_controls:
        
        # Call the function to display the parental controls page
        parental_controls_page()
