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

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    
    # Initialize OpenAI key and create agent graph
    openai_key = get_keys()
    inject_custom_css()
    
    page = st.sidebar.selectbox("Select a page", ["Home", "Parental Controls"])
    if 'graph' not in st.session_state:
        st.session_state['graph'] = create_nodes(openai_key)
    
    if page == "Home":
        # App title and description
        st.title("🎨 AI ArtBuddy")
        st.subheader("Let's Draw!")
        st.write("Ask me anything about art and drawing! I'm here to help you learn and have fun. 😊")
        
        # Initialize session state variables
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
            st.image(uploaded_image, caption="Your uploaded artwork", width = 200)
            image_moderator_result = image_moderation(openai_key, image_path = save_path)
        
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
                        
                        # Handle different types of content
                        if isinstance(content, dict):
                            # Extract content from AIMessage if present
                            for node_key in ['conversation_moderator_node']:
                                if node_key in content:
                                    messages = content[node_key].get('messages', [])
                                    if messages and hasattr(messages[0], 'content'):
                                        content = messages[0].content
                                        break
                        
                        # Display the message
                        if role == 'user':
                            st.write(f'👤 **You:** {content}')
                        elif role == 'assistant':
                            st.write(f'🎨 **ArtBuddy:** {content}')
            
            # Create a spacer
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Input area in the second container
            with input_container:
                # Chat input
                user_input = st.text_input("Type your message here:", key="user_input")
                
                # Buttons in columns
                col1, col2, col3 = st.columns([1, 1, 4])
                
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
                
                if send_pressed and user_input:
                    # Add user message to history
                    st.session_state['messages'].append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Get AI response
                    thread_config = {"configurable": {"user_id": "1", "thread_id": "1"}}
                    
                    try:
                        response = None
                        if st.session_state['current_image']:
                            if image_moderator_result is True: 
                                response = stream_messages(
                                    st.session_state['graph'],
                                    text=user_input,
                                    thread=thread_config,
                                    image_path=st.session_state['current_image']
                                )
                            else:
                                moderation_message = ("Oops! This app is designed for your own art sketches. "
                                                   "Please try uploading a drawing or sketch you've made by hand. "
                                                   "We can't wait to see your creative artwork! 🎨")
                                st.session_state['messages'].append({
                                    "role": "assistant",
                                    "content": moderation_message
                                })
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
                                for node_key in ['conversation_moderator_node']:
                                    if node_key in response:
                                        messages = response[node_key].get('messages', [])
                                        if messages and hasattr(messages[0], 'content'):
                                            content = messages[0].content
                                            break
                            
                            if content:
                                st.session_state['messages'].append({
                                    "role": "assistant",
                                    "content": content
                                })
                        
                        # Rerun to update the chat display
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                
                if clear_pressed:
                    st.session_state['messages'] = []
                    st.experimental_rerun()
    
    elif page == "Parental Controls":
        st.title("Parental Controls")
        # Call the function to display the parental controls page
        parental_controls_page()

else:
    login, registration = st.tabs(["Login", "Register"])
    with login:
        login_page()
    with registration:
        registration_page()
    
        
