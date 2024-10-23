import openai
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image as PilImage
import os
from agent_helpers.api_keys import get_keys
from agent_helpers.multi_agent import create_nodes, stream_messages
from helpers import inject_custom_css

# Initialize OpenAI key and create agent graph
openai_key = get_keys()
inject_custom_css()
if 'graph' not in st.session_state:
    st.session_state['graph'] = create_nodes(openai_key)

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
    st.image(uploaded_image, caption="Your uploaded artwork", use_column_width=True)

# Start Chat button
if not st.session_state['chat_active']:
    if st.button("💬 Start Chat!"):
        st.session_state['chat_active'] = True
        st.experimental_rerun()

# Chat interface
if st.session_state['chat_active']:
    # Create a container for the chat
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for msg in st.session_state['messages']:
            if isinstance(msg, dict):
                role = msg.get('role', '')
                content = msg.get('content', '')
                
                # Handle different types of content
                if isinstance(content, dict):
                    # Extract content from AIMessage if present
                    for node_key in ['conversation_moderator_node', 'image_moderator_node']:
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
        
        # Chat input
        user_input = st.text_input("Type your message here:", key="user_input")
        col1, col2 = st.columns([1, 4])
        
        # Send button
        with col1:
            send_pressed = st.button("Send", key="send_button")
        
        # Clear chat button
        with col2:
            clear_pressed = st.button("Clear Chat")
        
        if send_pressed and user_input:
            # Add user message to history
            st.session_state['messages'].append({
                "role": "user",
                "content": user_input
            })
            
            # Get AI response
            thread_config = {"configurable": {"user_id": "1", "thread_id": "1"}}
            
            try:
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
                content = None
                if isinstance(response, dict):
                    for node_key in ['conversation_moderator_node', 'image_moderator_node']:
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
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
            # Clear the input
            # st.session_state['user_input'] = ""
            # st.experimental_rerun()
        
        if clear_pressed:
            st.session_state['messages'] = []
            st.experimental_rerun()

        # Download chat history button
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

