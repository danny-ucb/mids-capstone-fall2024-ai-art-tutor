import openai
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
from helpers import generate_response, inject_custom_css
import os
 


inject_custom_css()

# Title for the app
st.title("🎨 AI ArtBuddy") 
st.subheader("Let's Draw!") 

st.write("Ask me anything about art and drawing! I’m here to help you learn and have fun. 😊")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant for 6 to 8 year old kids interested in drawing and art."}
    ]

if 'chat_active' not in st.session_state:
    st.session_state['chat_active'] = False  # Track if chat is active    

test_image_url = "https://media.npr.org/assets/img/2014/12/05/family-drawing-examples-together_wide-ea9ec863740594906c9e520cd05e29da72b54887.jpg?s=1400&c=100&f=jpeg"
# Sidebar options
st.sidebar.header("Choose Input Method")
option = st.sidebar.radio("Select an option:", ("Upload Image", "Draw"))

if option == "Upload Image":
    # Upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open and display the image using PIL
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

elif option == "Draw":
    # Canvas settings
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color: ", "#000000")
    bg_color = st.sidebar.color_picker("Background color: ", "#ffffff")
    drawing_mode = st.sidebar.selectbox(
        "Drawing mode:", ("freedraw", "rect", "circle", "transform")
    )
    
    # Canvas to draw on
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=400,
        width=600,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    
    # If a drawing is made, save it
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data, caption='Your Drawing')
            # Convert NumPy array to PIL image
        img = Image.fromarray(canvas_result.image_data)



# Footer
st.sidebar.markdown("### Select an image or draw on the canvas.")
# chat_button = """
#     <a href="#" class="chat-button">💬 Chat Now!</a>
# """
# st.markdown(chat_button, unsafe_allow_html=True)
if st.button("💬 Chat Now!"):
    st.session_state['chat_active'] = True  # Set chat state to active

if st.session_state['chat_active']:
    st.write("**Chat History**")
    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.write(f"**You:** {message['content']}")
        elif message['role'] == 'assistant':    
            st.write(f"**AI ArtBuddy:** {message['content']}")

    user_input = st.text_input("You:", "", placeholder="Type something...")

    # Send button
    if st.button("Send"):
        if user_input:
    #        Append the user's message to the session state
            st.session_state['messages'].append({"role": "user", "content": user_input})

            # Generate a response from the AI based on the conversation history
            ai_response = generate_response(user_input, test_image_url) 

            # # Append the AI's response to the session state
            st.session_state['messages'].append({"role": "assistant", "content": ai_response})

            # # Refresh the chat history after sending the message
            st.experimental_rerun()
        
# Button to clear the conversation
if st.button("Clear Conversation"):
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant for kids interested in drawing and art."}
    ]
    st.session_state['chat_active'] = False  # Reset chat state
    st.experimental_rerun()

# # When user submits input, generate a response from GPT-4
# if chat_button:
#     response = ""
#     if user_input:
#         if option == "Draw":
#             if canvas_result.image_data is not None:
#         # Call the function to get a response from GPT-4
#                 response = generate_response(user_input, image_url = "https://media.npr.org/assets/img/2014/12/05/family-drawing-examples-together_wide-ea9ec863740594906c9e520cd05e29da72b54887.jpg?s=1400&c=100&f=jpeg")
#         if option == "Upload Image":
#             if uploaded_file is not None:
#                 response = generate_response(user_input, image_url = "https://media.npr.org/assets/img/2014/12/05/family-drawing-examples-together_wide-ea9ec863740594906c9e520cd05e29da72b54887.jpg?s=1400&c=100&f=jpeg")

# st.write(f"AI ArtBuddy: {response}")

