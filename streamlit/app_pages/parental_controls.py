import openai
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image as PilImage
import os
import bcrypt
import json
import random
from helpers import * 
from datetime import datetime, timedelta

def parental_controls_page():
    st.header("Parental Controls")

    # Parental Authentication (Optional)
    with st.expander("Enter Parental Passcode"):
        passcode = st.text_input("Passcode", type="password")
        if passcode != "parent123":  # Example passcode
            st.warning("Incorrect Passcode")
            return

    # Control Options
    st.subheader("Manage Child’s Progress")
    progress_view = st.checkbox("Allow viewing of child’s progress")
    
    st.subheader("Set Time Limits")
    time_limit = st.slider("Set maximum usage time (hours per day)", 0, 5, 1)

    st.subheader("Block or Allow Features")
    block_story_feature = st.checkbox("Block Story Feature")
    block_critique_feature = st.checkbox("Block Critique Feature")

    # Display summary
    st.subheader("Parental Control Summary")
    st.write(f"Time Limit: {time_limit} hours/day")
    st.write("Features Blocked: ", 
             "Story" if block_story_feature else "None", 
             ", Critique" if block_critique_feature else "")

    # Save parental settings (you can use session state, a database, etc.)
    if st.button("Save Settings"):
        # Example of saving parental settings in session state (can also be saved in a DB)
        st.session_state['parental_settings'] = {
            "time_limit": time_limit,
            "block_story_feature": block_story_feature,
            "block_critique_feature": block_critique_feature,
            "progress_view": progress_view
        }
        st.success("Settings saved!")