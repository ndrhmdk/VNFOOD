import streamlit as st
API_KEY = st.secrets["GEMINI_API_KEY"]


GEMINI_INSTRUCTION = """
    From the given image, please describe the ingredients that you see and return them in a valid json format as followed:
    {
        "ingredients" : ['beef', 'pepper', 'egg']
    }
    Remember not to make up any ingredients that you don't find in the image
"""