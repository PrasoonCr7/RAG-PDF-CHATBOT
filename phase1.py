# 1. Chatbot
# 2. Streamlit 
# 3. Usuage from the UI

# Phase 2 have all the coding , this is just part 1 one RAG Chatbot

# # Phase 1 imports
import streamlit as st

# Phase 2 imports
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Set the API key securely (temporary, better via .env or Streamlit secrets)
os.environ["GROQ_API_KEY"] = "Enter Groq API Key"

# Title of the app
st.title("RAG Chatbot !")

# Initialize message history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Take user input
prompt = st.chat_input("Pass your prompt here!")

# If user submits a prompt
if prompt:
    # Append and display user message
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    st.chat_message("user").markdown(prompt)

    # Prompt template
    groq_sys_prompt = ChatPromptTemplate.from_template(
        """You are very smart at everything. You always give the best,
        most accurate, and most precise answers. Answer the following question:
        {user_prompt}
        Start the answer directly. No small talk please."""
    )

    # Model setup
    model = "llama3-8b-8192"
    groq_chat = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model_name=model
    )

    # Chain setup
    chain = groq_sys_prompt | groq_chat | StrOutputParser()

    # Generate response
    response = chain.invoke({"user_prompt": prompt})

    # Append and display assistant message
    st.session_state.messages.append({'role': 'assistant', 'content': response})
    st.chat_message("assistant").markdown(response)
