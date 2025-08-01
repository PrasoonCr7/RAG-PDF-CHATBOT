# # Phase 1 imports
import streamlit as st

# Phase 2 imports
import os
from langchain_groq import ChatGroq # connect to groq llm
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate # for llm receive prompt

# Phase 3 imports 
from langchain.embeddings import HuggingFaceEmbeddings # load the pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter # split in chunks
from langchain.document_loaders import PyPDFLoader # convert text into embeddings (vector) and read pdf
from langchain.indexes import VectorstoreIndexCreator # database (vector)
from langchain.chains import RetrievalQA # retrieve the most relevant chunks


# Set the API key securely (temporary, better via .env or Streamlit secrets)
os.environ["GROQ_API_KEY"] = "Enter Your GROQ API Key"

# Title of the app
st.title("RAG Chatbot !")  # title

# Initialize message history
if 'messages' not in st.session_state: # memory of past messages
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# caching vector store ( pdf stuff )
@st.cache_resource
def get_vectorstore():
    pdf_name = "./MembersFAQ.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    # Create Chunks, aka vectors (ChromaDB)
    index = VectorstoreIndexCreator(
        embedding = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L12-v2'),
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 100)
    ).from_loaders(loaders)
    return index.vectorstore

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

    #LLM Model setup
    model = "llama3-8b-8192"
    groq_chat = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model_name=model
    )
    
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load the documents")
        # connect llm with pdf data
        chain = RetrievalQA.from_chain_type(
            llm = groq_chat,
            chain_type = 'stuff',
            retriever = vectorstore.as_retriever(search_kwargs = ({"k":3})),
            return_source_documents = True
        )
    # Chain setup
        # chain = groq_sys_prompt | groq_chat | StrOutputParser()

    # Generate response
        result = chain({"query":prompt})
        response = result["result"]
        # response = chain.invoke({"user_prompt": prompt})

    # Append and display assistant message
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.chat_message("assistant").markdown(response)
    # Error handling
    except Exception as e:
        st.error(f"Error:[{str(e)}]")
