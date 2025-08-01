# RAG-PDF-CHATBOT
This is a simple yet powerful Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Streamlit, and Groq's LLaMA3 model. The chatbot reads a PDF document and allows users to ask questions, generating accurate, context aware answers by retrieving the most relevant content chunks.


#🤖 RAG Chatbot with Groq, LangChain & Streamlit :

A smart, lightweight PDF-based RAG chatbot that lets users ask questions from a document and get precise answers using LangChain's retrieval pipeline and Groq’s blazing-fast LLaMA3 model — all wrapped in a simple Streamlit UI.

#📌 Features
Retrieval-Augmented Generation (RAG) using LangChain's RetrievalQA

LLM Support via Groq API with LLaMA3 (llama3-8b-8192)

PDF Processing: Reads and embeds PDFs using HuggingFace embeddings

Chunking: Text split using RecursiveCharacterTextSplitter

Vectorstore: Document vector indexing with VectorstoreIndexCreator

Chat UI with history powered by Streamlit

Easy-to-run locally with minimal setup

#🧱 Tech Stack
Frontend: Streamlit

Backend: LangChain + Groq

Embeddings: HuggingFace (all-MiniLM-L12-v2)

Vector Indexing: In-memory vector store

PDF Loader: LangChain's PyPDFLoader

#📁 Project Structure
your_script.py: Main Streamlit app integrating all logic

MembersFAQ.pdf: Sample document used for querying

requirements.txt: All required Python dependencies

#⚙️ Setup Instructions
Clone the repository:

cd langchain-groq-rag-chatbot

Install dependencies:
pip install -r requirements.txt

Set your Groq API key:

As environment variable
export GROQ_API_KEY=your_key_here (Linux/macOS)
set GROQ_API_KEY=your_key_here (Windows)


#📚 Example Usage
Just type your question in the chat like:
“What benefits are included in the membership?”
The chatbot retrieves the most relevant answer from the PDF and responds instantly.

#🔐 API Key Required
Groq API Key (get it from https://console.groq.com/)
Model used: llama3-8b-8192

#💡 Future Improvements
Add multi-turn memory (chat history context)

Support OpenAI or Mixtral models

Enable RAG over multiple PDFs or external documents

Add authentication and analytics

#📜 License
MIT License. Free to use, modify, and share.

#✨ Acknowledgements
Powered by:

LangChain

Groq

Streamlit
