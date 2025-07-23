import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load documents (JMP WASH data)
loader = TextLoader("jmp_data.txt", encoding='utf-8')
documents = loader.load()

# Split documents into manageable chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Load multilingual embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Create or load vector store
if not os.path.exists("./db"):
    vector_store = Chroma.from_documents(chunks, embedding_model, persist_directory="./db")
else:
    vector_store = Chroma(persist_directory="./db", embedding_function=embedding_model)

# Load LLM from Google Generative AI
llm = GoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")  # Use .env for security
)

# Create retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
)

# Streamlit page configuration
st.set_page_config(page_title="Bilingual RAG Assistant on JMP Wash Dataset", page_icon="💧", layout="wide")
st.title("Bilingual RAG Assistant on JMP Wash Dataset")
st.markdown("Ask your question in **Bangla** or **English**:")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show previous messages
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

# Handle new user input
if prompt := st.chat_input("Your Question"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.spinner("💬 Generating response..."):
        try:
            response = qa_chain.run(prompt)
        except Exception as e:
            response = "something went wrong."
            st.error(str(e))

    with st.chat_message("assistant"):
        st.markdown(response)

    # Save conversation to session state
    st.session_state.chat_history.append({
        "question": prompt,
        "answer": response
    })
