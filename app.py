import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI

import os
from dotenv import load_dotenv

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


llm = GoogleGenerativeAI(model="gemini-2.5-pro", google_api_key="AIzaSyA4NrkxLM36FDW0GKNcdwi57u_F-w9eU78")


# Create retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
)


# For WebPage with:  Streamlit Interface
st.set_page_config(page_title="Bilingual RAG Assistant on JMP Wash Dataset", page_icon="💧", layout="wide")
st.title("Bilingual RAG Assistant on JMP Wash Dataset")
st.markdown("Ask your question in **Bangla** or **English**:")


# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Display chat history first
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## Chat History")
    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}:** {chat['question']}")
        st.markdown(f"**A{i}:** {chat['answer']}")

# Place text input at the bottom and clear after submit
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("Your Question", value="")
    submitted = st.form_submit_button("Send")
    if submitted and query:
        response = qa_chain.run(query)
        st.session_state.chat_history.append({"question": query, "answer": response})
