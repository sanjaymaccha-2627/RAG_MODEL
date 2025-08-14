import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Set up Groq API key
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in secrets or .env file")
    st.stop()

# Streamlit page config
st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")

# Optional Logo
st.image("PragyanAI_Transperent.png", width=250)
st.title("Dynamic RAG with Groq, FAISS, and Llama3")

# Initialize session state
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF documents", 
        type="pdf", 
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                docs = []
                for uploaded_file in uploaded_files:
                    # Save uploaded file to a temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    # Load PDF
                    loader = PyPDFLoader(tmp_path)
                    docs.extend(loader.load())

                # Split text into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                split_docs = splitter.split_documents(docs)

                # Create embeddings
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                # Store in FAISS
                st.session_state.vector = FAISS.from_documents(split_docs, embeddings)

                st.success("✅ Documents processed and indexed!")

# Chat interface
if st.session_state.vector:
    query = st.text_input("Ask a question about your documents:")

    if query:
        with st.spinner("Thinking..."):
            # Create retriever
            retriever = st.session_state.vector.as_retriever()

            # Define LLM
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama3-8b-8192"
            )

            # Create prompt template
            prompt = ChatPromptTemplate.from_template("""
            Answer the question based on the following context:
            {context}

            Question: {input}
            """)

            # Create document chain
            document_chain = create_stuff_documents_chain(llm, prompt)

            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Run the chain
            result = retrieval_chain.invoke({"input": query})

            # Display answer
            st.write("### Answer:")
            st.write(result["answer"])

            # Show retrieved context (optional)
            with st.expander("View retrieved context"):
                for doc in result["context"]:
                    st.write(doc.page_content)
                    st.write("---")
