import os
import tempfile
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# --- Setup ---
st.set_page_config(page_title="MAINA - Maintenance Assistant")
st.title("🔧 MAINA - Maintenance Assistant")

# --- Session state setup ---
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# --- Load sample manual automatically ---
sample_manual_path = "sample_manuals/aztech_flexo_manual.pdf"
if os.path.exists(sample_manual_path):
    loader = PyPDFLoader(sample_manual_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.session_state.vectorstore = vectorstore
    st.success("✅ Sample AZTECH Flexo Manual loaded!")

# --- File Upload ---
st.sidebar.header("📄 Upload Your Manuals")
uploaded_files = st.sidebar.file_uploader("Upload PDF manuals", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)
        os.remove(tmp_path)

    # Split and embed
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.session_state.vectorstore = vectorstore
    st.success("✅ Uploaded manuals embedded and ready to answer questions!")

# --- Q&A Interface ---
if st.session_state.vectorstore:
    st.subheader("🔍 Ask a Question")
    question = st.text_input("Type your maintenance query:", placeholder="What does error code 102 mean?")
    ask_button = st.button("🔍 Ask")

    if ask_button and question:
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            return_source_documents=True
        )
        result = qa_chain({"query": question})
        answer = result['result']
        source_docs = result['source_documents']

        st.markdown("""
        ---
        💡 **Answer:**
        """)
        st.write(answer)

        st.markdown("""
        ---
        📚 **Source(s):**
        """)
        for doc in source_docs:
            st.write(f"{doc.metadata.get('source', 'Unknown Source')}")

        # Optional Features
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🧭 Step-by-Step Fix"):
                followup_q = f"Give step-by-step instructions to resolve: {question}"
                step_result = qa_chain({"query": followup_q})
                st.info(step_result['result'])
        with col2:
            if st.button("📸 View Diagram"):
                diagram_q = f"Is there a diagram related to: {question}? If yes, describe or name it."
                diagram_result = qa_chain({"query": diagram_q})
                st.info(diagram_result['result'])
        with col3:
            st.markdown("Coming Soon: 🗣️ Voice Input")

else:
    st.info("👈 Upload manuals from the sidebar or use the sample one to get started.")
