import os
import tempfile
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import speech_recognition as sr
import io

# --- Setup ---
st.set_page_config(page_title="MAINA - Maintenance Assistant")
st.title("🔧 MAINA - Maintenance Assistant")

# --- Session state setup ---
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# --- Load sample manuals from sample_manuals/ folder ---
def load_sample_manuals():
    folder_path = "sample_manuals"
    all_docs = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(folder_path, filename))
                all_docs.extend(loader.load())
    return all_docs

# --- Load and embed sample manuals on startup ---
sample_docs = load_sample_manuals()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(sample_docs)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# --- File Upload ---
st.sidebar.header("📄 Upload Your Manuals")
uploaded_files = st.sidebar.file_uploader("Upload PDF manuals (optional)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    uploaded_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        uploaded_docs.extend(loader.load())
        os.remove(tmp_path)

    # Split and embed uploaded docs
    new_chunks = text_splitter.split_documents(uploaded_docs)
    vectorstore = FAISS.from_documents(chunks + new_chunks, embeddings)
    st.success("✅ Uploaded manuals added to the assistant!")

# Store vectorstore
st.session_state.vectorstore = vectorstore

# --- Q&A Interface ---
if st.session_state.vectorstore:
    st.subheader("🔍 Ask a Question")
    question = st.text_input("Type your maintenance query:", placeholder="What does error code 102 mean?")
    voice_input = st.file_uploader("🎙️ Or upload a voice question (WAV format)", type=["wav"])

    if voice_input is not None:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(voice_input.read())) as source:
            audio_data = recognizer.record(source)
            try:
                question = recognizer.recognize_google(audio_data)
                st.success(f"Recognized question: {question}")
            except sr.UnknownValueError:
                st.error("Could not understand audio")
            except sr.RequestError:
                st.error("Error with the speech recognition service")

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

        st.markdown("""---
💡 **Answer:**""")
        st.write(answer)

        st.markdown("""---
📚 **Source(s):**""")
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
            st.markdown("🗣️ Voice input supported via WAV upload")
else:
    st.info("👈 Load default manuals or upload new ones to get started.")
