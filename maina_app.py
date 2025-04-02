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

st.set_page_config(page_title="MAINA - Maintenance Assistant")
st.title("🔧 MAINA - Maintenance Assistant")

# Embedding the logo as base64 string
logo_base64 = "{img_base64}"

# Display the logo in the app
st.image("data:image/png;base64," + logo_base64, width=300)

st.markdown("""
    ### **How to use MAINA:**
    1. **Ask your maintenance questions** directly in text or upload a voice query.
    2. **Get instant answers** sourced from your equipment manuals.
    3. **Click on step-by-step fixes** for detailed troubleshooting guides.
""")

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'question' not in st.session_state:
    st.session_state.question = ""

if 'run_query' not in st.session_state:
    st.session_state.run_query = ""

def load_sample_manuals():
    folder_path = "sample_manuals"
    all_docs = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(folder_path, filename))
                all_docs.extend(loader.load())
    return all_docs

sample_docs = load_sample_manuals()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(sample_docs)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.from_documents(chunks, embeddings)
st.session_state.vectorstore = vectorstore

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

    new_chunks = text_splitter.split_documents(uploaded_docs)
    vectorstore = FAISS.from_documents(chunks + new_chunks, embeddings)
    st.success("✅ Uploaded manuals added to the assistant!")
    st.session_state.vectorstore = vectorstore

st.subheader("🔍 Ask a Question")
question_input = st.text_input("Type your maintenance query:", value=st.session_state.question, placeholder="What does error code 102 mean?")
voice_input = st.file_uploader("🎙️ Or upload a voice question (WAV format)", type=["wav"])

if voice_input is not None:
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(voice_input.read())) as source:
        audio_data = recognizer.record(source)
        try:
            question_input = recognizer.recognize_google(audio_data)
            st.success(f"Recognized question: {question_input}")
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError:
            st.error("Error with the speech recognition service")

if st.button("🔍 Ask"):
    st.session_state.question = question_input
    st.session_state.run_query = "ask"

if st.button("🧭 Step-by-Step Fix"):
    st.session_state.run_query = "step"

if st.session_state.run_query in ["ask", "step"] and st.session_state.question:
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(),
        return_source_documents=True
    )
    if st.session_state.run_query == "ask":
        result = qa_chain({"query": st.session_state.question})
        answer = result['result']
        source_docs = result['source_documents']
        st.markdown("### 💡 Answer:")
        st.write(answer)
        st.markdown("### 📚 Source(s):")
        for doc in source_docs:
            st.write(f"{doc.metadata.get('source', 'Unknown Source')}")
    elif st.session_state.run_query == "step":
        followup_q = f"Give step-by-step instructions to resolve: {st.session_state.question}"
        step_result = qa_chain({"query": followup_q})
        st.info(step_result['result'])

    st.session_state.run_query = ""
else:
    st.info("👈 Load default manuals or upload new ones to get started.")

# Footer with company tagline
st.markdown("""
    ---
    **Powered by BMGI**  
    Unlocking Potential. Delivering Results.  
    Celebrating 30 Years
""")
