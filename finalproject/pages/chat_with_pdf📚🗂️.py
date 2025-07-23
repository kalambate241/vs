import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import google.generativeai as genai
import datetime
import io

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Save vector index
def get_vector_store(chunks):
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(chunks, embedding=embed)
    db.save_local("faiss_index")

# Load QA Chain
def get_conversational_chain():
    prompt = PromptTemplate(
        template="""
        Answer the question as detailed as possible from the provided context.
        If the answer is not in the context, just say "answer is not available in the context".

        Context: {context}
        Question: {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle Q&A
def user_input(question):
    embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings=embed, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    answer = response["output_text"]
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append((question, answer, timestamp))
    return answer

# Export chat history
def export_chat():
    buffer = io.StringIO()
    for i, (q, a, t) in enumerate(st.session_state.chat_history, 1):
        buffer.write(f"[{t}] Q{i}: {q}\nA{i}: {a}\n\n")
    filename = f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button("📥 Download Chat", buffer.getvalue(), file_name=filename, mime="text/plain")

# Main app
def main():
    st.set_page_config(page_title="📚 Chat with PDF using Gemini", layout="wide")

    # Compact spacing styles
    st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 1rem !important;
        }
        .stTextInput > div > div {
            margin-bottom: 0.3rem !important;
        }
        .stButton {
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
        }
        h1, h2, h3, h4 {
            margin-bottom: 0.4rem !important;
            margin-top: 0.4rem !important;
        }
        hr {
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        header, footer {
            visibility: visible;
        }
    </style>
""", unsafe_allow_html=True)


    # Title
    st.markdown("### 📚 Chat with PDF using Gemini 💬")
    st.caption("Ask intelligent questions from your uploaded PDF files using Google Gemini + LangChain.")

    # Session state
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar: History
    with st.sidebar:
        st.header("🕘 Chat History")
        with st.container(height=300, border=True):
            if st.session_state.chat_history:
                for i, (q, _, t) in enumerate(reversed(st.session_state.chat_history), 1):
                    st.markdown(f"**[{t}] Q{i}:** {q}")
            else:
                st.info("No chat yet.")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []

    # File Upload
    st.markdown("#### 📂 Upload your PDF file(s)")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown("**Uploaded PDFs:**")
        for f in uploaded_files:
            reader = PdfReader(f)
            st.write(f"📄 {f.name} — {len(reader.pages)} page(s)")

        if st.button("🚀 Process PDFs"):
            with st.spinner("🔄 Processing PDFs..."):
                text = get_pdf_text(uploaded_files)
                chunks = get_text_chunks(text)
                get_vector_store(chunks)
                st.session_state.pdf_processed = True
                st.success("✅ PDFs indexed and ready to chat!")

    # Question input
    st.markdown("#### 💡 Ask a Question")
    col1, col2 = st.columns([5, 1])
    with col1:
        user_q = st.text_input("Type your question here:", label_visibility="collapsed")
    with col2:
        ask = st.button("Ask")

    if ask and user_q:
        if st.session_state.pdf_processed:
            answer = user_input(user_q)
            st.markdown("**🔎 Answer:**")
            st.info(answer)
        else:
            st.warning("⚠️ Please upload and process PDFs before asking questions.")

    if st.session_state.chat_history:
        st.markdown("---")
        export_chat()

if __name__ == "__main__":
    main()
