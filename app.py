import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.main-title {
    text-align: center;
    color: #ffffff;
    padding: 2rem;
    font-size: 2.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    background: rgba(255,255,255,0.1);
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 2rem;
}

.stSidebar {
    background: rgba(0,0,0,0.2) !important;
    backdrop-filter: blur(10px);
}

.stButton > button {
    background: linear-gradient(45deg, #ff6b6b, #ee5a24);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.5rem 2rem;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

.stTextInput > div > div > input {
    background: rgba(255,255,255,0.9) !important;
    color: #2c3e50 !important;
    border-radius: 10px;
    border: 2px solid rgba(255,255,255,0.3);
}

.stFileUploader > div {
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    backdrop-filter: blur(10px);
    border: 2px dashed rgba(255,255,255,0.3);
}

.stInfo, .stSuccess, .stWarning, .stError {
    background: rgba(255,255,255,0.1) !important;
    backdrop-filter: blur(10px);
    border-radius: 10px;
}

.stSuccess {
    border-left: 5px solid #00d4aa !important;
}

.stError {
    border-left: 5px solid #ff4757 !important;
}

/* Make all text white */
.stMarkdown, .stText, p, div {
    color: white !important;
}

.stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3, 
.stSidebar .stMarkdown p, .stSidebar .stMarkdown div {
    color: white !important;
}

.stSubheader {
    color: white !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load and process PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        os.unlink(tmp_file_path)
        
        if not documents:
            return None, "PDF could not be read"
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = load_embeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        return vectorstore, f"✅ Successfully processed {len(documents)} pages!"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def main():
    # Title
    st.markdown('<h1 class="main-title">📚 PDF Question-Answering Chatbot</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Project Info")
        st.write("**Features:**")
        st.write("- PDF Upload & Processing")
        st.write("- AI-powered Q&A")
        st.write("- Vector Search")
        st.write("- Real-time Results")
        
        st.write("**Tech Stack:**")
        st.write("- Streamlit")
        st.write("- LangChain")
        st.write("- FAISS")
        st.write("- HuggingFace")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Upload PDF")
        uploaded_file = st.file_uploader("Choose PDF file", type="pdf")
    
    with col2:
        st.write("")  
        st.write("")  
        if uploaded_file and st.button("Process PDF", key="process_btn"):
            with st.spinner("Processing..."):
                vectorstore, message = process_pdf(uploaded_file)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.success(message)
                else:
                    st.error(message)
    
    # Q&A Section
    if st.session_state.vectorstore:
        st.subheader("💬 Ask Questions")
        
        # Input
        question = st.text_input("Enter your question:", key="question_input")
        
        if st.button("Ask Question", key="ask_btn") and question:
            try:
                # Search
                docs = st.session_state.vectorstore.similarity_search(question, k=3)
                
                if docs:
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    answer = ""
                    for i, doc in enumerate(docs, 1):
                        answer += f"**Result {i}:**\n{doc.page_content[:500]}...\n\n"
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.warning("No relevant information found!")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Display messages
        if st.session_state.messages:
            st.subheader("📋 Chat History")
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.info(f"🙋 You: {msg['content']}")
                else:
                    st.success("🤖 AI Assistant:")
                    st.write(msg["content"])
            
            # Clear button
            if st.button("Clear History", key="clear_btn"):
                st.session_state.messages = []
                st.rerun()
    
    else:
        st.info("👆 Please upload a PDF file to start asking questions!")

if __name__ == "__main__":
    main()