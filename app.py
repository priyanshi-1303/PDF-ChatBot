import streamlit as st
import os
import tempfile

# Try different import approaches for deployment
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    try:
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
    except ImportError:
        st.error("Required packages not installed. Please check requirements.txt")
        st.stop()

# Configure page
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide"
)

# Enhanced CSS
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
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None

def process_pdf(uploaded_file):
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load and process PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Clean up temp file
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        
        if not documents:
            return None, "PDF could not be read or is empty"
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = load_embeddings()
        if embeddings is None:
            return None, "Failed to load embeddings model"
            
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        return vectorstore, f"✅ Successfully processed {len(documents)} pages with {len(docs)} chunks!"
        
    except Exception as e:
        return None, f"❌ Error processing PDF: {str(e)}"

def main():
    # Title
    st.markdown('<h1 class="main-title">📚 RAG-based PDF Chatbot</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Project Info")
        st.markdown("""
        **🚀 Features:**
        - 📄 PDF Upload & Processing
        - 🤖 AI-powered Q&A
        - 🔍 Vector Search (FAISS)
        - ⚡ Real-time Results
        
        **🛠️ Tech Stack:**
        - 🌟 Streamlit
        - 🔗 LangChain
        - 📊 FAISS Vector DB
        - 🤗 HuggingFace Transformers
        
        **💡 How to use:**
        1. Upload your PDF file
        2. Click 'Process PDF'
        3. Ask questions about content
        4. Get intelligent answers!
        """)
    
    # Main content
    st.markdown("### 📄 Upload Your PDF Document")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF file to analyze", 
            type="pdf",
            help="Upload any PDF document to start asking questions"
        )
    
    with col2:
        st.write("")
        st.write("")
        if uploaded_file and st.button("🚀 Process PDF", key="process_btn"):
            with st.spinner("🔄 Processing your PDF..."):
                vectorstore, message = process_pdf(uploaded_file)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.success(message)
                else:
                    st.error(message)
    
    # Q&A Section
    if st.session_state.vectorstore:
        st.markdown("---")
        st.markdown("### 💬 Ask Questions About Your PDF")
        
        question = st.text_input(
            "What would you like to know?",
            key="question_input",
            placeholder="e.g., What is the main topic discussed?"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔍 Ask Question", key="ask_btn") and question:
                try:
                    with st.spinner("🤔 Searching..."):
                        docs = st.session_state.vectorstore.similarity_search(question, k=3)
                        
                        if docs:
                            st.session_state.messages.append({"role": "user", "content": question})
                            
                            answer = "**📖 Found relevant information:**\n\n"
                            for i, doc in enumerate(docs, 1):
                                content = doc.page_content.strip()
                                if len(content) > 300:
                                    content = content[:300] + "..."
                                answer += f"**Source {i}:**\n{content}\n\n"
                            
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            st.warning("🔍 No relevant information found!")
                            
                except Exception as e:
                    st.error(f"❌ Search error: {str(e)}")
        
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_btn") and st.session_state.messages:
                st.session_state.messages = []
                st.rerun()
        
        # Display chat history
        if st.session_state.messages:
            st.markdown("---")
            st.markdown("### 💭 Chat History")
            
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="background: rgba(52, 152, 219, 0.2); border-left: 4px solid #3498db; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                        <strong>🙋‍♀️ You:</strong> {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: rgba(46, 204, 113, 0.2); border-left: 4px solid #2ecc71; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                        <strong>🤖 AI Assistant:</strong><br>
                        {msg['content'].replace('**', '<strong>').replace('**', '</strong>')}
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        st.markdown("---")
        st.info("👆 **Getting Started:** Upload a PDF file to begin!")
        
        st.markdown("""
        ### 💡 Example Questions You Can Ask:
        - What is the main topic of this document?
        - Can you summarize the key points?
        - What are the important findings?
        - Who are the main people mentioned?
        - What conclusions are drawn?
        """)

if __name__ == "__main__":
    main()
