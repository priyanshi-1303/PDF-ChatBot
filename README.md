Built a RAG-based PDF Question-Answering chatbot using Streamlit, LangChain, FAISS, and HuggingFace embeddings to retrieve semantically relevant content from documents in real time.
The project uses a Retrieval-based approach where PDFs are converted into vector embeddings using HuggingFace models, stored in FAISS, and queried via semantic similarity search using LangChain.
Programming Language: Python

Frontend / UI: Streamlit

Document Processing: PyPDF (PDF Loader)

Text Chunking: Recursive Character Text Splitter

Embeddings: HuggingFace Sentence Transformers

Vector Database: FAISS

Semantic Search: LangChain

Caching & State Management: Streamlit Session State

Deployment: Streamlit Cloud 

Version Control: Git & GitHub
