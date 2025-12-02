import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st 
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import tempfile
import shutil
from pathlib import Path
import shutil
import chromadb

from langchain_community.tools import DuckDuckGoSearchResults
import re

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize cached components
@st.cache_resource
def init_embeddings():
    """Initialize sentence transformer embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def init_llm():
    """Initialize Groq LLM via Groq API"""
    # Try getting key from Streamlit secrets first, then environment variable
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please check your .env file or Streamlit secrets.")
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )

def process_documents(uploaded_files):
    """Process uploaded PDF files into vector store"""
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    if not uploaded_files:
        return None
    
    docs = []
    temp_dir = Path(tempfile.mkdtemp())
    
    # Clean previous chroma db
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    st.info(f"Processing {len(uploaded_files)} documents...")
    
    for file in uploaded_files:
        if file.name.endswith('.pdf'):
            # Save uploaded file temporarily
            file_path = temp_dir / file.name
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file, f)
            
            # Load PDF with page metadata
            loader = PyPDFLoader(str(file_path))
            pdf_docs = loader.load()
            
            # Add source and page metadata
            for i, doc in enumerate(pdf_docs):
                doc.metadata.update({
                    'source': file.name,
                    'page': i + 1
                })
            docs.extend(pdf_docs)
            st.success(f"✅ Loaded {file.name} ({len(pdf_docs)} pages)")
    
    if not docs:
        st.error("No valid PDF documents found!")
        return None
    
    # Split documents into chunks
    st.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(docs)
    st.success(f"Created {len(splits)} text chunks")
    
    # Create vector store
    st.info("Generating embeddings and indexing...")
    embeddings = init_embeddings()

    if os.path.exists("./chroma_db"):
        try:
            shutil.rmtree("./chroma_db")  # Force delete old DB
        except Exception as e:
            print(f"Warning: Could not delete old DB: {e}")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    st.success("✅ Vector store ready!")
    return vectorstore

def get_context_with_surround(retriever, question, k=3):
    """Get relevant context with 50 words back and 100 words forward around key content"""
    relevant_docs = retriever.invoke(question) # Updated to invoke for newer langchain versions
    contexts = []
    
    for doc in relevant_docs:
        words = doc.page_content.split()
        # If the chunk is very small, just return the whole thing
        if len(words) < 150: 
            context = doc.page_content
        else:
            # Find a rough center or just use the beginning if searching is hard
            # In a simple RAG, the whole chunk is "relevant", so we often just take the chunk.
            # But to strictly follow your logic of "around the center":
            center_idx = len(words) // 2
            
            # 50 words back
            start_idx = max(0, center_idx - 50)
            # 100 words forward
            end_idx = min(len(words), center_idx + 100)
            
            context = ' '.join(words[start_idx:end_idx])
        
        contexts.append({
            'document': doc.metadata.get('source', 'Unknown'),
            'page': doc.metadata.get('page', 'N/A'),
            'context': context
        })
    
    return contexts

def generate_search_query(llm, question):
    """Generates a keyword-optimized search query using the LLM"""
    prompt = f"""
    You are a search query optimizer. Convert the user's question into a specific, 3-5 word keyword search query for a search engine.
    Remove words like "the document", "pdf", "mentioned". Focus on the core topic.
    
    User Question: {question}
    
    Search Query:
    """
    response = llm.invoke(prompt)
    # Clean up response to get just the text
    return response.content.strip().replace('"', '')

def search_web(query):
    """Performs a web search and returns results with links"""
    search = DuckDuckGoSearchResults()
    try:
        return search.run(query) 
    except Exception as e:
        return f"Error searching web: {e}"


# Streamlit UI Configuration
st.set_page_config(
    page_title="Multi-Doc RAG Assistant", 
    page_icon="🔍",
    layout="wide"
)

# Header
st.title("🔍 Multi-Document RAG Assistant")
st.markdown("**Upload PDFs → Ask Questions → Get Answers with Source References**")

# Sidebar for document management
st.sidebar.header("📁 Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",   # non-empty
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="visible"        # or "collapsed"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Process documents when uploaded
if uploaded_files:
    with st.spinner("🔄 Processing your documents..."):
        vectorstore = process_documents(uploaded_files)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            st.sidebar.success(f"✅ Processed {len(uploaded_files)} documents")
            # NEW: List file names in the main panel
            st.markdown("#### Uploaded Documents:")
            for f in uploaded_files:
                st.write(f"• `{f.name}`")
            st.sidebar.markdown("---")


# Clear button
if st.sidebar.button("🗑️ Clear Documents"):
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.rerun()

# Main content area
col1, col2 = st.columns([1, 3])

with col1:
    st.header("❓ Ask Question")
    question = st.text_input(
        "Your question",           # non-empty label
        placeholder="What are the key findings...?",
        key="question",
        label_visibility="visible"  # or "collapsed" if you want to hide it
    )

with col2:
    st.header("📊 Status")
    if st.session_state.vectorstore:
        st.success("✅ Ready to answer questions!")
        st.info("📈 Vector store active with documents loaded")
    else:
        st.warning("⚠️ Please upload PDF documents first")

# Answer generation
if question:
    with st.spinner("🧠 Thinking & Searching..."):
        try:
            llm = init_llm()
            web_query = generate_search_query(llm, question)
            st.toast(f"Searching web for: '{web_query}'")
            
            pdf_context = ""
            contexts = []
            if st.session_state.retriever:
                contexts = get_context_with_surround(st.session_state.retriever, question)
                pdf_text = "\n\n".join([f"Document: {c['document']} (Page {c['page']}): {c['context']}" for c in contexts])
                pdf_context = f"### PDF DOCUMENT CONTEXT:\n{pdf_text}\n"
            
            web_results = search_web(web_query)
            web_context = f"### WEB SEARCH RESULTS for '{web_query}':\n{web_results}\n"
            
            combined_context = pdf_context + "\n" + web_context
            
            prompt_template = f"""
            You are an expert researcher. Answer the user's question using the provided context.
            
            GUIDELINES:
            1. Answer in detail (3-4 paragraphs).
            2. CITATIONS ARE MANDATORY:
               - Use [Document Name, p.X] for PDF info.
               - Use [Source Title] for Web info.
            3. Prioritize the PDF content, but use Web content to fill gaps or provide current updates.
            
            CONTEXT:
            {combined_context}
            
            USER QUESTION: 
            {question}
            
            ANSWER:
            """
            
            response = llm.invoke(prompt_template)
            
            st.header("🤖 Detailed Answer")
            st.markdown(response.content)
            
            # --- Sources Display ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 PDF Sources")
                if contexts:
                    for ctx in contexts:
                        with st.expander(f"{ctx['document']} - p.{ctx['page']}"):
                            st.write(ctx['context'])
                else:
                    st.info("No PDF context used.")

            with col2:
                st.subheader("🌍 Web Sources")
                import re
                links = re.findall(r"link:\s*(https?://[^,\]\s]+)", str(web_results))
                titles = re.findall(r"title:\s*([^,]+)", str(web_results))
                
                if links:
                    st.markdown(f"**Search Query used:** `{web_query}`")
                    st.markdown("### 🔗 Related Links:")
                    for i, link in enumerate(links):
                        # Safety check in case titles/links count doesn't match
                        title = titles[i] if i < len(titles) else "Source Link"
                        # Clean up title
                        title = title.strip().replace("'", "")
                        st.markdown(f"{i+1}. [{title}]({link})")
                    
                    with st.expander("View Raw Search Data"):
                        st.write(web_results)
                else:
                    st.info("No specific links found.")
                    with st.expander("View Raw Output"):
                        st.write(web_results)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Check your API key and internet connection.")

elif question and not st.session_state.vectorstore:
    st.warning("⚠️ Please upload documents first!")

# Footer
st.markdown("---")
st.markdown("Made and Maintained by Akash Mishra")
