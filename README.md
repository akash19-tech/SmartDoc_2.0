# 🧠 SmartDoc RAG: Multi-Document AI Assistant

An advanced Retrieval-Augmented Generation (RAG) application that combines **local document analysis** with **real-time web search**. Built with Python, Streamlit, LangChain, and Groq's Llama 3.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Groq](https://img.shields.io/badge/AI-Groq_Llama3-orange)

## 🚀 Key Features

- **📄 Multi-Document RAG**: Upload multiple PDF files and ask questions across all of them simultaneously.
- **🌍 Hybrid Search**: Intelligent agent that combines answers from your PDFs *and* live web search results.
- **�� Smart Query Generation**: Automatically converts user questions into optimized search keywords for better web results.
- **🔍 Precision Context**: Retrieves exact page numbers and relevant text snippets (±50 words context) for every claim.
- **🔗 Verified Sources**: Provides clickable links to web sources and precise citations for PDF content.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq API (Llama 3.3 70B Versatile)
- **Orchestration**: LangChain
- **Vector Store**: ChromaDB
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Search Tool**: DuckDuckGo Search

## ⚙️ Installation & Setup

1. **Clone the repository**

2. **Create a Virtual Environment**

3. **Install Dependencies**

4. **Set up Environment Variables**
Create a `.env` file in the root directory and add your Groq API key:

5. **Run the Application**

## 💡 How to Use

1. **Upload PDFs**: Use the sidebar to upload one or more PDF documents.
2. **Ask a Question**: Type your query in the main text box (e.g., "What are the key financial results?").
3. **View Results**: 
- The AI generates a detailed answer.
- **Left Column**: Shows exact text excerpts from your PDFs with page numbers.
- **Right Column**: Shows related web search results with clickable links.

## 📂 Project Structure

rag-multi-doc-app/
├── app.py # Main application logic
├── requirements.txt # Project dependencies
├── .env # API keys (not committed)
├── .gitignore # Ignored files
└── README.md # Project documentation

Contributions, issues, and feature requests are welcome! Feel free to check the (https://github.com/akash19-tech/SmartDoc_2.0).

