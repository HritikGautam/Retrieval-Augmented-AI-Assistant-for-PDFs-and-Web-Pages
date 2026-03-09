# Retrieval-Augmented AI Assistant for PDFs and Web Pages

An AI-powered assistant that allows users to ask questions from PDFs or website URLs.

The system uses Retrieval-Augmented Generation (RAG) with LangChain to retrieve relevant information from documents or web pages and generate accurate answers using a Large Language Model.

Instead of relying only on the model’s internal knowledge, the system first retrieves relevant context from the provided sources, making responses more accurate and grounded in real data.

---

# Features

- Upload PDF documents and ask questions about their content
- Provide website URLs to extract and analyze webpage information
- Context-aware AI answers using Retrieval-Augmented Generation
- Semantic search using vector embeddings
- Interactive interface built with Streamlit
- Efficient document retrieval using FAISS vector store

---

# How It Works

The system follows a Retrieval-Augmented Generation (RAG) pipeline.

## 1. Data Input

Users can provide:

- A PDF document
- A Website URL

## 2. Content Extraction

The system extracts text from the provided source:

- PDF text extraction
- Web page content extraction

## 3. Text Chunking

The extracted content is split into smaller chunks to make retrieval more efficient.

## 4. Embedding Generation

Each text chunk is converted into vector embeddings using a sentence transformer model.

## 5. Vector Storage

The embeddings are stored in a FAISS vector database.

## 6. Query Processing

When a user asks a question:

- The question is converted into an embedding
- The system retrieves the most relevant document chunks

## 7. Answer Generation

The retrieved context along with the user question is sent to the LLM, which generates the final answer.

---

# Tech Stack

### Language
- Python

### Frameworks & Libraries
- LangChain
- Streamlit

### LLM
- Groq LLM

### Embeddings
- sentence-transformers/all-MiniLM-L6-v2

### Vector Store
- FAISS

---

# Example Usage

1. Upload a PDF or provide a website URL

2. Ask a question such as:

What are the key topics discussed in this document?

3. The system retrieves relevant information and generates a context-aware answer.
