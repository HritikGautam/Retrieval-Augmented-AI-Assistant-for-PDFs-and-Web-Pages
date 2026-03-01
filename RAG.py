# Imports
import streamlit as st
# import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv

''' Important info:- 
To run this project, please use the "Data Science Syllabus" PDF, because I have limited the 
number of pages to be processed from the PDF to 3 to reduce processing time(Line 82).
Embeddings model :- Ollama's Llama 3.2
GROQ LLM model :- groq/compound
Question to be asked to the llm:- "What is the growth rate of data science jobs?"
'''
load_dotenv()
groq_api_key = st.secrets['GROQ_API_KEY']
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"] # Hugging face token

#Title of the app
st.title("RAG App (URL/PDF)")

# LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="groq/compound"
)

# type of input selection

input_type = st.radio(
    "Choose Input Type",
    ["URL", "PDF"]
)


# input for URL

if input_type == "URL":
    url_input = st.text_input("Enter URL")


# Input for PDF

if input_type == "PDF":
    pdf_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )


# Process button

if st.button("Process"):

    # Validation of url and pdf inputs
    
    if input_type == "URL" and not url_input:
        st.warning("Please enter a URL first")
        st.stop()

    if input_type == "PDF" and not pdf_file:
        st.warning("Please upload a PDF first")
        st.stop()


    # loading documents

    if input_type == "URL":
        loader = WebBaseLoader(url_input)
        docs = loader.load()

    if input_type == "PDF":

        # Save temporary pdf
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        docs = docs[2:5] # Limit to pages from 3 to 5 for testing

    
    # Text Splitting

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    # Above we are limiting the number of documents/chunks "[:20]" to be processed to 20 to reduce processing time.
    # Suppose if a URL have only one page but it is too long, then it will be similar to creating chunks for more number of pages.
    # As we reduced the number of pages/docs earlier for pdf to create embeddings faster, we are limiting the number of chunks to be 
    # processed to 20 here to reduce processing time. You can increase it as per your requirement and system capabilities.
    
    final_documents = st.session_state.text_splitter.split_documents(docs)[:20]
   

    # Making embeddings
    if hf_token:
        st.write(f"Token diagnostic: ✅ Loaded (Starts with: {hf_token[:5]}...)")
    else:
        st.error("❌ Token diagnostic: NOT LOADED. Check your Streamlit Secrets name.")
        st.stop()
    
    st.session_state.embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )
    # st.session_state.embeddings = OllamaEmbeddings(model="llama3.2")
    
    try:
        test_res = st.session_state.embeddings.embed_query("test")
        st.write("Embedding check: ✅ Success")
    except Exception :
        st.error("❌ API returned empty or is loading. Wait 30s and click Process again.")
        st.stop()

    test_res = st.session_state.embeddings.embed_query("test")
    st.write(f"Embedding check: {'✅ Success' if test_res else '❌ EMPTY'}")
    if not test_res: 
        st.stop()

    # --- DEBUG CHECK ---
    st.write(f"1. Raw Docs Loaded: {len(docs)}")
    st.write(f"2. Final Chunks Created: {len(final_documents)}")

    if not final_documents:
        st.error("❌ STOPPING: 'final_documents' is empty. FAISS will crash if we continue.")
        st.stop()
    # Vector Store Creation

    vectors = FAISS.from_documents(
        final_documents,
        st.session_state.embeddings
    )


    # Store vectors
    st.session_state.vectors = vectors

    st.success("Processing Complete")


#Base prompt template

prompt_template = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question

<context>
{context}
</context>

Question:
{input}
"""
)


# User question is asked

user_question = st.text_input("Ask Question")



# User question is processed

if user_question:

    if "vectors" not in st.session_state:

        st.warning("Please process a URL or PDF first")
        st.stop()


    document_chain = create_stuff_documents_chain(
        llm,
        prompt_template
    )

    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start = time.time()

    response = retrieval_chain.invoke({
        "input": user_question
    })

    st.write(response['answer'])

    print("Response time:", time.time() - start)


    # Show retrieved documents

    with st.expander("Document Similarity Search"):

        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("--------------------")