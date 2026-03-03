import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()
working_dir=os.path.dirname(os.path.abspath(__file__))

embedding=HuggingFaceEmbeddings()

# Load the Llama-3.3-70B model from Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def process_document_to_chroma_db(file_name):
    # Use os.path.join to handle Windows backslashes (\) correctly
    file_path = os.path.join(working_dir, file_name)
    
    # Switch to PyPDFLoader (No 'unstructured_inference' needed!)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # 200 was a bit small, 1000 is better for context
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)
    
    # Store in Chroma
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=os.path.join(working_dir, "doc_vectorstore")
    )
    
    return 0

def answer_question(user_question):
    # Load the persistent Chroma vector database
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    
    # Create a retriever for document search
    retriever = vectordb.as_retriever()
    
    # Create a RetrievalQA chain to answer user questions using Llama-3.3-70B
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    
    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]
    
    return answer