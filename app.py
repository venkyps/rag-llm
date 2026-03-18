import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("🦙 Llama-3.3-70B - Document RAG")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# 1. Use session_state to track if the document has been processed
if "processed" not in st.session_state:
    st.session_state.processed = False

if uploaded_file is not None and not st.session_state.processed:
    save_path = os.path.join(working_dir, uploaded_file.name)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process and update state
    with st.spinner("Processing document..."):
        process_document_to_chroma_db(uploaded_file.name)
        st.session_state.processed = True
        st.success("Document Processed Successfully")

# 2. Add a check to prevent querying before a file is uploaded
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    if st.session_state.processed:
        with st.spinner("Llama is thinking..."):
            answer = answer_question(user_question)
            st.markdown("### Llama-3.3-70B Response")
            st.markdown(answer)
    else:
        st.warning("Please upload and process a document first.")
