import os
import streamlit as st

from utils import (
    load_pdf_documents,
    split_documents,
    get_embedding_model,
    create_vector_store,
    save_vector_store,
    load_vector_store,
    get_retriever,
    get_prompt,
    get_llm,
    answer_question
)

DATA_DIR = "data/sample_pdfs"
VECTOR_PATH = "vector_store/faiss_index"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("vector_store", exist_ok=True)

st.set_page_config(page_title="DocuSense AI", layout="wide")

st.title("DocuSense AI – Intelligent Document Question Answering System")
st.write("Upload one or more PDF documents and ask questions based on their content.")

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

col1, col2 = st.columns([1, 1])

with col1:
    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected.")

with col2:
    process_clicked = st.button("Process Documents")

if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file first.")
    else:
        try:
            file_paths = []

            for uploaded_file in uploaded_files:
                file_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)

            with st.spinner("Reading PDFs, chunking text, and building vector index..."):
                documents = load_pdf_documents(file_paths)
                chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
                embeddings = get_embedding_model()
                vector_store = create_vector_store(chunks, embeddings)
                save_vector_store(vector_store, VECTOR_PATH)

            st.session_state.vector_store_ready = True
            st.success("Documents processed successfully.")

        except Exception as e:
            st.session_state.vector_store_ready = False
            st.error(f"Error while processing documents: {e}")

query = st.text_input("Ask a question about the uploaded documents")

ask_clicked = st.button("Get Answer")

if ask_clicked:
    if not st.session_state.vector_store_ready:
        st.warning("Please upload and process documents first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Retrieving context and generating answer..."):
                embeddings = get_embedding_model()
                vector_store = load_vector_store(embeddings, VECTOR_PATH)
                retriever = get_retriever(vector_store, k=4)
                prompt = get_prompt()
                llm = get_llm()

                answer, retrieved_docs = answer_question(query, retriever, llm, prompt)

                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer,
                    "sources": retrieved_docs
                })

        except Exception as e:
            st.error(f"Error while generating answer: {e}")

if st.session_state.chat_history:
    st.subheader("Chat History")

    for idx, item in enumerate(reversed(st.session_state.chat_history), start=1):
        st.markdown(f"### Question {idx}")
        st.write(item["question"])

        st.markdown("**Answer:**")
        st.write(item["answer"])

        with st.expander("Retrieved Context / Sources"):
            for i, doc in enumerate(item["sources"], start=1):
                st.markdown(f"**Chunk {i}**")
                page_num = doc.metadata.get("page", "N/A")
                source_file = doc.metadata.get("source", "Unknown source")
                st.write(f"Source: {source_file}")
                st.write(f"Page: {page_num}")
                st.write(doc.page_content[:1200])
                st.write("---")