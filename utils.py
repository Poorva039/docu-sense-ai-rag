import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def load_pdf_documents(file_paths: list[str]):
    """
    Load PDF documents from a list of file paths.
    Returns a list of LangChain Document objects.
    """
    documents = []

    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)

    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def get_embedding_model():
    """
    Return a Hugging Face embedding model.
    Free and good enough for resume-worthy RAG projects.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store(chunks, embeddings):
    """
    Create a FAISS vector store from document chunks.
    """
    if not chunks:
        raise ValueError("No chunks were provided to create the vector store.")

    return FAISS.from_documents(chunks, embeddings)


def save_vector_store(vector_store, path: str = "vector_store/faiss_index"):
    """
    Save FAISS vector store locally.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vector_store.save_local(path)


def load_vector_store(embeddings, path: str = "vector_store/faiss_index"):
    """
    Load FAISS vector store from local disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Vector store not found at '{path}'. Please process documents first."
        )

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_retriever(vector_store, k: int = 4):
    """
    Convert vector store into a retriever.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})


def get_prompt():
    """
    Prompt template to keep answers grounded in retrieved context.
    """
    template = """
You are DocuSense AI, an intelligent document question answering assistant.

Answer the user's question only using the provided context.
If the answer is not available in the context, say:
"I could not find that information in the uploaded documents."

Rules:
- Be clear and concise
- Do not make up information
- Use only the retrieved context
- If possible, summarize in simple language

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def get_llm():
    """
    Return OpenAI chat model.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing in your .env file.")

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )


def format_retrieved_context(retrieved_docs):
    """
    Join retrieved docs into a single context string.
    """
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


def answer_question(query: str, retriever, llm, prompt):
    """
    End-to-end RAG answer generation.
    Returns:
      - final answer
      - retrieved documents
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    retrieved_docs = retriever.invoke(query)
    context = format_retrieved_context(retrieved_docs)

    formatted_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(formatted_prompt)

    return response.content, retrieved_docs