# DocuSense AI – Intelligent Document Question Answering System

DocuSense AI is a simple application that allows users to upload PDF documents and ask questions based on the document content. The system processes the documents, converts text into embeddings, stores them in a FAISS vector database, and retrieves the most relevant information to generate answers.

This project demonstrates how Retrieval-Augmented Generation (RAG) works using vector search and large language models.

---

## Features

- Upload PDF documents
- Extract and process document text
- Convert text into embeddings using Sentence Transformers
- Store embeddings in a FAISS vector database
- Retrieve relevant document sections for a given question
- Generate answers using a language model
- Simple web interface built with Streamlit

---

## Technologies Used

- Python
- LangChain
- FAISS (Vector Database)
- Sentence Transformers
- OpenAI API
- Streamlit

---

## Project Structure

```
DocuSenseAI
│
├── app.py
├── api.py
├── utils.py
├── requirements.txt
├── .env
├── data/
└── vector_store/
```

---

## Installation

Clone the repository

```
git clone https://github.com/yourusername/docu-sense-ai-rag.git
cd docu-sense-ai-rag
```

Create a virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file and add your OpenAI API key.

```
OPENAI_API_KEY=your_api_key_here
```

---

## Run the Application

Start the Streamlit app

```
streamlit run app.py
```

The application will open in your browser.

---

## How It Works

1. Upload a PDF document
2. The system extracts text from the document
3. Text is split into smaller chunks
4. Each chunk is converted into embeddings
5. Embeddings are stored in a FAISS vector database
6. When a user asks a question, relevant chunks are retrieved
7. The language model generates an answer using the retrieved context
