# This script loads the knowledge base, converts it into embeddings, and stores it in FAISS.

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
# Load data from knowledge base
loader = TextLoader(r"data\knowledge.txt")
docs = loader.load()
# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# Generate embedding using GeminiAPI
embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key = api_key
)

# Store embeddings in FAISS
vector_db = FAISS.from_documents(documents, embeddings)

# Retriever for querying the database
retriever = vector_db.as_retriever(search_kwargs={"k": 5})


