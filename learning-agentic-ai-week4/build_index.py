import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Setup embedding (The 'Translator')
# This turns the text into numbers so the AI "search" by meaning, not just keywords
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004", # Google's efficient embedding model
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

# Load PDF
pdf_file = "my_data.pdf"
if not os.path.exists(pdf_file):
    print("Error: Please put a file named 'my_data.pdf' in this folder!")
    exit()

print("Loading PDF...")
loader = PyPDFLoader(pdf_file)
docs = loader.load()

# Split text (Chunking)
# We can't feed a whole book to the AI. We split into paragraphs
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Each chunk is ~1000 characters
    chunk_overlap=200 # Overlap helps keep context between chunks
)

splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} chunks")

# Create vector store
print("Building Vector Database (This might take a moment)...")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# Save it
# We save this to a folder so we don't have to rebuild it every time.
vectorstore.save_local("faiss_index_react")
print("Success! Index saved to folder 'faiss_index_react'")
