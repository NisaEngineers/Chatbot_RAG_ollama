import os
import fitz  # PyMuPDF: install via `pip install PyMuPDF`
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Define the PDF file path.
pdf_path = "monaarch.pdf"

# Open and extract text from the PDF.
doc = fitz.open(pdf_path)
pdf_text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    pdf_text += page.get_text()

# Initialize the free Ollama embedding model.
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Specify a new persistence directory to ensure embeddings match.
db_location = "./chroma_langchain_db_monaarch"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    # Split the PDF text into manageable chunks (using double newlines as delimiters).
    chunks = pdf_text.split("\n\n")
    for i, chunk in enumerate(chunks):
        cleaned_chunk = chunk.strip()
        if cleaned_chunk:
            document = Document(
                page_content=cleaned_chunk,
                metadata={"Source": "monaarch.pdf", "Chunk": i + 1},
                id=str(i)
            )
            documents.append(document)
            ids.append(str(i))

# Create (or load) the vector store using Chroma.
vector_store = Chroma(
    collection_name="monaarch",
    persist_directory=db_location,
    embedding_function=embeddings
)

# On the first run (or with a new persistence directory), add our documents.
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Create a retriever that fetches the top 5 relevant context chunks.
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
