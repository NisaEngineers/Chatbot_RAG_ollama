import os
import fitz  # PyMuPDF: install via `pip install PyMuPDF`
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Define the path to the Monaarch PDF file.
pdf_path = "monaarch.pdf"

# Open and extract text from the PDF.
doc = fitz.open(pdf_path)
pdf_text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    pdf_text += page.get_text()

# Initialize the Hugging Face embedding model.
# We'll use a Sentence Transformer model. Adjust the model name as needed.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Specify a new persistence directory for the vector store.
db_location = "./chroma_langchain_db_hf"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    # Split the PDF text into chunks (using double newlines).
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

# Create the vector store using Chroma.
vector_store = Chroma(
    collection_name="monaarch",
    persist_directory=db_location,
    embedding_function=embeddings
)

# On first run (or when using a new persistence directory), add documents.
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Create a retriever that returns the top 5 matching context chunks.
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
