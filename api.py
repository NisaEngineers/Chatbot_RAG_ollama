from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Initialize the free Ollama language model.
model = OllamaLLM(model="llama3.2")
# model = OllamaLLM(model="phi3")

# Define the prompt template tailored for Monaarch technical and business support.
template = """
You are a customer service agent representing Monaarch. Your role is to provide friendly, professional, and empathetic support to Monaarch customers. Utilize the documentation context below to deliver accurate, clear, and helpful responses. Response should be short.

Documentation context:
{reviews}

Customer question:
{question}
"""


# Create the prompt template and chain it with the model.
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Initialize the FastAPI app.
app = FastAPI()

# Add CORS middleware to allow cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this list with your client's domains for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a pydantic model for request validation.
class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(question: Question):
    # Ensure that the question isn't empty.
    if not question.question.strip():
        raise HTTPException(status_code=400, detail="Empty question provided.")

    try:
        # Retrieve relevant context from the vector store.
        reviews = retriever.invoke(question.question)
        # Generate the chatbot's response.
        result = chain.invoke({"reviews": reviews, "question": question.question})
    except Exception as e:
        # Optionally, log the error and return a server error response.
        raise HTTPException(status_code=500, detail=str(e))

    return {"result": result}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
