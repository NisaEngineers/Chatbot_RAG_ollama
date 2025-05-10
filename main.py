from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Initialize the free Ollama language model.
model = OllamaLLM(model="llama3.2")
#model = OllamaLLM(model="phi3")

# Define a prompt tailored for Monaarch technical and business support.
template = """
You are a chatbot specialized in providing technical and business support for Monaarch.
Utilize the context from the Monaarch documentation provided below to answer accurately.

Retrieved context:
{reviews}

User question:
{question}
"""

# Create the prompt template.
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model  # Chain the prompt with the LLM

while True:
    question = input("Ask your question (q to quit): ")
    if question.strip().lower() == "q":
        break

    # Retrieve relevant context from the vector store.
    reviews = retriever.invoke(question)

    # Generate and display the chatbot's answer.
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
