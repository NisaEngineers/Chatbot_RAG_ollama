from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from vector_ol import retriever

# Build a Hugging Face Transformers pipeline for text generation.
# Here we use "google/flan-t5-large", but you can swap in any other supported model.
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=512  # adjust based on your requirements
)

# Wrap the Transformers pipeline in LangChain's HuggingFacePipeline.
llm = HuggingFacePipeline(pipeline=pipe)

# Define a prompt template for Monaarch business and technical support.
template = """
You are a chatbot specialized in providing technical and business support for Monaarch.
Utilize the context from the Monaarch documentation provided below to answer accurately.

Retrieved context:
{reviews}

User question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm  # Combine the prompt with the language model

while True:
    question = input("Ask your question (q to quit): ")
    if question.strip().lower() == "q":
        break

    # Retrieve context from the vector store.
    reviews = retriever.invoke(question)

    # Generate the answer using the chain.
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
