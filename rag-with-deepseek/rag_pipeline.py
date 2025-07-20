import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Ensure the GROQ_API_KEY is available
if "GROQ_API_KEY" not in os.environ:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in a .env file.")

# Step 1: Setup LLM (Use DeepSeek R1 with Groq)
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Step 2: Load the existing vector database
FAISS_DB_PATH = "vectorstore/db_faiss"
ollama_model_name = "deepseek-r1:1.5b"

# We need the same embedding function used to create the DB
embeddings = OllamaEmbeddings(model=ollama_model_name)
faiss_db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)

def retrieve_docs(query):
    """Retrieve relevant documents from the vector database."""
    results = faiss_db.similarity_search(query)
    return results

def get_context(documents):
    """Extract page content from documents to form a context string."""
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# Step 3: Generate response using the LLM and retrieved documents
custom_prompt_template = """
Use the piece of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Dont provide anything out of the given context.
User's question: {question}
Context from documents: {context}
Answer:
"""

def answer_query(documents, model, query):
    """Generate an answer to a query based on provided documents and a model."""
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    response = chain.invoke({"question": query, "context": context})
    return response.content # Extract the string content from the AIMessage


#"""Main function to run the RAG pipeline."""

# question = "If a government forbids the right to assemble peacefully which article are violated and why?"
# retrieved_docs = retrieve_docs(question)
# response = answer_query(retrieved_docs, model=llm, query=question)
# print("\nAI Lawyer:", response)