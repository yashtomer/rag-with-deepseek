from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

pdfs_directory = "pdfs/"

def upload_pdf(file):
    with open(f"{pdfs_directory}{file.name}", "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

file_path = "eng.pdf"
document = load_pdf(file_path)
#print(len(document))


#Step 2: Create chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    return chunks

text_chunks = create_chunks(document)
#print(len(text_chunks))


#Steps 3: Create embeddings model (use deepseek r1 with ollama)
ollama_model_name = "deepseek-r1:1.5b"
def create_embeddings(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings


#Steps 4: Index Documents Store emmbeddings in vector database
FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db=FAISS.from_documents(text_chunks, create_embeddings(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)

