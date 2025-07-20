#Project setup

sudo apt install pipenv pipenv install streamlit pipenv shell

pipenv install langchain langchain_community langchain_ollama langchain_core langchain_groq faiss-cpu pdfplumber

ollama pull deepseek-r1:1.5b
