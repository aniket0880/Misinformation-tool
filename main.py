import asyncio
import os
import streamlit as st

# 🔧 Fix for Streamlit + asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

# 🔑 Hardcoded API Key (replace with your real key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDrvRfUAwRNU55bFmkU8Ihl1wNKJ0jTEYc"

from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate


# Define embeddings using Gemini
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="QUESTION_ANSWERING"
)
vector_store = InMemoryVectorStore(embeddings)

# Define Gemini LLM
model = GoogleGenerativeAI(model="gemini-2.0-flash")

# Prompt template
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

def load_page(url):
    loader = SeleniumURLLoader(urls=[url])
    return loader.load()

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return splitter.split_documents(documents)

def index_docs(docs):
    vector_store.add_documents(docs)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Streamlit UI
st.title("AI Crawler (Gemini)")
url = st.text_input("Enter URL:")

if url:
    documents = load_page(url)
    chunks = split_text(documents)
    index_docs(chunks)

question = st.chat_input()
if question:
    st.chat_message("user").write(question)
    retrieved = retrieve_docs(question)
    context = "\n\n".join(doc.page_content for doc in retrieved)
    answer = answer_question(question, context)
    st.chat_message("assistant").write(answer)
