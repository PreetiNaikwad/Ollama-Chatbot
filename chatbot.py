import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

app = Flask(__name__)
CORS(app)

VECTOR_DB_PATH = "./vectorstore"

# Global objects for fast access
vectorstore = None
llm = None
qa_chain = None  # 💡 preload the chain

def load_documents():
    loader = TextLoader("documents/sample.txt")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def create_vectorstore(docs):
    embeddings = OllamaEmbeddings(model="llama2")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    vectordb.persist()
    return vectordb

def load_existing_vectorstore():
    embeddings = OllamaEmbeddings(model="llama2")
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

def initialize_system():
    global vectorstore, llm, qa_chain

    # Initialize LLM once
    llm = OllamaLLM(model="llama2")

    # Load or create vector store
    if os.path.exists(VECTOR_DB_PATH):
        vectorstore = load_existing_vectorstore()
    else:
        docs = load_documents()
        vectorstore = create_vectorstore(docs)

    # 💡 Build QA chain only once and reuse
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    try:
        result = qa_chain.invoke({"query": question})
        return jsonify({"response": result.get("result", "No answer found.")})
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    initialize_system()
    app.run(debug=True)
