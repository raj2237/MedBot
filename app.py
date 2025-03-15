from flask import Flask, render_template , request
import jsonify
from src.helper import download_hugface_embedding
from langchain_pinecone import PineconeVectorStore
import os
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *

app =Flask(__name__)
load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY


embeddings = download_hugface_embedding()
index_name = "medbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type= "similarity" , search_kwargs = {"k": 3})

llm = ChatGroq(
    temperature= 0.6,
    model = "llama3-70b-8192",
    api_key= "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"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm , prompt)
rag_chain = create_retrieval_chain(retriever , question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods = ["GET","POST"])
def chat():
    print("hi")
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response:", response["answer"])
    return str(response["answer"])
 

if __name__ == "__main__":
    app.run(host = "0.0.0.0",port = 8080, debug=True)