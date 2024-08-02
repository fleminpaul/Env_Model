from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import WebBaseLoader

app = Flask(__name__)

html_doc_path="C:/Users/Flemin/RAGDemo/RawData"

folder_path = "db"

cached_llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

# raw_prompt = PromptTemplate.from_template(
#     """ 
#     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#     You are good support assistant. you are capable of answering any queries based on India History, Geography ,Economy,Religion,Language.
#     Any queries outside the context of India should be ignored and should be replied as just "Go to Hell". you should not hallucinate and reply.
#     you should remove any junk characters or unwanted spaces or new lines.
#     If you do not have an answer from the provided information say so. Strictly follow the Guardrails. Donot answer anything Generic or outside the context .Strictly follow the rules<|eot_id|>
#     <|start_header_id|>user<|end_header_id|>
#     What is CAGR?<|eot_id|>
#     <|start_header_id|>assistant<|end_header_id|> 
#     Go to Hell, I can answer only about India<|eot_id|>
#     <|start_header_id|>user<|end_header_id|>
#     Who is Narendara Modi?<|eot_id|>
#     <|start_header_id|>assistant<|end_header_id|> 
#     He is the currentPrimeminister of India
#     <|start_header_id|>user<|end_header_id|>
#     {input}
#     Context: {context}
#     Answer:
#     <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """
# )

raw_prompt = PromptTemplate.from_template(
    """ 
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are good support assistant. you are capable of answering any queries related to Envestnet.
    Any queries outside the context of Envestnet should be ignored and should be replied as just "I don't know what you are talking about". you should not hallucinate and reply.
    you should remove any junk characters or unwanted spaces or new lines.
    If you do not have an answer from the provided information say so. Strictly follow the Guardrails. Donot answer anything Generic or outside the context .Strictly follow the rules<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    What is CAGR?<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|> 
    I don't know what you are talking about<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Who is Narendara Modi?<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|> 
    I don't know what you are talking about
    <|start_header_id|>user<|end_header_id|>
    {input}
    Context: {context}
    Answer:
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
)

@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)

    print(response)

    response_answer = {"answer": response}
    return response_answer


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    print("request :{}",request)
    json_content = request.json
    print("request_json :{}",json_content)
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 11,
            "score_threshold": 0.75,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer

# def html_loader():
#     file_name=html_doc_path+"/India - Wikipedia.html"
#     loader = BSHTMLLoader(file_name, open_encoding='utf-8',)
#     docs = loader.load_and_split()
#     print(f"docs len={len(docs)}")
#     chunks = text_splitter.split_documents(docs)
#     print(f"chunks len={len(chunks)}")
#     vector_store = Chroma.from_documents(
#         documents=chunks, embedding=embedding, persist_directory=folder_path
#     )
#     vector_store.persist()
#     response = {
#         "status": "Successfully Uploaded",
#         "filename": file_name,
#         "doc_len": len(docs),
#         "chunks": len(chunks),
#     }

def webbase_loader():
    file_name="https://www.envestnet.com/"
    loader = WebBaseLoader(file_name)
    loader.requests_per_second = 1
    docs = loader.aload()
    print(f"docs len={len(docs)}")
    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )
    vector_store.persist()
    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }

def start_app():
    webbase_loader()
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
