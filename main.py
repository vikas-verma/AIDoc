# from fastapi import FastAPI, UploadFile, File
# from docx import Document
# from transformers import pipeline
# from elasticsearch import Elasticsearch
# import uvicorn
# from pydantic import BaseModel
# import faiss
# import numpy as np
# from typing import List

# #from embedding import generate_embeddings  # Import your function

# from embedding import split_text, generate_embeddings, store_in_faiss, search_faiss

# app = FastAPI()
# chunks = []

# # Load FAISS index
# index = faiss.read_index("faiss_index.idx")

# # Connect to Elasticsearch
# #es = Elasticsearch("http://localhost:9200")

# # Load Hugging Face summarization model
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# @app.post("/upload/")
# async def upload_file(file: UploadFile = File(...)):
#     if not file.filename.endswith(".docx"):
#         return {"error": "Only .docx files are supported"}

#     # Read the Word document
#     doc = Document(file.file)
#     full_text = "\n".join([para.text for para in doc.paragraphs])

#     # Summarize the content
#     summary = summarizer(full_text, max_length=100, min_length=30, do_sample=False)
#     summarized_text = summary[0]["summary_text"]

#     # Store document in Elasticsearch
#     doc_id = file.filename  # Using filename as document ID
#     #es.index(index="aipocdoc", id=doc_id, document={"text": full_text, "summary": summarized_text})

#     return {"summary": summarized_text, "message": "Stored in Elasticsearch"}

# @app.get("/")
# def read_root():
#     return {"message": "FastAPI is working!"}

# class TextRequest(BaseModel):
#     #text: str  # Ensure text is expected
#     documents:List[str] # Expecting a list of text inputs


# # @app.post("/embed/") #for single text
# # async def embed_text(request: TextRequest):
# #     """Receives text, generates embeddings, and stores them in FAISS"""
# #     text_chunks = split_text(request.text)
# #     embeddings = generate_embeddings(text_chunks)
# #     store_in_faiss(embeddings)
# #     return {"message": "Embeddings generated and stored in FAISS!"}


# @app.post("/embed/")
# async def embed_text(request: TextRequest):
#     """Receives multiple text inputs, generates embeddings, and stores them in FAISS"""
#     all_embeddings = []
    
#     for text in request.documents:  # Iterate over each text input
#         text_chunks = split_text(text)
#         embeddings = generate_embeddings(text_chunks)
#         all_embeddings.extend(embeddings)  # Collect embeddings for all texts

#     store_in_faiss(all_embeddings)  # Store all embeddings in FAISS
#     return {"message": "Embeddings generated and stored in FAISS!"}

# @app.post("/search")
# # async def search(request: TextRequest):
# #     query_vector = np.random.rand(1, 384).astype("float32")  # Dummy vector, replace with actual embedding logic
# #     _, indices = index.search(query_vector, 3)
# #     return {"text": request.query, "matches": indices.tolist()[0]}

# #     return {"text": request.query, "matches": indexes[0].tolist()}

# async def search(request: TextRequest):
#     query_vector = np.random.rand(1, 384).astype("float32")  # Replace with actual embeddings
#     _, indices = index.search(query_vector, 3)
#     return {"text": request.text, "matches": indices.tolist()[0]}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, UploadFile, File
from docx import Document
from transformers import pipeline
import uvicorn
from pydantic import BaseModel
import faiss
import numpy as np
from typing import List

from embedding import split_text, generate_embeddings, store_in_faiss, search_faiss

app = FastAPI()
chunks = []

# Load FAISS index
index = faiss.read_index("faiss_index.idx")

# Load Hugging Face summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class TextRequest(BaseModel):
    documents: List[str]  # Expecting a list of text inputs

class QueryModel(BaseModel):
    query: str
    top_k: int = 5

@app.post("/upload/", tags=["Upload"])
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".docx"):
        return {"error": "Only .docx files are supported"}

    # Read the Word document
    doc = Document(file.file)
    full_text = "\n".join([para.text for para in doc.paragraphs])

    # Summarize the content
    summary = summarizer(full_text, max_length=100, min_length=30, do_sample=False)
    summarized_text = summary[0]["summary_text"]

    return {"summary": summarized_text, "message": "Document processed successfully"}

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "FastAPI is working!"}

@app.post("/embed/", tags=["Embedding"])
async def embed_text(request: TextRequest):
    """Receives multiple text inputs, generates embeddings, and stores them in FAISS"""
    all_embeddings = []
    all_texts = []  # Initialize list to store text chunks

    for text in request.documents:  # Iterate over each text input
        text_chunks = split_text(text)  # Split text into chunks
        embeddings = generate_embeddings(text_chunks)  # Generate embeddings for chunks
        all_embeddings.extend(embeddings)  # Collect embeddings for all texts
        all_texts.extend(text_chunks)  # Collect corresponding text chunks

    store_in_faiss(all_embeddings, all_texts)  # Store both embeddings & texts in FAISS
    return {"message": "Embeddings generated and stored in FAISS!"}


@app.post("/store/", tags=["Embedding"])
async def store_documents(documents: List[str]):
    """Store documents in FAISS after embedding them."""
    embeddings = generate_embeddings(documents)
    store_in_faiss(embeddings)
    return {"message": f"Stored {len(documents)} documents in FAISS."}

@app.post("/search/", tags=["Search"])
async def search_documents(query: QueryModel):
    """Search documents using FAISS index."""
    results = search_faiss(query.query, query.top_k)
    return {"query": query.query, "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
