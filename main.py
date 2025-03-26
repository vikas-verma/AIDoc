# # from fastapi import FastAPI, UploadFile, File
# # from docx import Document
# # from transformers import pipeline
# # from elasticsearch import Elasticsearch
# # import uvicorn
# # from pydantic import BaseModel
# # import faiss
# # import numpy as np
# # from typing import List

# # #from embedding import generate_embeddings  # Import your function

# # from embedding import split_text, generate_embeddings, store_in_faiss, search_faiss

# # app = FastAPI()
# # chunks = []

# # # Load FAISS index
# # index = faiss.read_index("faiss_index.idx")

# # # Connect to Elasticsearch
# # #es = Elasticsearch("http://localhost:9200")

# # # Load Hugging Face summarization model
# # summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# # @app.post("/upload/")
# # async def upload_file(file: UploadFile = File(...)):
# #     if not file.filename.endswith(".docx"):
# #         return {"error": "Only .docx files are supported"}

# #     # Read the Word document
# #     doc = Document(file.file)
# #     full_text = "\n".join([para.text for para in doc.paragraphs])

# #     # Summarize the content
# #     summary = summarizer(full_text, max_length=100, min_length=30, do_sample=False)
# #     summarized_text = summary[0]["summary_text"]

# #     # Store document in Elasticsearch
# #     doc_id = file.filename  # Using filename as document ID
# #     #es.index(index="aipocdoc", id=doc_id, document={"text": full_text, "summary": summarized_text})

# #     return {"summary": summarized_text, "message": "Stored in Elasticsearch"}

# # @app.get("/")
# # def read_root():
# #     return {"message": "FastAPI is working!"}

# # class TextRequest(BaseModel):
# #     #text: str  # Ensure text is expected
# #     documents:List[str] # Expecting a list of text inputs


# # # @app.post("/embed/") #for single text
# # # async def embed_text(request: TextRequest):
# # #     """Receives text, generates embeddings, and stores them in FAISS"""
# # #     text_chunks = split_text(request.text)
# # #     embeddings = generate_embeddings(text_chunks)
# # #     store_in_faiss(embeddings)
# # #     return {"message": "Embeddings generated and stored in FAISS!"}


# # @app.post("/embed/")
# # async def embed_text(request: TextRequest):
# #     """Receives multiple text inputs, generates embeddings, and stores them in FAISS"""
# #     all_embeddings = []
    
# #     for text in request.documents:  # Iterate over each text input
# #         text_chunks = split_text(text)
# #         embeddings = generate_embeddings(text_chunks)
# #         all_embeddings.extend(embeddings)  # Collect embeddings for all texts

# #     store_in_faiss(all_embeddings)  # Store all embeddings in FAISS
# #     return {"message": "Embeddings generated and stored in FAISS!"}

# # @app.post("/search")
# # # async def search(request: TextRequest):
# # #     query_vector = np.random.rand(1, 384).astype("float32")  # Dummy vector, replace with actual embedding logic
# # #     _, indices = index.search(query_vector, 3)
# # #     return {"text": request.query, "matches": indices.tolist()[0]}

# # #     return {"text": request.query, "matches": indexes[0].tolist()}

# # async def search(request: TextRequest):
# #     query_vector = np.random.rand(1, 384).astype("float32")  # Replace with actual embeddings
# #     _, indices = index.search(query_vector, 3)
# #     return {"text": request.text, "matches": indices.tolist()[0]}

# # if __name__ == "__main__":
# #     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, UploadFile, File
# from docx import Document
# from transformers import pipeline
# import uvicorn
# from pydantic import BaseModel
# import faiss
# import numpy as np
# from typing import List

# from embedding import split_text, generate_embeddings, store_in_faiss, search_faiss

# from fastapi import FastAPI, Query
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# from gpt4all import GPT4All





# app = FastAPI()
# chunks = []

# # Load FAISS index
# index = faiss.read_index("faiss_index.idx")

# # Load Hugging Face summarization model
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# # Load embedding model (AI-powered)
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # # Load LLM (GPT4All - AI Model for Answering)
# # llm = GPT4All("gpt4all-falcon-q4_0")

# model_path = r"C:\Users\vikas\ai-doc-reader\local_models\mistral-7b-instruct-v0.2-q4_k_m.gguf"

# # Load the model
# llm = GPT4All(model_path, allow_download=False)  # Prevent download

# # Example documents
# docs = [
#     "Artificial intelligence is the simulation of human intelligence by machines.",
#     "FastAPI is a modern, high-performance web framework for Python.",
#     "Elasticsearch is used for full-text search and analytics.",
# ]

# # Convert documents into embeddings
# doc_embeddings = embedding_model.encode(docs)

# # Store embeddings in FAISS index
# index = faiss.IndexFlatL2(doc_embeddings.shape[1])
# index.add(np.array(doc_embeddings))


# class TextRequest(BaseModel):
#     documents: List[str]  # Expecting a list of text inputs

# class QueryModel(BaseModel):
#     query: str
#     top_k: int = 5

# @app.post("/upload/", tags=["Upload"])
# async def upload_file(file: UploadFile = File(...)):
#     if not file.filename.endswith(".docx"):
#         return {"error": "Only .docx files are supported"}

#     # Read the Word document
#     doc = Document(file.file)
#     full_text = "\n".join([para.text for para in doc.paragraphs])

#     # Summarize the content
#     summary = summarizer(full_text, max_length=100, min_length=30, do_sample=False)
#     summarized_text = summary[0]["summary_text"]

#     return {"summary": summarized_text, "message": "Document processed successfully"}

# @app.get("/", tags=["Root"])
# def read_root():
#     return {"message": "FastAPI is working!"}

# @app.post("/embed/", tags=["Embedding"])
# async def embed_text(request: TextRequest):
#     """Receives multiple text inputs, generates embeddings, and stores them in FAISS"""
#     all_embeddings = []
#     all_texts = []  # Initialize list to store text chunks

#     for text in request.documents:  # Iterate over each text input
#         text_chunks = split_text(text)  # Split text into chunks
#         embeddings = generate_embeddings(text_chunks)  # Generate embeddings for chunks
#         all_embeddings.extend(embeddings)  # Collect embeddings for all texts
#         all_texts.extend(text_chunks)  # Collect corresponding text chunks

#     store_in_faiss(all_embeddings, all_texts)  # Store both embeddings & texts in FAISS
#     return {"message": "Embeddings generated and stored in FAISS!"}


# @app.post("/store/", tags=["Embedding"])
# async def store_documents(documents: List[str]):
#     """Store documents in FAISS after embedding them."""
#     embeddings = generate_embeddings(documents)
#     store_in_faiss(embeddings)
#     return {"message": f"Stored {len(documents)} documents in FAISS."}

# @app.post("/search/", tags=["Search"])
# async def search_documents(query: QueryModel):
#     """Search documents using FAISS index."""
#     results = search_faiss(query.query, query.top_k)
#     return {"query": query.query, "results": results}

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/process_query/")
# async def process_query(request: QueryRequest):
#     user_query = request.query
#     # Implement your query processing logic here
#     return {"response": f"Received query: {user_query}"}

# # @app.post("/query/")
# # async def process_query(request: QueryRequest):
# #     user_query = request.query
# #     query_embedding = get_query_embedding(user_query)
# #     relevant_docs = search_documents(query_embedding)
# #     response = generate_summary(relevant_docs)
# #     return {"response": response}






# @app.get("/query")
# def query_ai(user_query: str = Query(..., title="User Query")):
#     # Convert user query into embedding
#     query_embedding = embedding_model.encode([user_query])

#     # Search for the best matching document
#     _, I = index.search(np.array(query_embedding), 1)
#     best_match = docs[I[0][0]]

#     # Use AI (GPT4All) to generate an answer
#     response = llm.generate(f"Answer this based on the retrieved document: {best_match}")

#     return {"user_query": user_query, "retrieved_info": best_match, "ai_answer": response}



# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)



from fastapi import FastAPI, UploadFile, File, Query
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
import uvicorn
import faiss
import numpy as np
from docx import Document
from typing import List
import os

app = FastAPI()

# Initialize AI models dynamically
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Can be changed dynamically
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load FAISS index dynamically (if exists, else create a new one)
index_path = "faiss_index.idx"
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = None  # Will be initialized dynamically

# Load GPT4All LLM dynamically
model_path = os.getenv("MODEL_PATH", "local_models/tinyllama-1.1b-chat-v1.0-q4_k_m.gguf")
llm = GPT4All(model_path, allow_download=False)

# Store document texts (to retrieve later by index)
doc_texts = []

class QueryModel(BaseModel):
    query: str
    top_k: int = 5

# @app.post("/upload/")
# async def upload_file(file: UploadFile = File(...)):
#     """Uploads a .docx file, extracts text, generates embeddings, and stores them dynamically in FAISS."""
#     global index, doc_texts
    
#     if not file.filename.endswith(".docx"):
#         return {"error": "Only .docx files are supported"}

#     doc = Document(file.file)
#     full_text = "\n".join([para.text for para in doc.paragraphs])
#     doc_texts.append(full_text)

#     # Summarize content
#     summary = summarizer(full_text, max_length=100, min_length=30, do_sample=False)
#     summarized_text = summary[0]["summary_text"]
    
#     # Generate embeddings
#     embeddings = embedding_model.encode([full_text])
    
#     # Initialize FAISS index if not already
#     if index is None:
#         index = faiss.IndexFlatL2(embeddings.shape[1])
    
#     index.add(np.array(embeddings))
#     faiss.write_index(index, index_path)

#     return {"summary": summarized_text, "message": "Document stored successfully!"}

@app.post("/reset_index", tags=["Debug"])
def reset_index():
    global index, doc_texts
    # Clear the in-memory document list
    doc_texts = []
    # Remove the FAISS index file if it exists
    index_file = "faiss_index.idx"
    if os.path.exists(index_file):
        os.remove(index_file)
    # Reinitialize the index to None (it will be created upon the next upload)
    index = None
    return {"message": "FAISS index and document storage have been reset."}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global index, doc_texts  # Make sure these globals are accessible
    if not file.filename.endswith(".docx"):
        return {"error": "Only .docx files are supported"}

    doc = Document(file.file)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    
    # Append the full text or its chunks to doc_texts
    doc_texts.append(full_text)
    
    # Summarize and generate embeddings
    summary = summarizer(full_text, max_length=100, min_length=30, do_sample=False)
    summarized_text = summary[0]["summary_text"]
    
    embeddings = embedding_model.encode([full_text])
    
    # Initialize FAISS if needed
    if index is None:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    
    index.add(np.array(embeddings))
    faiss.write_index(index, "faiss_index.idx")
    
    return {"summary": summarized_text, "message": "Document stored successfully!"}


@app.post("/search/")
async def search_documents(query: QueryModel):
    """Searches for the most relevant document embeddings from FAISS."""
    if index is None or len(doc_texts) == 0:
        return {"error": "No documents found in the FAISS index."}
    
    query_embedding = embedding_model.encode([query.query])
    distances, indices = index.search(np.array(query_embedding), query.top_k)
    
    results = [doc_texts[idx] for idx in indices[0] if idx < len(doc_texts)]
    return {"query": query.query, "results": results}

@app.get("/query")
def query_ai(user_query: str = Query(..., title="User Query")):
    """Queries the LLM based on retrieved document embeddings."""
    if index is None or len(doc_texts) == 0:
        return {"error": "No stored documents for AI to reference."}
    
    query_embedding = embedding_model.encode([user_query])
    distances, indices = index.search(np.array(query_embedding), 1)
    best_match = doc_texts[indices[0][0]] if indices[0][0] < len(doc_texts) else "No relevant document found."
    
    response = llm.generate(f"Answer this based on the retrieved document: {best_match}")
    return {"user_query": user_query, "retrieved_info": best_match, "ai_answer": response}

@app.get("/faiss-docs", tags=["Debug"])
def get_faiss_docs():
    if index is None:
        return {"error": "FAISS index is not created yet."}
    return {"faiss_count": index.ntotal, "documents": doc_texts}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
