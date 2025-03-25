# import faiss
# import numpy as np
# import os
# from sentence_transformers import SentenceTransformer

# # Load the sentence transformer model
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Example text chunks
# documents = [
#     "This is the first document.",
#     "This is the second document.",
#     "FAISS is great for fast document searches.",
# ]

# # Convert documents to embeddings
# embeddings = model.encode(documents)

# # Convert to FAISS format
# dimension = embeddings.shape[1]  # Get embedding dimension
# index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance index
# index.add(np.array(embeddings, dtype=np.float32))  # Add vectors to the index

# # Save FAISS index
# faiss.write_index(index, "faiss_index.idx")
# print("FAISS index created and saved as 'faiss_index.idx'")

# # Check if FAISS index file exists
# faiss_index_file = "faiss_index.idx"

# # Check if FAISS index file exists
# if os.path.exists(faiss_index_file):
#     index = faiss.read_index(faiss_index_file)
#     print("FAISS index loaded successfully.")
# else:
#     print(f"FAISS index file '{faiss_index_file}' not found. Please create it first.")


# # Load the model
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# def split_text(text, chunk_size=512):
#     """Splits a long text into smaller chunks"""
#     return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

# def generate_embeddings(text_chunks):
#     """Generates embeddings for given text chunks"""
#     return model.encode(text_chunks)

# def store_in_faiss(embeddings, index_file="faiss_index.idx"):
#     """Stores embeddings in FAISS index"""
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array(embeddings, dtype=np.float32))
#     faiss.write_index(index, index_file)
#     print(f"FAISS index saved as {index_file}")

# def search_faiss(query, index_file="faiss_index.idx", top_k=5):
#     """Searches FAISS index for the most similar vectors"""
#     index = faiss.read_index(index_file)
#     query_embedding = model.encode([query])
#     distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
#     return indices

# # Ensure this module only runs these steps when executed directly
# if __name__ == "__main__":
#     print("Embedding module loaded successfully.")






# # Load FAISS index
# #INDEX_FILE = "faiss_index.idx"
# #model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Load FAISS index
# def load_faiss_index():
#     index = faiss.read_index(faiss_index_file)
#     return index

# faiss_index = load_faiss_index()



# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Sample documents (Replace this with actual document loading logic)
# documents = [
#     "This is the first document.",
#     "Second document with some text.",
#     "Another important document about AI.",
# ]

# # Generate embeddings
# embeddings = model.encode(documents)

# # Create FAISS index
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(np.array(embeddings))

# # Save FAISS index
# faiss.write_index(index, "faiss_index.idx")

# # Also save original texts
# with open("documents.txt", "w", encoding="utf-8") as f:
#     for doc in documents:
#         f.write(doc + "\n")

# print("FAISS index and document texts saved!")



# def search_faiss(query, top_k=3):
#     # Load FAISS index
#     index = faiss.read_index("faiss_index.idx")

#     # Load documents from file
#     with open("documents.txt", "r", encoding="utf-8") as f:
#         documents = f.readlines()

#     # Generate query embedding
#     query_embedding = model.encode([query])

#     # Search FAISS for the closest matches
#     distances, indices = index.search(np.array(query_embedding), top_k)

#     # Fetch the actual documents
#     results = [documents[i].strip() for i in indices[0]]

#     return results


import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# FAISS Index File
FAISS_INDEX_FILE = "faiss_index.idx"
DOCUMENTS_FILE = "documents.txt"


def split_text(text, chunk_size=512):
    """Splits a long text into smaller chunks."""
    return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]


def generate_embeddings(texts):
    """Generates embeddings for a list of texts."""
    if not isinstance(texts, list):
        raise ValueError("Input should be a list of text strings.")
    return np.array(model.encode(texts), dtype=np.float32)

def store_in_faiss(embeddings, texts):
    """Stores embeddings and documents in FAISS index."""
    embeddings = np.array(embeddings, dtype=np.float32)  # Ensure NumPy array
    if embeddings.ndim != 2:
        raise ValueError("Embeddings should be a 2D array.")

    dimension = embeddings.shape[1]  # Get embedding size
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance index
    index.add(embeddings)  # Add vectors to the FAISS index

    # Save the FAISS index
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    # Save the original documents
    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    print(f"Stored {len(texts)} documents in FAISS.")

def load_faiss_index():
    """Loads FAISS index if it exists, otherwise returns None."""
    if os.path.exists(FAISS_INDEX_FILE):
        return faiss.read_index(FAISS_INDEX_FILE)
    return None

def search_faiss(query, top_k=3):
    """Searches FAISS index for similar documents."""
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError("FAISS index file not found. Please create it first.")

    index = faiss.read_index(FAISS_INDEX_FILE)

    # Load original documents
    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        documents = f.readlines()

    # Generate query embedding
    query_embedding = np.array(model.encode([query]), dtype=np.float32)

    # Search FAISS for closest matches
    distances, indices = index.search(query_embedding, top_k)

    # Fetch the actual documents
    results = [documents[i].strip() for i in indices[0] if i < len(documents)]

    return results

if __name__ == "__main__":
    print("Embedding module loaded successfully.")

