import faiss
import numpy as np
from data_embedding import *
import pickle
import os

def store_emb():
    if os.path.exists("faiss_index.idx"):
        return faiss.read_index("faiss_index.idx")
    
    embeddings, metadata = data_embedding()
    print(embeddings)
    dimension = len(embeddings[0])

    index = faiss.IndexFlatIP(dimension)
    embedding_array = np.array(embeddings)

    index.add(embedding_array)

    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    faiss.write_index(index, "faiss_index.idx")

    return index