from sentence_transformers import SentenceTransformer
from data_preparation_app import *
import pandas as pd
import numpy as np
import pickle


model = SentenceTransformer("all-mpnet-base-v2")

def chunk_text(text, max_length=128):
    """
    Splits a long text into smaller chunks of approximately `max_length` tokens.
    """
    if not text or len(text.split()) <= max_length:
        return [text]
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks

def data_embedding():
    data = data_cleaning()
    embeddings = []
    metadata = []

    for index, row in data.iterrows():
        row_embeddings = []
        row_metadata = {"id": row.get("id", index)}

        for col in ["title", "occupation", "employer_description", "description"]:
            if pd.notnull(row[col]):
                text_chunks = chunk_text(row[col])
                chunk_embeddings = []
                
                for chunk in text_chunks:
                    embedding = model.encode(chunk).astype("float32")
                    embedding = embedding / np.linalg.norm(embedding)
                    chunk_embeddings.append(embedding)
                
                if chunk_embeddings:
                    col_embedding = np.mean(chunk_embeddings, axis=0)
                    row_embeddings.append(col_embedding)
                    row_metadata[col] = row[col]
        if row_embeddings:
            combined_embedding = np.mean(row_embeddings, axis=0)
            embeddings.append(combined_embedding)
            metadata.append(row_metadata)

    embedding_df = pd.DataFrame(embeddings)
    embedding_df.to_csv("data/embedded_output.csv", index=False)
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return embeddings, metadata