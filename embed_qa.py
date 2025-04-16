import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load dataset
df = pd.read_csv('college_qa.csv')

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['question'].tolist())

# Save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, "college_index.faiss")
with open("answers.pkl", "wb") as f:
    pickle.dump(df['answer'].tolist(), f)
