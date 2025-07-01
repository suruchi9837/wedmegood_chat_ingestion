import os
import tiktoken
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

# === Load API Key ===
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(url="http://localhost:6333")

# === Config ===
collection_name = "vendor_master"
vector_dim = 1536
batch_size = 50

def ensure_collection():
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance="Cosine")
        )

def load_excel(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna(subset=["vendor_id", "brand_name", "city", "ratings", "reviews", "score", "text"])
    df["vendor_id"] = df["vendor_id"].astype(int)
    df["ratings"] = df["ratings"].astype(float)
    df["reviews"] = df["reviews"].astype(int)
    df["score"] = df["score"].astype(float)
    df["text"] = df["text"].astype(str).str.strip()
    return df

def embed_texts(texts, batch_token_limit=280000, max_batch_size=1000):
    enc = tiktoken.get_encoding("cl100k_base")
    all_embeddings = []

    current_batch = []
    current_tokens = 0

    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            print(f"⚠️ Skipping invalid text at index {i}")
            continue

        token_len = len(enc.encode(text))

        if token_len >= batch_token_limit:
            print(f"⚠️ Skipping overly long text at index {i} ({token_len} tokens)")
            continue

        if (current_tokens + token_len > batch_token_limit) or (len(current_batch) >= max_batch_size):
            print(f"🚀 Sending batch of {len(current_batch)} texts ({current_tokens} tokens)")
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=current_batch
            )
            all_embeddings.extend([record.embedding for record in response.data])
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += token_len

    if current_batch:
        print(f"🚀 Sending final batch of {len(current_batch)} texts ({current_tokens} tokens)")
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=current_batch
        )
        all_embeddings.extend([record.embedding for record in response.data])

    return all_embeddings

def build_points(df):
    texts = df["text"].tolist()
    embeddings = embed_texts(texts)

    if len(embeddings) != len(df):
        raise ValueError(f"❌ Mismatch: {len(embeddings)} embeddings vs {len(df)} rows!")

    points = []
    for idx, (_, row) in enumerate(df.iterrows()):
        points.append({
            "id": int(row["vendor_id"]),
            "vector": embeddings[idx],
            "payload": {
                "id": int(row["vendor_id"]),
                "text": row["text"],
                # "score": float(row["score"]),
                "metadata": {
                    "score": float(row["score"]),
                    "name": row["brand_name"],
                    "city": row["city"],
                    "category": row.get("category", ""),
                    "ratings": float(row["ratings"]),
                    "reviews": int(row["reviews"])
                }
            }
        })
    return points

def batch_upload(points):
    for i in range(0, len(points), batch_size):
        print(f"⬆️ Uploading batch {i} to {i + batch_size}")
        qdrant.upsert(collection_name=collection_name, points=points[i:i + batch_size])

def main(excel_path="vendor_data_final1.xlsx"):  # Now accepts dynamic path
    ensure_collection()
    df = load_excel(excel_path)
    print(f"✅ Loaded {len(df)} rows from Excel")
    points = build_points(df)
    batch_upload(points)
    print(f"✅ Uploaded {len(points)} vendors to Qdrant")

if __name__ == "__main__":
    main()
