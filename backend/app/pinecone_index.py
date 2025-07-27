# app/pinecone_index.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from app.embedding_generator import generate_embedding
from pathlib import Path

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_ENVIRONMENT") or "us-east-1"
index_name = os.getenv("PINECONE_INDEX_NAME")
dimension = int(os.getenv("DIMENSION"))

assets_dir = Path(__file__).parent / "assets"

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Create index if it doesn't exist
if index_name not in [i['name'] for i in pc.list_indexes()]:
    print(f"[INFO] Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec={"cloud": "aws", "region": env}
    )

index = pc.Index(index_name)

# Embed and upsert images
def upsert_assets_to_pinecone():
    vectors = []
    for file in os.listdir(assets_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = assets_dir / file
            try:
                vector = generate_embedding(str(path)).tolist()
                vectors.append((file, vector))
                print(f"[UPLOADED] {file}")
            except Exception as e:
                print(f"[ERROR] Failed to embed {file}: {e}")
    
    if vectors:
        index.upsert(vectors=vectors)
        print(f"[✅] {len(vectors)} embeddings uploaded to Pinecone index '{index_name}'")
    else:
        print("[❌] No embeddings generated.")

if __name__ == "__main__":
    upsert_assets_to_pinecone()
