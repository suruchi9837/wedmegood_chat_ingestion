
import os
import requests
import tiktoken
from uuid import uuid4
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams

# Load environment variables
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Qdrant config
COLLECTION_NAME = "blog_data3"  # Update if needed
VECTOR_DIM = 1536
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
qdrant = QdrantClient(url="http://localhost:6333")
enc = tiktoken.get_encoding("cl100k_base")


def ensure_blog_collection():
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance="Cosine")
        )


def scrape_blog(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch {url} - Status {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")
    content_blocks = []

    for elem in soup.find_all(["p", "h2", "h3", "img", "iframe", "a"]):
        if elem.name in ["h2", "h3", "p"]:
            text = elem.get_text(strip=True)
            if text:
                content_blocks.append({"type": "text", "content": text})
        elif elem.name == "img" and elem.get("src", "").startswith("http"):
            content_blocks.append({"type": "image", "url": elem["src"]})
        elif elem.name == "iframe" and elem.get("src", "").startswith("http"):
            content_blocks.append({"type": "video", "url": elem["src"]})
        elif elem.name == "a" and elem.get("href", "").startswith("http"):
            href = elem["href"]
            content_blocks.append({"type": "link", "url": href})

    return content_blocks


def chunk_and_embed_blog(content_blocks, blog_url=""):
    ensure_blog_collection()

    chunks = []
    current_text = ""
    images, videos, links = [], [], []

    def add_chunk(text, imgs, vids, lks, chunk_index):
        if not text.strip():
            return
        tokens = enc.encode(text)
        if len(tokens) > CHUNK_SIZE:
            text = enc.decode(tokens[:CHUNK_SIZE])
        emb = openai_client.embeddings.create(
            model="text-embedding-ada-002", input=text).data[0].embedding
        chunks.append({
            "id": str(uuid4()),
            "vector": emb,
            "payload": {
                "chunk": text.strip(),
                "images": imgs[:3],
                "videos": vids[:2],
                "links": lks[:3],
                "index": chunk_index,
                "url": blog_url
            }
        })

    chunk_index = 0
    for block in content_blocks:
        if block["type"] == "text":
            if len(enc.encode(current_text)) + len(enc.encode(block["content"])) > CHUNK_SIZE:
                add_chunk(current_text, images, videos, links, chunk_index)
                current_text, images, videos, links = "", [], [], []
                chunk_index += 1
            current_text += " " + block["content"]
        elif block["type"] == "image":
            images.append(block["url"])
        elif block["type"] == "video":
            videos.append(block["url"])
        elif block["type"] == "link":
            links.append(block["url"])

    # Final chunk
    add_chunk(current_text, images, videos, links, chunk_index)

    # Save to Qdrant
    qdrant.upsert(collection_name=COLLECTION_NAME, points=chunks)
    return f"✅ Ingested blog with {len(chunks)} chunks."


def ingest_blog_url(url: str):
    content_blocks = scrape_blog(url)
    return chunk_and_embed_blog(content_blocks, blog_url=url)
