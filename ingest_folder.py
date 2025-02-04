from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import FilterSelector, VectorParams, Distance, \
    PointStruct, Filter, MatchValue, FieldCondition
from dotenv import load_dotenv
import os
import argparse
import uuid

# Config
load_dotenv()
qdrant_ip = str(os.getenv("QDRANT_IP"))
qdrant_port = int(os.getenv("QDRANT_PORT") or 0)
collection_name = str(os.getenv("QDRANT_COLLECTION_NAME"))
embed_model_name = str(os.getenv("EMBEDDING_MODEL_NAME"))
assert (qdrant_ip)
assert (qdrant_port != 0)
assert (collection_name)
assert (embed_model_name)

# Parse args
parser = argparse.ArgumentParser("RAG File ingester")
parser.add_argument(
    "folderpath", help="Path to the folder to ingest files from", type=str)
args = parser.parse_args()

# Setup
embedder = SentenceTransformer(embed_model_name)


def setup_qdrant():
    client = QdrantClient(host=qdrant_ip, port=qdrant_port)
    vec_size = embedder.get_sentence_embedding_dimension()
    assert (vec_size is not None)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vec_size,
                distance=Distance.COSINE,
            ),
        )
    return client


def chunk_fixed_size(filepath: str) -> list[str]:
    chunk_size = 300
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        start, end = 0, chunk_size
        while start < len(text):
            end = min(end, len(text))
            chunks.append(text[start:end])
            start = end
            end += chunk_size
    return chunks


def import_file(client: QdrantClient, filepath: str):
    chunks = chunk_fixed_size(filepath)
    embeddings = embedder.encode(chunks)
    # delete previous vectors with this filepath before inserting new ones
    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(
                    key="filepath",
                    match=MatchValue(value=filepath),
                )],
            ),
        ),
    )
    client.upload_points(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk,
                payload={
                    "filepath": filepath,
                    "chunk_num": chunk_num,
                    "text": chunks[chunk_num],
                }
            ) for chunk_num, chunk in enumerate(embeddings)
        ],
    )


client = setup_qdrant()
print(client)
print(chunk_fixed_size(args.folderpath)[:5])
import_file(client, args.folderpath)
