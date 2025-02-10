from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


def setup_qdrant(
        qdrant_ip: str,
        qdrant_port: int,
        vec_size: int,
        collection_name: str,
) -> QdrantClient:
    client = QdrantClient(host=qdrant_ip, port=qdrant_port)

    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vec_size,
                    distance=Distance.COSINE,
                ),
            )
    except Exception:
        print(
            Exception("Could not create collection in QDrant. \n" +
                      "Please ensure the qdrant client is running")
        )
        quit(1)

    return client
