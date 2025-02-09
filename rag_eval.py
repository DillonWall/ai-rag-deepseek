from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import argparse
import torch

# Config
load_dotenv()
qdrant_ip = str(os.getenv("QDRANT_IP"))
qdrant_port = int(os.getenv("QDRANT_PORT") or 0)
collection_name = str(os.getenv("QDRANT_COLLECTION_NAME"))
embed_model_name = str(os.getenv("EMBEDDING_MODEL_NAME"))
model_name = str(os.getenv("MODEL_NAME"))
max_new_tokens = int(os.getenv("MAX_NEW_TOKENS") or 0)
assert qdrant_ip
assert qdrant_port != 0
assert collection_name
assert embed_model_name
assert model_name
assert max_new_tokens != 0

# Parse args
parser = argparse.ArgumentParser("DeepSeek-R1 RAG Query")
parser.add_argument("prompt", help="The prompt to ask the AI model", type=str)
args = parser.parse_args()

# Setup
embedder = SentenceTransformer(embed_model_name)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 4-bit quantization (requires bitsandbytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
prompt = args.prompt + "\n<think>\n"


def setup_qdrant() -> QdrantClient:
    client = QdrantClient(host=qdrant_ip, port=qdrant_port)
    vec_size = embedder.get_sentence_embedding_dimension()
    assert vec_size is not None

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vec_size,
                distance=Distance.COSINE,
            ),
        )
    return client


def generate_rag_prompt(query: str) -> str:
    qdrant = setup_qdrant()
    embedded_prompt = embedder.encode(args.prompt)
    query_response = qdrant.query_points(
        collection_name=collection_name,
        query=embedded_prompt,
        limit=5,
    )
    context = "\n".join([r.payload["text"] for r in query_response.points])
    return f"Context: {context}\nQuestion {query}\n<think>"


# Eval
rag_prompt = generate_rag_prompt(args.prompt)
print(rag_prompt)
inputs = tokenizer(rag_prompt, return_tensors="pt").to(device)
model = model.to(device)
outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
