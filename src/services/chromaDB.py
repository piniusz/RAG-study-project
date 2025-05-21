from google.genai import types
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.genai as genai
from dotenv import load_dotenv, find_dotenv
import os
import logging
import google.genai as genai
from chromadb import ClientAPI as ChromadbClientAPI


logging.basicConfig(level=logging.INFO)

load_dotenv(find_dotenv())

# Constants
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-001")
API_KEY = os.environ.get("PAID_GEMINI_API_KEY")
MAX_RETRIES = 3
INITIAL_WAIT_SECONDS = 10


# Initialize genai_client (ensure this is done only once)
genai_client = genai.Client(api_key=API_KEY)


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        model_id: str = EMBEDDING_MODEL,
        task_type: str = "retrieval_document",  # This will be used
    ):
        self.model_id = model_id
        self.task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:

        response = genai_client.models.embed_content(
            model=self.model_id,
            contents=input,
            config=types.EmbedContentConfig(
                task_type=self.task_type,
            ),
        )
        return [embedding.values for embedding in response.embeddings]


def load_to_chromadb(
    collection_name: str,
    client: ChromadbClientAPI,
    data: list[dict],
    batch_size: int = 100,
) -> str:
    pydantic_ai_collection = client.get_or_create_collection(
        name=collection_name, embedding_function=GeminiEmbeddingFunction()
    )
    logging.info(
        f"Loading {len(data)} documents to ChromaDB collection {collection_name}"
    )
    documents = []
    metadatas = []
    ids = []
    for k, i in enumerate(data):
        documents.append(i["content"])
        metadatas.append(i["metadata"])
        ids.append(str(k))

    for i in range(0, len(documents), batch_size):
        logging.info(
            f"Loading batch {i // batch_size + 1} of {len(documents) // batch_size + 1}"
        )
        batch_documents = documents[i : i + batch_size]
        bach_metadatas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        logging.info(
            f"Batch {i // batch_size + 1} of {len(documents) // batch_size + 1}:"
        )
        logging.info(f"  len(batch_documents): {len(batch_documents)}")
        logging.info(f"  len(bach_metadatas): {len(bach_metadatas)}")
        logging.info(f"  len(batch_ids): {len(batch_ids)}")

        # Add the batch to the collection
        pydantic_ai_collection.add(
            documents=batch_documents,
            metadatas=bach_metadatas,
            ids=batch_ids,
        )

    logging.info(
        f"Successfully loaded {len(data)} documents to ChromaDB collection {collection_name}"
    )
    return "Success"
