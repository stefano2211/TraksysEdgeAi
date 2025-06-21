import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self, host: str, port: int):
        if not host or not port:
            raise ValueError("Qdrant host and port must be provided")
        self.client = QdrantClient(host=host, port=port)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collections = []

    def initialize_collection(self, collection_name: str):
        """Inicializa una colección en Qdrant."""
        try:
            vector_config = models.VectorParams(size=384, distance=models.Distance.COSINE)
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=vector_config
            )
            self.collections.append(collection_name)
            logger.info(f"Collection {collection_name} initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize collection {collection_name}: {str(e)}")
            raise

    def upsert_data(self, collection_name: str, points: list):
        """Inserta o actualiza puntos en una colección."""
        try:
            self.client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Upserted {len(points)} points in {collection_name}.")
        except Exception as e:
            logger.error(f"Failed to upsert data in {collection_name}: {str(e)}")
            raise

    def scroll_data(self, collection_name: str, filter_conditions: models.Filter = None, limit: int = 1000):
        """Recupera datos de una colección con filtros opcionales."""
        try:
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_conditions,
                limit=limit,
                with_payload=True
            )
            return results[0] if results else []
        except Exception as e:
            logger.error(f"Failed to scroll data in {collection_name}: {str(e)}")
            raise