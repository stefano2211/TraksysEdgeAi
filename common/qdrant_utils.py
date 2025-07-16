import logging
import json
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self, host: str, port: int, sop_cache_ttl: int):
        if not host or not port:
            raise ValueError("Qdrant host and port must be provided")
        self.client = QdrantClient(host=host, port=port)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sop_cache_ttl = sop_cache_ttl  # Tiempo de vida de la caché en segundos
        self.collection_name = "sop_cache"
        self.initialize_collection()

    def initialize_collection(self):
        """Inicializa la colección sop_cache en Qdrant."""
        try:
            vector_config = models.VectorParams(size=384, distance=models.Distance.COSINE)
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vector_config
            )
            logger.info(f"Collection {self.collection_name} initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize collection {self.collection_name}: {str(e)}")
            raise

    def upsert_sop(self, key_values: dict, content: str, point_id: str):
        """Inserta o actualiza un SOP en la colección sop_cache con un timestamp de expiración."""
        try:
            # Generar embedding solo para los key_values
            key_values_str = json.dumps(key_values, sort_keys=True)
            embedding = self.model.encode(key_values_str).tolist()
            
            # Calcular tiempo de expiración
            expiration_timestamp = int(time.time()) + self.sop_cache_ttl
            
            # Crear punto con key_values y contenido
            point = models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "key_values": key_values,
                    "content": content,
                    "expiration_timestamp": expiration_timestamp
                }
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])
            logger.info(f"Upserted SOP for key_values {key_values} in {self.collection_name}.")
        except Exception as e:
            logger.error(f"Failed to upsert SOP in {self.collection_name}: {str(e)}")
            raise

    def get_sop(self, key_values: dict) -> dict:
        """Recupera un SOP de la caché basado en key_values, eliminando puntos caducados."""
        try:
            key_values_str = json.dumps(key_values, sort_keys=True)
            embedding = self.model.encode(key_values_str).tolist()
            
            # Filtrar puntos no expirados
            current_time = int(time.time())
            filter_conditions = models.Filter(
                must=[
                    models.FieldCondition(
                        key="expiration_timestamp",
                        range=models.Range(
                            gte=current_time
                        )
                    )
                ]
            )
            
            # Buscar puntos con el embedding más cercano
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                query_filter=filter_conditions,
                limit=1,
                with_payload=True
            )
            
            # Eliminar puntos caducados
            self._clean_expired_points(current_time)
            
            if results and results[0].payload.get("key_values") == key_values:
                logger.info(f"Cache hit for SOP with key_values {key_values}")
                return {
                    "status": "success",
                    "content": results[0].payload.get("content", ""),
                    "key_values": key_values
                }
            
            logger.info(f"Cache miss for SOP with key_values {key_values}")
            return {"status": "not_found", "content": "", "key_values": key_values}
        except Exception as e:
            logger.error(f"Failed to retrieve SOP from {self.collection_name}: {str(e)}")
            return {"status": "error", "message": str(e), "content": "", "key_values": key_values}

    def _clean_expired_points(self, current_time: int):
        """Elimina puntos caducados de la colección sop_cache."""
        try:
            filter_conditions = models.Filter(
                must=[
                    models.FieldCondition(
                        key="expiration_timestamp",
                        range=models.Range(
                            lte=current_time
                        )
                    )
                ]
            )
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                limit=1000
            )
            if results[0]:
                point_ids = [point.id for point in results[0]]
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                logger.info(f"Deleted {len(point_ids)} expired points from {self.collection_name}.")
        except Exception as e:
            logger.warning(f"Failed to clean expired points from {self.collection_name}: {str(e)}")