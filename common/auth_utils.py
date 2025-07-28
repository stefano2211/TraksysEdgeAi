import httpx
import logging
import os
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class AuthClient:
    def __init__(self, api_url: str):
        if not api_url:
            raise ValueError("API URL must be provided")
        self.api_url = api_url
        self.token = os.getenv("TOKEN")
        if not self.token:
            logger.error("No TOKEN provided in environment variables")
            raise ValueError("No TOKEN provided in environment variables")
        self.client = httpx.Client()

    def get(self, endpoint: str, params: Optional[Dict] = None) -> httpx.Response:
        """Realiza una solicitud GET a la API."""
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = self.client.get(
                f"{self.api_url}{endpoint}",
                headers=headers,
                params=params
            )
            logger.info(f"HTTP GET {self.api_url}{endpoint}: {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"Error en solicitud GET: {str(e)}")
            raise

    def post(self, endpoint: str, json_data: Optional[Dict] = None) -> httpx.Response:
        """Realiza una solicitud POST a la API."""
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = self.client.post(
                f"{self.api_url}{endpoint}",
                headers=headers,
                json=json_data
            )
            logger.info(f"HTTP POST {self.api_url}{endpoint}: {response.status_code}")
            return response
        except Exception as e:
            logger.error(f"Error en solicitud POST: {str(e)}")
            raise

    def close(self):
        self.client.close()