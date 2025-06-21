import httpx
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class AuthClient:
    def __init__(self, api_url: str, token_api_url: str):
        if not api_url or not token_api_url:
            raise ValueError("API URL and Token API URL must be provided")
        self.api_url = api_url
        self.token_api_url = token_api_url
        self.client = httpx.Client()
        self.token = None

    def fetch_token(self):
        """Obtiene un token JWT de la token-api."""
        try:
            response = self.client.get(f"{self.token_api_url}/get-token")
            response.raise_for_status()
            data = response.json()
            self.token = data["access_token"]
            logger.info("Token fetched from token-api")
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch token: {e.response.status_code} {e.response.text}")
            raise ValueError(f"No se pudo obtener token: {e.response.text}")
        except Exception as e:
            logger.error(f"Unexpected error fetching token: {str(e)}")
            raise ValueError(f"Error al obtener token: {str(e)}")

    def get(self, endpoint: str, params: Optional[Dict] = None) -> httpx.Response:
        """Realiza una solicitud GET a la API."""
        if not self.token:
            self.fetch_token()
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = self.client.get(
                f"{self.api_url}{endpoint}",
                headers=headers,
                params=params
            )
            if response.status_code == 401:
                logger.info("Token inválido, obteniendo nuevo token...")
                self.token = None
                self.fetch_token()
                headers = {"Authorization": f"Bearer {self.token}"}
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
        if not self.token:
            self.fetch_token()
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = self.client.post(
                f"{self.api_url}{endpoint}",
                headers=headers,
                json=json_data
            )
            if response.status_code == 401:
                logger.info("Token inválido, obteniendo nuevo token...")
                self.token = None
                self.fetch_token()
                headers = {"Authorization": f"Bearer {self.token}"}
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