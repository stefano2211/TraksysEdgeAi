import logging
from cryptography.fernet import Fernet, InvalidToken
import hashlib
import json

logger = logging.getLogger(__name__)

class EncryptionManager:
    def __init__(self, key: str):
        self.fernet = Fernet(key.encode())

    def encrypt_data(self, data: dict) -> str:
        """Encripta un diccionario como JSON."""
        try:
            payload_json = json.dumps(data)
            encrypted = self.fernet.encrypt(payload_json.encode()).decode()
            return encrypted
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt_data(self, encrypted: str) -> dict:
        """Desencripta datos y devuelve un diccionario."""
        try:
            decrypted = self.fernet.decrypt(encrypted.encode()).decode()
            return json.loads(decrypted)
        except InvalidToken:
            logger.error("Failed to decrypt payload: Invalid token")
            raise
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise

    @staticmethod
    def generate_id(data: dict) -> str:
        """Genera un ID único basado en el hash del diccionario."""
        payload_json = json.dumps(data, sort_keys=True)
        return hashlib.md5(payload_json.encode()).hexdigest()