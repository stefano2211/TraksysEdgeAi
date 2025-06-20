import io
import json
import logging
from minio import Minio
from minio.error import S3Error
import pdfplumber

logger = logging.getLogger(__name__)

class MinioClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket_name = None

    def ensure_bucket(self, bucket_name: str):
        """Crea el bucket si no existe."""
        self.bucket_name = bucket_name
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Bucket {bucket_name} created.")
        except S3Error as e:
            logger.error(f"Failed to create bucket {bucket_name}: {str(e)}")
            raise

    def get_pdf_content(self, filename: str) -> str:
        """Extrae el contenido de un PDF desde MinIO."""
        try:
            response = self.client.get_object(self.bucket_name, filename)
            pdf_data = response.read()
            response.close()
            response.release_conn()
            with pdfplumber.open(io.BytesIO(pdf_data)) as pdf:
                content = "\n".join(page.extract_text() or "" for page in pdf.pages)
            return json.dumps({
                "status": "success",
                "filename": filename,
                "content": content
            }, ensure_ascii=False)
        except S3Error as e:
            available_pdfs = [obj.object_name for obj in self.client.list_objects(self.bucket_name)]
            logger.warning(f"PDF not found: {filename}. Available PDFs: {', '.join(available_pdfs)}")
            return json.dumps({
                "status": "error",
                "message": f"PDF not found: {filename}. Available PDFs: {', '.join(available_pdfs)}",
                "filename": filename,
                "content": ""
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error extracting content from {filename}: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": str(e),
                "filename": filename,
                "content": ""
            }, ensure_ascii=False)