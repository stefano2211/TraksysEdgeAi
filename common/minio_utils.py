import io
import json
import logging
from typing import List, Dict
from minio import Minio
from minio.error import S3Error
import pdfplumber
import os

logger = logging.getLogger(__name__)

class MinioClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False):
        if not all([endpoint, access_key, secret_key]):
            raise ValueError("MinIO endpoint, access_key, and secret_key must be provided")
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket_name = None
        self.mes_logs_bucket = None

    def ensure_bucket(self, bucket_name: str, mes_logs_bucket: str = None):
        """Crea los buckets si no existen."""
        self.bucket_name = bucket_name
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Bucket {bucket_name} created.")
        except S3Error as e:
            logger.error(f"Failed to create bucket {bucket_name}: {str(e)}")
            raise
        if mes_logs_bucket:
            self.mes_logs_bucket = mes_logs_bucket
            try:
                if not self.client.bucket_exists(mes_logs_bucket):
                    self.client.make_bucket(mes_logs_bucket)
                    logger.info(f"Bucket {mes_logs_bucket} created.")
            except S3Error as e:
                logger.error(f"Failed to create bucket {mes_logs_bucket}: {str(e)}")
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

    def get_all_json_logs(self) -> List[Dict]:
        """Lee todos los archivos JSON del bucket de logs."""
        try:
            objects = self.client.list_objects(self.mes_logs_bucket)
            logs = []
            for obj in objects:
                if obj.object_name.endswith('.json'):
                    response = self.client.get_object(self.mes_logs_bucket, obj.object_name)
                    json_data = response.read().decode('utf-8')
                    response.close()
                    response.release_conn()
                    try:
                        data = json.loads(json_data)
                        if isinstance(data, list):
                            logs.extend(data)
                        elif isinstance(data, dict):
                            logs.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in {obj.object_name}: {str(e)}")
            logger.info(f"Loaded {len(logs)} records from {self.mes_logs_bucket}")
            return logs
        except S3Error as e:
            logger.error(f"Failed to read JSON logs from {self.mes_logs_bucket}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error reading JSON logs: {str(e)}")
            return []