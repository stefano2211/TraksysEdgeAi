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
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool = False, tool_name: str = "manufacturing", bucket: str = "compliance-data", sop_prefix: str = "sop-pdfs/", mes_logs_prefix: str = "mes-logs/"):
        if not all([endpoint, access_key, secret_key]):
            raise ValueError("MinIO endpoint, access_key, and secret_key must be provided")
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket_name = bucket  # Usa el bucket especificado en config.yaml
        self.sop_prefix = sop_prefix
        self.mes_logs_prefix = mes_logs_prefix

    def ensure_bucket(self):
        """Crea el bucket y los prefijos sop-pdfs/ y mes-logs/ si no existen."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Bucket {self.bucket_name} created.")
            empty_data = io.BytesIO(b"")
            self.client.put_object(self.bucket_name, f"{self.sop_prefix}.marker", empty_data, 0)
            self.client.put_object(self.bucket_name, f"{self.mes_logs_prefix}.marker", empty_data, 0)
            logger.info(f"Prefijos {self.sop_prefix} y {self.mes_logs_prefix} inicializados en {self.bucket_name}.")
        except S3Error as e:
            logger.error(f"Failed to create bucket or prefixes in {self.bucket_name}: {str(e)}")
            raise

    def get_pdf_content(self, filename: str) -> str:
        try:
            object_name = f"{self.sop_prefix}{filename}"
            response = self.client.get_object(self.bucket_name, object_name)
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
            available_pdfs = [obj.object_name[len(self.sop_prefix):] for obj in self.client.list_objects(self.bucket_name, prefix=self.sop_prefix, recursive=True)]
            logger.warning(f"PDF not found: {object_name}. Available PDFs: {', '.join(available_pdfs)}")
            return json.dumps({
                "status": "error",
                "message": f"PDF not found: {filename}. Available PDFs: {', '.join(available_pdfs)}",
                "filename": filename,
                "content": ""
            }, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error extracting content from {object_name}: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": str(e),
                "filename": filename,
                "content": ""
            }, ensure_ascii=False)

    def get_all_json_logs(self) -> List[Dict]:
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=self.mes_logs_prefix, recursive=True)
            logs = []
            for obj in objects:
                if obj.object_name.endswith('.json'):
                    response = self.client.get_object(self.bucket_name, obj.object_name)
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
            logger.info(f"Loaded {len(logs)} records from {self.bucket_name}/{self.mes_logs_prefix}")
            return logs
        except S3Error as e:
            logger.error(f"Failed to read JSON logs from {self.bucket_name}/{self.mes_logs_prefix}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving JSON logs: {str(e)}")
            return []