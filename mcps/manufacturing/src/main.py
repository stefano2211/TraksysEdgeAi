import logging
import json
import yaml
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from mcp.server.fastmcp import FastMCP, Context
from common.minio_utils import MinioClient
from common.qdrant_utils import QdrantManager
from common.auth_utils import AuthClient
from common.encryption_utils import EncryptionManager
from qdrant_client.http import models
from utils import expand_env_vars, DataValidator
import os
import inspect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("/app/config.yaml", "r") as f:
    config = yaml.safe_load(f)

config = expand_env_vars(config)
logger.info(f"API URL: {config['api']['url']}, TOKEN API URL: {config['api']['token_url']}")

mcp = FastMCP("Manufacturing Compliance Processor")

minio_client = MinioClient(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
)
minio_client.ensure_bucket(config["minio"]["bucket"], config["minio"]["mes_logs_bucket"])

qdrant_manager = QdrantManager(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT"))
)
for collection in config["qdrant"]["collections"]:
    if collection != "custom_rules":
        qdrant_manager.initialize_collection(collection)

auth_client = AuthClient(
    api_url=config["api"]["url"],
    token_api_url=config["api"]["token_url"]
)

encryption_manager = EncryptionManager(
    os.getenv("ENCRYPTION_KEY")
)

def fetch_mes_data(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_dates: Optional[List[str]] = None
) -> str:
    try:
        key_values = key_values or {}
        if not key_figures:
            fields_info = json.loads(list_fields(ctx))
            if fields_info["status"] != "success":
                return json.dumps({
                    "status": "error",
                    "message": "No se pudieron obtener campos válidos",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)
            key_figures = fields_info["key_figures"]
        else:
            key_figures = key_figures or []

        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values, start_date, end_date, specific_dates)
        must_conditions = []
        for k, v in key_values.items():
            must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))

        # Obtener datos crudos para identificar el campo de fecha
        processed_data = []
        collection_name = config["qdrant"]["collections"][0] if not config["data_source"]["use_minio_logs"] else config["qdrant"]["collections"][2]
        if config["data_source"]["use_minio_logs"]:
            all_data = minio_client.get_all_json_logs()
            if not all_data:
                return json.dumps({
                    "status": "no_data",
                    "message": f"No se encontraron datos en el bucket {config['minio']['mes_logs_bucket']}",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)
            full_data = []
            for record in all_data:
                if all(record.get(k) == v for k, v in key_values.items()):
                    date_field = DataValidator.identify_date_field([record])
                    item = {"date": DataValidator.detect_and_normalize_date(record, date_field) or "Desconocida"}
                    for field in fields_info["key_figures"] + list(fields_info["key_values"].keys()):
                        if field in record:
                            item[field] = record[field]
                    full_data.append(item)
            processed_data = full_data
        else:
            qdrant_results = qdrant_manager.scroll_data(
                collection_name=collection_name,
                filter_conditions=models.Filter(must=must_conditions) if must_conditions else None,
                limit=1000
            )
            for r in qdrant_results:
                encrypted_payload = r.payload.get("encrypted_payload")
                if encrypted_payload:
                    decrypted_data = encryption_manager.decrypt_data(encrypted_payload)
                    processed_data.append(decrypted_data)
            if not processed_data:
                params = {}
                if specific_dates:
                    params["specific_date"] = specific_dates[0]
                elif start_date and end_date:
                    params.update({"start_date": start_date, "end_date": end_date})
                response = auth_client.get("/machines/", params=params)
                response.raise_for_status()
                all_data = response.json()
                full_data = []
                for record in all_data:
                    if all(record.get(k) == v for k, v in key_values.items()):
                        date_field = DataValidator.identify_date_field([record])
                        item = {"date": DataValidator.detect_and_normalize_date(record, date_field) or "Desconocida"}
                        for field in fields_info["key_figures"] + list(fields_info["key_values"].keys()):
                            if field in record:
                                item[field] = record[field]
                        full_data.append(item)
                processed_data = full_data

        if processed_data:
            processed_data = [
                r for r in processed_data
                if all(r.get(k) == v for k, v in key_values.items())
            ]
        if key_figures:
            missing_figures = [k for k in key_figures if not any(k in r for r in processed_data)]
            if missing_figures:
                return json.dumps({
                    "status": "no_data",
                    "count": 0,
                    "data": [],
                    "message": f"No data found for fields: {', '.join(missing_figures)}.",
                    "covered_dates": []
                }, ensure_ascii=False)
        response_fields = ["date"] + list(key_values.keys()) + key_figures
        response_data = [
            {k: r[k] for k in response_fields if k in r}
            for r in processed_data
        ]
        coverage = DataValidator.check_date_coverage(response_data, start_date, end_date, specific_dates)
        return json.dumps({
            "status": "success" if response_data else "no_data",
            "count": len(response_data),
            "data": response_data,
            "message": coverage["message"],
            "covered_dates": coverage["covered_dates"]
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Data retrieval failed: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "count": 0,
            "data": [],
            "covered_dates": []
        }, ensure_ascii=False)
    
@mcp.tool()
def get_pdf_content(ctx: Context, filename: str) -> str:
    """Recupera el contenido de un archivo PDF almacenado en MinIO.
    Args:
        ctx (Context): Contexto de la solicitud proporcionado por el MCP.
        filename (str): Nombre del archivo PDF a recuperar.
    """
    return minio_client.get_pdf_content(filename)

@mcp.tool()
def list_fields(ctx: Context) -> str:
    """
    Lista los campos disponibles en el dataset MES, incluyendo métricas numéricas y valores categóricos.
    """
    try:
        if config["data_source"]["use_minio_logs"]:
            all_data = minio_client.get_all_json_logs()
        else:
            response = auth_client.get("/machines/")
            response.raise_for_status()
            all_data = response.json()
        if not all_data:
            return json.dumps({
                "status": "no_data",
                "message": "No se encontraron registros en el sistema MES",
                "key_figures": [],
                "key_values": {}
            }, ensure_ascii=False)
        sample = all_data[0]
        key_figures = [k for k, v in sample.items() if isinstance(v, (int, float))]
        key_values = {k: sorted({rec[k] for rec in all_data if k in rec}) for k, v in sample.items() if isinstance(v, str)}
        return json.dumps({
            "status": "success",
            "key_figures": key_figures,
            "key_values": key_values
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Field listing failed: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "key_figures": [],
            "key_values": {}
        }, ensure_ascii=False)

@mcp.tool()
def analyze_compliance(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_dates: Optional[List[str]] = None
) -> str:
    """
    Analiza el cumplimiento de los datos MES contra reglas SOP.

    Esta función recupera datos del sistema MES, los compara con procedimientos operativos estándar (SOPs)
    almacenados en MinIO y genera un informe de cumplimiento.

    Args:
        ctx (Context): Contexto de la solicitud proporcionado por el MCP.
        key_values (Optional[Dict[str, str]]): Filtros categóricos (e.g., {"machine": "ModelA"}).
        key_figures (Optional[List[str]]): Campos numéricos a analizar (e.g., ["temperature", "uptime"]).
        start_date (Optional[str]): Fecha de inicio para el rango de análisis (formato YYYY-MM-DD).
        end_date (Optional[str]): Fecha de fin para el rango de análisis (formato YYYY-MM-DD).
        specific_dates (Optional[List[str]]): Lista de fechas específicas para el análisis (formato YYYY-MM-DD).

    Returns:
        str: Respuesta en formato JSON con el estado, resultados, contenido de SOPs y notas.

    INSTRUCCIONES PARA EL LLM:
    1. **Obtener campos válidos**: Antes de construir la consulta, llama a la función `list_fields` para obtener los
       campos disponibles en el dataset MES (key_figures y key_values).
    2. **Validar campos**: Usa solo campos presentes en la respuesta de `list_fields` para `key_figures` (numéricos) y
       `key_values` (categóricos).
    3. **Estructura de la consulta**: La consulta debe seguir esta estructura:
       ```json
       {
           "key_values": {
               "<campo_categórico_1>": "<valor>",
               "<campo_categórico_2>": "<valor>"
           },
           "key_figures": [
               "<campo_numérico_1>",
               "<campo_numérico_2>"
           ],
           // Usa EITHER specific_dates OR start_date/end_date, no ambos
           "specific_dates": ["YYYY-MM-DD", ...], // Para fechas específicas
           // O
           "start_date": "YYYY-MM-DD", // Para un rango de fechas
           "end_date": "YYYY-MM-DD"
       }
       ```
    4. **Cuándo usar specific_dates vs. start_date/end_date**:
       - Usa `specific_dates` cuando la consulta menciona días concretos (e.g., "solo el 9 de abril de 2025" o
         "9 y 11 de abril de 2025"). Ejemplo: `specific_dates: ["2025-04-09", "2025-04-11"]`.
       - Usa `start_date` y `end_date` cuando la consulta menciona un rango de fechas (e.g., "del 9 al 11 de abril de
         2025"). Ejemplo: `start_date: "2025-04-09", end_date: "2025-04-11"`.
       - No combines `specific_dates` con `start_date`/`end_date` en la misma consulta.
       - Si la consulta no especifica fechas, omite ambos parámetros.
    5. **Ejemplo dinámico**:
       Supón que `list_fields` devuelve:
       ```json
       {
           "key_figures": ["temperature", "uptime", "vibration"],
           "key_values": {
               "machine": ["ModelA", "ModelB"],
               "production_line": ["Line1", "Line2", "Line3"]
           }
       }
       ```
       Consultas válidas serían:
       - Para fechas específicas:
         ```json
         {
             "key_values": {
                 "machine": "ModelA",
                 "production_line": "Line3"
             },
             "key_figures": ["temperature", "uptime"],
             "specific_dates": ["2025-04-09"]
         }
         ```
       - Para un rango de fechas:
         ```json
         {
             "key_values": {
                 "machine": "ModelA",
                 "production_line": "Line3"
             },
             "key_figures": ["temperature", "uptime"],
             "start_date": "2025-04-09",
             "end_date": "2025-04-11"
         }
         ```
       - Para un rango de fechas sin key figures:
         ```json
         {
             "key_values": {
                 "machine": "ModelA",
                 "production_line": "Line3"
             },
             "key_figures": [],
             "start_date": "2025-04-09",
             "end_date": "2025-04-11"
         }
         ```
    6. **Manejo de errores**:
       - Si los campos en `key_values` o `key_figures` no están en `list_fields`, ignora la consulta y devuelve un mensaje
         de error solicitando campos válidos.
       - Si las fechas proporcionadas no tienen el formato correcto (YYYY-MM-DD), solicita al usuario que las corrija.
    """
    try:
        key_values = key_values or {}
        if not key_figures:
            fields_info = json.loads(list_fields(ctx))
            if fields_info["status"] != "success":
                return json.dumps({
                    "status": "error",
                    "message": "No se pudieron obtener campos válidos",
                    "results": [],
                    "analysis_notes": ["No se pudieron obtener campos válidos"]
                }, ensure_ascii=False)
            key_figures = fields_info["key_figures"]
            logger.info(f"No key_figures provided, using all numeric fields: {key_figures}")
        else:
            key_figures = key_figures or []

        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values, start_date, end_date, specific_dates)
        valid_values = fields_info["key_values"]
        logger.info(f"Analyzing data: key_figures={key_figures}, key_values={key_values}, start_date={start_date}, end_date={end_date}, specific_dates={specific_dates}")
        identifier_field = None
        identifier_value = None
        if valid_values:
            for field in valid_values.keys():
                if field in key_values:
                    identifier_field = field
                    identifier_value = key_values[field]
                    break
            if not identifier_field:
                identifier_field = next(iter(valid_values))
                identifier_value = key_values.get(identifier_field)
        logger.info(f"Selected identifier_field: {identifier_field}, identifier_value: {identifier_value}")
        fetch_result = json.loads(fetch_mes_data(ctx, key_values, key_figures, start_date, end_date, specific_dates))
        analysis_notes = [fetch_result.get("message", "")] if fetch_result.get("message") else []
        if fetch_result["status"] == "no_data":
            return json.dumps({
                "status": "no_data",
                "message": fetch_result["message"],
                "period": f"{start_date or 'N/A'} to {end_date or 'N/A'}" if start_date else f"Specific dates: {specific_dates or 'N/A'}",
                "identifier": f"{identifier_field}={identifier_value}" if identifier_field and identifier_value else "all records",
                "metrics_analyzed": key_figures,
                "results": [],
                "sop_content": {},
                "analysis_notes": analysis_notes
            }, ensure_ascii=False)
        if fetch_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": fetch_result.get("message", "Error retrieving data"),
                "results": [],
                "analysis_notes": analysis_notes
            }, ensure_ascii=False)
        identifiers = {r[identifier_field] for r in fetch_result["data"] if identifier_field in r} if identifier_field else set()
        sop_content = {}
        if identifiers and identifier_field:
            for identifier in identifiers:
                pdf_result = json.loads(get_pdf_content(ctx, f"{identifier}.pdf"))
                if pdf_result["status"] == "success":
                    sop_content[identifier] = pdf_result.get("content", "")
                    logger.info(f"SOP content for {identifier_field}={identifier}: {sop_content[identifier][:100]}...")
                else:
                    sop_content[identifier] = ""
                    analysis_notes.append(f"Failed to load SOP for {identifier_field}={identifier}: {pdf_result['message']}")
                    logger.warning(f"Failed to load SOP for {identifier_field}={identifier}: {pdf_result['message']}")
        else:
            analysis_notes.append("No identifier field or identifiers found; no SOPs loaded.")
        results = []
        for record in fetch_result["data"]:
            analysis = {
                "date": record.get("date", "Desconocida"),
                **{k: record.get(k) for k in key_values},
                "metrics": {k: record[k] for k in key_figures if k in record}
            }
            results.append(analysis)
        analysis_notes.append(f"Filtered data for {key_values}")
        period = "all dates"
        if specific_dates:
            period = f"Specific dates: {', '.join(specific_dates)}"
        elif start_date and end_date:
            period = f"{start_date} to {end_date}"
        return json.dumps({
            "status": "success",
            "period": period,
            "identifier": f"{identifier_field}={identifier_value}" if identifier_field and identifier_value else "all records",
            "metrics_analyzed": key_figures,
            "results": results,
            "sop_content": sop_content,
            "analysis_notes": analysis_notes
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Compliance analysis failed: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "results": [],
            "analysis_notes": [str(e)]
        }, ensure_ascii=False)

@mcp.tool()
def get_mes_dataset(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_dates: Optional[List[str]] = None
) -> str:
    """
    Recupera datos del sistema MES aplicando filtros por campos categóricos, métricas numéricas y fechas.

    INSTRUCCIONES PARA EL LLM:
    - Antes de construir la consulta, llama a `list_fields` para obtener los campos válidos (`key_figures` y `key_values`).
    - Usa solo campos presentes en la respuesta de `list_fields`.
    - Usa `specific_dates` (lista de fechas YYYY-MM-DD) para días concretos, o `start_date` y `end_date` (YYYY-MM-DD) para rangos. No combines ambos.
    - Si los campos o fechas no son válidos, devuelve un mensaje de error solicitando corrección.

    Ejemplos de uso:
    1. Fechas específicas:
       {
           "key_values": {"machine": "ModelA"},
           "key_figures": ["defects"],
           "specific_dates": ["2025-04-09", "2025-04-11"]
       }
    2. Rango de fechas:
       {
           "key_values": {"machine": "ModelA"},
           "key_figures": ["defects"],
           "start_date": "2025-04-09",
           "end_date": "2025-04-11"
       }
    3. Sin filtros:
       {
           "key_values": {"machine": "ModelA"},
           "key_figures": [],
           "start_date": "2025-04-09",
           "end_date": "2025-04-11"
       }

    Args:
        ctx (Context): Contexto FastMCP.
        key_values (Optional[Dict[str, str]]): Filtros categóricos.
        key_figures (Optional[List[str]]): Métricas numéricas.
        start_date (Optional[str]): Fecha inicio (YYYY-MM-DD).
        end_date (Optional[str]): Fecha fin (YYYY-MM-DD).
        specific_dates (Optional[List[str]]): Lista de fechas específicas (YYYY-MM-DD).

    Returns:
        str: JSON con los datos filtrados.
    """
    try:
        fetch_result = json.loads(fetch_mes_data(ctx, key_values, key_figures, start_date, end_date, specific_dates))
        if fetch_result["status"] != "success":
            logger.info(f"No data retrieved: {fetch_result.get('message', 'No message')}")
            return json.dumps([], ensure_ascii=False)
        return json.dumps(fetch_result["data"], ensure_ascii=False)
    except Exception as e:
        logger.error(f"Dataset retrieval failed: {str(e)}")
        return json.dumps([], ensure_ascii=False)

@mcp.tool()
def list_available_tools(ctx: Context) -> str:
    """
    Lista las herramientas disponibles en el MCP, incluyendo las definidas por el usuario y las internas de FastMCP.
    INSTRUCCIONES PARA EL LLM:
    - Esta función devuelve un JSON con el nombre, descripción y parámetros de cada herramienta.
    - Si no hay herramientas disponibles, devuelve un mensaje indicando que no se encontraron herramientas.
    - Si ocurre un error al listar las herramientas, devuelve un mensaje de error.
    """
    try:
        tools = []
        try:
            if hasattr(mcp, 'tools'):
                tool_registry = mcp.tools
            elif hasattr(mcp, 'get_tools'):
                tool_registry = mcp.get_tools()
            else:
                tool_registry = None
            if tool_registry:
                for tool_name, tool_func in tool_registry.items():
                    if callable(tool_func):
                        signature = inspect.signature(tool_func)
                        parameters = [
                            {
                                "name": param_name,
                                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                                "default": param.default if param.default != inspect.Parameter.empty else None
                            }
                            for param_name, param in signature.parameters.items()
                            if param_name != 'ctx'
                        ]
                        docstring = inspect.getdoc(tool_func) or "Sin descripción disponible."
                        tools.append({
                            "name": tool_name,
                            "description": docstring,
                            "parameters": parameters
                        })
        except Exception as e:
            logger.warning(f"Fallo al acceder al registro interno de FastMCP: {str(e)}")
        if not tools:
            module = inspect.getmodule(inspect.currentframe())
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj):
                    try:
                        source = inspect.getsource(obj)
                        if '@mcp.tool' in source:
                            signature = inspect.signature(obj)
                            parameters = [
                                {
                                    "name": param_name,
                                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                                    "default": param.default if param.default != inspect.Parameter.empty else None
                                }
                                for param_name, param in signature.parameters.items()
                                if param_name != 'ctx'
                            ]
                            docstring = inspect.getdoc(obj) or "Sin descripción disponible."
                            tools.append({
                                "name": name,
                                "description": docstring,
                                "parameters": parameters
                            })
                    except Exception as e:
                        logger.debug(f"No se pudo inspeccionar la función {name}: {str(e)}")
        return json.dumps({
            "status": "success" if tools else "no_data",
            "count": len(tools),
            "tools": tools,
            "message": "Lista de herramientas recuperada exitosamente." if tools else "No se encontraron herramientas disponibles."
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to list available tools: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "count": 0,
            "tools": []
        }, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run()