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
from utils import expand_env_vars, DataValidator
import os
import inspect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("/app/config.yaml", "r") as f:
    config = yaml.safe_load(f)

config = expand_env_vars(config)
logger.info(f"API URL: {config['api']['url']}, TOKEN API URL: {config['api']['token_url']}")

mcp = FastMCP("Multi-Area Compliance Processor")
tool_name = mcp.name.lower().replace(" ", "-")

auth_client = AuthClient(
    api_url=config["api"]["url"],
    token_api_url=config["api"]["token_url"]
)
encryption_manager = EncryptionManager(
    os.getenv("ENCRYPTION_KEY")
)
qdrant_manager = QdrantManager(
    host=os.getenv("QDRANT_HOST"),
    port=int(os.getenv("QDRANT_PORT")),
    collections=config.get("qdrant", {}).get("collections", ["sop_cache"]),
    sop_cache_ttl=config.get("qdrant", {}).get("sop_cache_ttl", 36000)
)

def get_minio_client(tool_name: str) -> MinioClient:
    if tool_name not in config.get("tools", {}):
        raise ValueError(f"Tool {tool_name} not configured")
    tool_config = config["tools"][tool_name]
    return MinioClient(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        tool_name=tool_name,
        bucket=tool_config["minio"]["bucket"],
        sop_prefix=tool_config["minio"]["sop_prefix"],
        mes_logs_prefix=tool_config["minio"]["mes_logs_prefix"]
    )

def initialize_buckets():
    for tool_name in config.get("tools", {}):
        minio_client = get_minio_client(tool_name)
        minio_client.ensure_bucket()

initialize_buckets()

def get_tool_client(tool_name: str):
    minio_client = get_minio_client(tool_name)
    return minio_client, qdrant_manager

def fetch_data(ctx: Context, tool_name: str, key_values: Optional[Dict[str, str]] = None, key_figures: Optional[List[Dict]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, specific_dates: Optional[List[str]] = None) -> str:
    try:
        if tool_name not in config.get("tools", {}):
            return json.dumps({
                "status": "error",
                "message": f"Tool {tool_name} not configured",
                "count": 0,
                "data": [],
                "covered_dates": []
            }, ensure_ascii=False)
        
        tool_config = config["tools"][tool_name]
        minio_client, qdrant_manager = get_tool_client(tool_name)
        key_values = key_values or {}
        normalized_key_figures = []
        figure_ranges = {}
        if key_figures:
            for item in key_figures:
                if isinstance(item, str):
                    normalized_key_figures.append(item)
                elif isinstance(item, dict) and "field" in item:
                    normalized_key_figures.append(item["field"])
                    if "min" in item or "max" in item:
                        figure_ranges[item["field"]] = {
                            "min": item.get("min", None),
                            "max": item.get("max", None)
                        }
        if not normalized_key_figures:
            fields_info = json.loads(list_fields(ctx, tool_name))
            if fields_info["status"] != "success":
                return json.dumps({
                    "status": "error",
                    "message": "No se pudieron obtener campos válidos",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)
            normalized_key_figures = fields_info["key_figures"]
        else:
            normalized_key_figures = normalized_key_figures or []

        logger.info(f"Validating fields in fetch_data for tool_name: {tool_name}")
        fields_info = DataValidator.validate_fields(ctx, normalized_key_figures, key_values, start_date, end_date, specific_dates, tool_name=tool_name)
        
        if tool_config["type"] == "json":
            all_data = minio_client.get_all_json_logs()
            if not all_data:
                return json.dumps({
                    "status": "no_data",
                    "message": f"No se encontraron datos en el bucket {minio_client.bucket}/{minio_client.mes_logs_prefix}",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)
        else:
            api_endpoint = tool_config.get("api_endpoint")
            if not api_endpoint or api_endpoint == "null":
                return json.dumps({
                    "status": "error",
                    "message": f"api_endpoint not configured for tool {tool_name}",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)
            params = {}
            if specific_dates:
                params["specific_date"] = specific_dates[0]
            elif start_date and end_date:
                params.update({"start_date": start_date, "end_date": end_date})
            response = auth_client.get(api_endpoint, params=params)
            response.raise_for_status()
            all_data = response.json()

        processed_data = []
        for record in all_data:
            if all(record.get(k) == v for k, v in key_values.items()):
                date_field = DataValidator.identify_date_field([record])
                record_date = DataValidator.detect_and_normalize_date(record, date_field)
                if (not specific_dates and not start_date and not end_date) or \
                   (specific_dates and record_date in specific_dates) or \
                   (start_date and end_date and start_date <= record_date <= end_date):
                    item = {"date": record_date or "Desconocida"}
                    for field in fields_info["key_figures"] + list(fields_info["key_values"].keys()):
                        if field in record:
                            item[field] = record[field]
                    for field, ranges in figure_ranges.items():
                        if field in item:
                            value = item[field]
                            if isinstance(value, (int, float)):
                                if ranges["min"] is not None and value < ranges["min"]:
                                    item[field] = None
                                if ranges["max"] is not None and value > ranges["max"]:
                                    item[field] = None
                            else:
                                item[field] = None
                    processed_data.append(item)

        if processed_data:
            processed_data = [
                r for r in processed_data
                if all(r.get(k) == v for k, v in key_values.items())
            ]
        if normalized_key_figures:
            missing_figures = [k for k in normalized_key_figures if not any(k in r for r in processed_data)]
            if missing_figures:
                return json.dumps({
                    "status": "no_data",
                    "count": 0,
                    "data": [],
                    "message": f"No data found for fields: {', '.join(missing_figures)}.",
                    "covered_dates": []
                }, ensure_ascii=False)
        response_fields = ["date"] + list(key_values.keys()) + normalized_key_figures
        response_data = [
            {k: r[k] for k in response_fields if k in r and r[k] is not None}
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
        logger.error(f"Data retrieval failed for {tool_name}: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "count": 0,
            "data": [],
            "covered__dates": []
        }, ensure_ascii=False)

@mcp.tool()
def get_pdf_content(ctx: Context, tool_name: str, key_values: Dict[str, str]) -> str:
    """
    Recupera el contenido de un archivo PDF para un área específica usando Qdrant como caché si está disponible.

    INSTRUCCIONES PARA EL LLM:
    1. **Seleccionar el tool_name**: Usa `list_available_tools` para obtener las áreas disponibles y elige el `tool_name` adecuado (e.g., "manufacturing", "human_resources").
    2. **Validar key_values**: Asegúrate de que `key_values` contenga exactamente un par clave-valor (e.g., {"machine": "ModelA"} o {"employee_id": "001"}) basado en los campos categóricos de `list_fields` para el `tool_name` seleccionado.
    3. **Generar filename**: El filename se deriva del valor en `key_values` (e.g., "ModelA.pdf" para {"machine": "ModelA"}).
    4. **Ejemplo de consulta**:
       ```json
       {
           "tool_name": "manufacturing",
           "key_values": {"machine": "ModelA"}
       }
       ```
       O para `human_resources`:
       ```json
       {
           "tool_name": "human_resources",
           "key_values": {"employee_id": "001"}
       }
       ```
    5. **Manejo de errores**: Si `key_values` tiene más de un par o el archivo no existe, devuelve un error solicitando un solo par clave-valor válido.
    """
    try:
        if tool_name not in config.get("tools", {}):
            return json.dumps({
                "status": "error",
                "message": f"Tool {tool_name} not configured",
                "filename": "",
                "content": ""
            }, ensure_ascii=False)
        
        minio_client, qdrant_manager = get_tool_client(tool_name)
        if len(key_values) != 1:
            raise ValueError("get_pdf_content expects exactly one key-value pair")
        field, value = next(iter(key_values.items()))
        filename = f"{value}.pdf"
        logger.info(f"Attempting to fetch PDF: {filename} for key_values {key_values} in {tool_name}")
        point_id = encryption_manager.generate_id(key_values)
        
        if qdrant_manager:
            cache_result = qdrant_manager.get_sop(key_values)
            if cache_result["status"] == "success":
                logger.info(f"Cache hit for SOP with key_values {key_values} in {tool_name}")
                return json.dumps({
                    "status": "success",
                    "filename": filename,
                    "content": cache_result["content"]
                }, ensure_ascii=False)
        
        logger.info(f"Cache miss for {filename}, fetching from MinIO with full path: {minio_client.sop_prefix}{filename}")
        minio_result = json.loads(minio_client.get_pdf_content(filename))
        if minio_result["status"] != "success":
            return json.dumps(minio_result, ensure_ascii=False)
        
        content = minio_result["content"]
        if qdrant_manager:
            try:
                qdrant_manager.upsert_sop(key_values, content, point_id)
            except Exception as e:
                logger.warning(f"Failed to cache SOP in Qdrant for {tool_name}: {str(e)}. Continuing with MinIO content.")
        
        return json.dumps({
            "status": "success",
            "filename": filename,
            "content": content
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error retrieving PDF content for {filename} in {tool_name}: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "filename": filename,
            "content": ""
        }, ensure_ascii=False)

@mcp.tool()
def list_fields(ctx: Context, tool_name: str) -> str:
    """
    Lista los campos disponibles en el dataset de un área específica.

    INSTRUCCIONES PARA EL LLM:
    1. **Seleccionar el tool_name**: Usa `list_available_tools` para obtener las áreas disponibles y elige el `tool_name` adecuado (e.g., "manufacturing", "human_resources").
    2. **Uso**: Llama a esta función con el `tool_name` para obtener los campos `key_figures` (numéricos) y `key_values` (categóricos) específicos de esa área.
    3. **Ejemplo de consulta**:
       ```json
       {
           "tool_name": "manufacturing"
       }
       ```
       O:
       ```json
       {
           "tool_name": "human_resources"
       }
       ```
    4. **Resultado**: Devuelve un JSON con `key_figures` y `key_values` que deben usarse en otras funciones como `fetch_data` o `analyze_compliance`.
    """
    try:
        if tool_name not in config.get("tools", {}):
            return json.dumps({
                "status": "error",
                "message": f"Tool {tool_name} not configured",
                "key_figures": [],
                "key_values": {}
            }, ensure_ascii=False)
        
        minio_client, _ = get_tool_client(tool_name)
        tool_config = config["tools"][tool_name]
        if tool_config["type"] == "json":
            all_data = minio_client.get_all_json_logs()
        else:
            api_endpoint = tool_config.get("api_endpoint")
            if not api_endpoint or api_endpoint == "null":
                return json.dumps({
                    "status": "error",
                    "message": f"api_endpoint not configured for tool {tool_name}",
                    "key_figures": [],
                    "key_values": {}
                }, ensure_ascii=False)
            response = auth_client.get(api_endpoint)
            response.raise_for_status()
            all_data = response.json()
        if not all_data:
            return json.dumps({
                "status": "no_data",
                "message": f"No se encontraron registros en el sistema {tool_name}",
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
        logger.error(f"Field listing failed for {tool_name}: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "key_figures": [],
            "key_values": {}
        }, ensure_ascii=False)

@mcp.tool()
def analyze_compliance(ctx: Context, tool_name: str, key_values: Optional[Dict[str, str]] = None, key_figures: Optional[List[Dict]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, specific_dates: Optional[List[str]] = None) -> str:
    """
    Analiza el cumplimiento de los datos de un área específica contra reglas SOP.

    INSTRUCCIONES PARA EL LLM:
    1. **Seleccionar el tool_name**: Usa `list_available_tools` para obtener las áreas disponibles y elige el `tool_name` adecuado (e.g., "manufacturing", "human_resources").
    2. **Obtener campos válidos**: Llama a `list_fields` con el Ejemplo:`tool_name` para obtener los campos `key_figures` y `key_values` específicos de esa área.
    3. **Validar campos**: Usa solo campos presentes en la respuesta de `list_fields` para el `tool_name` seleccionado.
    4. **Estructura de la consulta**: Sigue esta estructura:
       ```json
       {
           "tool_name": "<nombre_del_area>",
           "key_values": {
               "<campo_categórico_1>": "<valor>",
               "<campo_categórico_2>": "<valor>"
           },
           "key_figures": [
               {"field": "<campo_numérico_1>", "min": <número>, "max": <número>},
               {"field": "<campo_numérico_2>", "min": <número>, "max": <número>}
           ],
           // Usa EITHER specific_dates OR start_date/end_date, no ambos
           "specific_dates": ["YYYY-MM-DD", ...], // Para fechas específicas
           // O
           "start_date": "YYYY-MM-DD", // Para un rango de fechas
           "end_date": "YYYY-MM-DD"
       }
       ```
       Nota: `min` y `max` son opcionales. Si no se especifican, se incluyen todos los valores.
    5. **Cuándo usar specific_dates vs. start_date/end_date**:
       - Usa `specific_dates` para días concretos (e.g., "solo el 18 de julio de 2025"). Ejemplo: `specific_dates: ["2025-07-18"]`.
       - Usa `start_date` y `end_date` para rangos (e.g., "del 18 al 20 de julio de 2025"). Ejemplo: `start_date: "2025-07-18", end_date: "2025-07-20"`.
       - No combines ambos.
       - Omite fechas si no se especifican.
    6. **Ejemplo por área**:
       - Para `manufacturing`:
         ```json
         {
             "tool_name": "manufacturing",
             "key_values": {"machine": "ModelA"},
             "key_figures": [{"field": "temperature", "min": 70, "max": 80}],
             "start_date": "2025-07-18",
             "end_date": "2025-07-20"
         }
         ```
       - Para `human_resources`:
         ```json
         {
             "tool_name": "human_resources",
             "key_values": {"employee_id": "001"},
             "key_figures": [{"field": "hours_worked", "min": 8}],
             "specific_dates": ["2025-07-18"]
         }
         ```
    7. **Manejo de errores**:
       - Si `tool_name` no es válido, solicita al usuario usar un nombre de `list_available_tools`.
       - Si los campos no son válidos, solicita corrección basada en `list_fields`.
       - Si las fechas o rangos son inválidos, solicita formato correcto (YYYY-MM-DD para fechas, numéricos para rangos).
    """
    try:
        if tool_name not in config.get("tools", {}):
            return json.dumps({
                "status": "error",
                "message": f"Tool {tool_name} not configured",
                "results": [],
                "analysis_notes": [f"Tool {tool_name} not configured"]
            }, ensure_ascii=False)
        
        minio_client, qdrant_manager = get_tool_client(tool_name)
        key_values = key_values or {}
        normalized_key_figures = []
        if key_figures:
            for item in key_figures:
                if isinstance(item, str):
                    normalized_key_figures.append(item)
                elif isinstance(item, dict) and "field" in item:
                    normalized_key_figures.append(item["field"])
        if not normalized_key_figures:
            fields_info = json.loads(list_fields(ctx, tool_name))
            if fields_info["status"] != "success":
                return json.dumps({
                    "status": "error",
                    "message": "No se pudieron obtener campos válidos",
                    "results": [],
                    "analysis_notes": ["No se pudieron obtener campos válidos"]
                }, ensure_ascii=False)
            normalized_key_figures = fields_info["key_figures"]
            logger.info(f"No key_figures provided, using all numeric fields for {tool_name}: {normalized_key_figures}")
        else:
            normalized_key_figures = normalized_key_figures or []

        fields_info = DataValidator.validate_fields(ctx, normalized_key_figures, key_values, start_date, end_date, specific_dates, tool_name=tool_name)
        valid_values = fields_info["key_values"]
        logger.info(f"Analyzing data for {tool_name}: key_figures={key_figures}, key_values={key_values}, start_date={start_date}, end_date={end_date}, specific_dates={specific_dates}")
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
        fetch_result = json.loads(fetch_data(ctx, tool_name, key_values, key_figures, start_date, end_date, specific_dates))
        analysis_notes = [fetch_result.get("message", "")] if fetch_result.get("message") else []
        if fetch_result["status"] == "no_data":
            return json.dumps({
                "status": "no_data",
                "message": fetch_result["message"],
                "period": f"{start_date or 'N/A'} to {end_date or 'N/A'}" if start_date else f"Specific dates: {specific_dates or 'N/A'}",
                "identifier": f"{identifier_field}={identifier_value}" if identifier_field and identifier_value else "all records",
                "metrics_analyzed": normalized_key_figures,
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
        
        sop_content = {}
        for field, value in key_values.items():
            pdf_result = json.loads(get_pdf_content(ctx, tool_name, {field: value}))
            if pdf_result["status"] == "success":
                sop_content[f"{field}={value}"] = pdf_result.get("content", "")
                logger.info(f"SOP content for {field}={value} in {tool_name}: {sop_content[f'{field}={value}'][:100]}...")
            else:
                sop_content[f"{field}={value}"] = ""
                analysis_notes.append(f"Failed to load SOP for {field}={value} in {tool_name}: {pdf_result['message']}")
                logger.warning(f"Failed to load SOP for {field}={value} in {tool_name}: {pdf_result['message']}")

        results = []
        for record in fetch_result["data"]:
            analysis = {
                "date": record.get("date", "Desconocida"),
                **{k: record.get(k) for k in key_values},
                "metrics": {k: record[k] for k in normalized_key_figures if k in record and record[k] is not None}
            }
            results.append(analysis)
        analysis_notes.append(f"Filtered data for {tool_name} with {key_values}")
        period = "all dates"
        if specific_dates:
            period = f"Specific dates: {', '.join(specific_dates)}"
        elif start_date and end_date:
            period = f"{start_date} to {end_date}"
        return json.dumps({
            "status": "success",
            "period": period,
            "identifier": f"{identifier_field}={identifier_value}" if identifier_field and identifier_value else "all records",
            "metrics_analyzed": normalized_key_figures,
            "results": results,
            "sop_content": sop_content,
            "analysis_notes": analysis_notes
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Compliance analysis failed for {tool_name}: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "results": [],
            "analysis_notes": [str(e)]
        }, ensure_ascii=False)

@mcp.tool()
def get_dataset(ctx: Context, tool_name: str, key_values: Optional[Dict[str, str]] = None, key_figures: Optional[List[Dict]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, specific_dates: Optional[List[str]] = None) -> str:
    """
    Recupera datos de un área específica aplicando filtros por campos categóricos, métricas numéricas y fechas.

    INSTRUCCIONES PARA EL LLM:
    1. **Seleccionar el tool_name**: Usa `list_available_tools` para obtener las áreas disponibles y elige el `tool_name` adecuado.
    2. **Obtener campos válidos**: Llama a `list_fields` con el `tool_name` para obtener los campos `key_figures` y `key_values` específicos.
    3. **Validar campos**: Usa solo campos de `list_fields` para el `tool_name` seleccionado.
    4. **Estructura de la consulta**: Sigue esta estructura:
       ```json
       {
           "tool_name": "<nombre_del_area>",
           "key_values": {
               "<campo_categórico_1>": "<valor>",
               "<campo_categórico_2>": "<valor>"
           },
           "key_figures": [
               {"field": "<campo_numérico_1>", "min": <número>, "max": <número>},
               {"field": "<campo_numérico_2>", "min": <número>, "max": <número>}
           ],
           // Usa EITHER specific_dates OR start_date/end_date, no ambos
           "specific_dates": ["YYYY-MM-DD", ...], // Para fechas específicas
           // O
           "start_date": "YYYY-MM-DD", // Para un rango de fechas
           "end_date": "YYYY-MM-DD"
       }
       ```
    5. **Ejemplos**:
       - Para `manufacturing`:
         ```json
         {
             "tool_name": "manufacturing",
             "key_values": {"machine": "ModelA"},
             "key_figures": [{"field": "temperature"}],
             "start_date": "2025-07-18",
             "end_date": "2025-07-20"
         }
         ```
       - Para `human_resources`:
         ```json
         {
             "tool_name": "human_resources",
             "key_values": {"employee_id": "001"},
             "specific_dates": ["2025-07-18"]
         }
         ```
    6. **Manejo de errores**: Si los campos o fechas son inválidos, solicita corrección basada en `list_fields`.
    """
    try:
        if tool_name not in config.get("tools", {}):
            return json.dumps({
                "status": "error",
                "message": f"Tool {tool_name} not configured",
                "data": []
            }, ensure_ascii=False)
        
        fetch_result = json.loads(fetch_data(ctx, tool_name, key_values, key_figures, start_date, end_date, specific_dates))
        if fetch_result["status"] != "success":
            logger.info(f"No data retrieved for {tool_name}: {fetch_result.get('message', 'No message')}")
            return json.dumps([], ensure_ascii=False)
        return json.dumps(fetch_result["data"], ensure_ascii=False)
    except Exception as e:
        logger.error(f"Dataset retrieval failed for {tool_name}: {str(e)}")
        return json.dumps([], ensure_ascii=False)

@mcp.tool()
def list_available_tools(ctx: Context) -> str:
    """
    Lista las áreas (tools) disponibles en el MCP.

    INSTRUCCIONES PARA EL LLM:
    1. **Uso**: Llama a esta función para obtener la lista de áreas disponibles (e.g., "manufacturing", "human_resources").
    2. **Resultado**: Devuelve un JSON con los nombres de las áreas, que deben usarse como `tool_name` en otras funciones.
    3. **Ejemplo de consulta**:
       ```json
       {}
       ```
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
                        tools.append({
                            "name": tool_name,
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
                            tools.append({
                                "name": name,
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