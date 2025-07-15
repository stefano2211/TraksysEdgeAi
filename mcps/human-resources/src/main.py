import logging
import json
import yaml
from typing import Optional, List, Dict
from mcp.server.fastmcp import FastMCP, Context
from common.minio_utils import MinioClient
from common.auth_utils import AuthClient
from common.encryption_utils import EncryptionManager
from utils import expand_env_vars, DataValidator
import os
import inspect
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("/app/config.yaml", "r") as f:
    config = yaml.safe_load(f)

config = expand_env_vars(config)
logger.info(f"API URL: {config['api']['url']}, TOKEN API URL: {config['api']['token_url']}")

mcp = FastMCP("Human Resources Compliance Processor")

minio_client = MinioClient(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
)
minio_client.ensure_bucket(config["minio"]["bucket"], config["minio"]["hr_logs_bucket"])

auth_client = AuthClient(
    api_url=config["api"]["url"],
    token_api_url=config["api"]["token_url"]
)

encryption_manager = EncryptionManager(
    os.getenv("ENCRYPTION_KEY")
)

sop_rules_cache = {}  # Formato: {identifier: {"content": str, "timestamp": float}}
CACHE_TTL = config["pdf_extraction"]["cache_time"]  # 3 horas en segundos

def clear_expired_cache():
    """Limpia las entradas de la caché que tienen más de 3 horas."""
    current_time = time.time()
    expired_keys = [
        identifier for identifier, data in sop_rules_cache.items()
        if current_time - data["timestamp"] > CACHE_TTL
    ]
    for identifier in expired_keys:
        del sop_rules_cache[identifier]
        logger.info(f"Caché para identifier={identifier} eliminada (expirada)")

def fetch_hr_data(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[Dict]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_dates: Optional[List[str]] = None
) -> str:
    try:
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
            fields_info = json.loads(list_fields(ctx))
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

        fields_info = DataValidator.validate_fields(ctx, normalized_key_figures, key_values, start_date, end_date, specific_dates)
        processed_data = []
        if config["data_source"]["use_minio_logs"]:
            all_data = minio_client.get_all_json_logs(bucket_name=config["minio"]["hr_logs_bucket"])
            if not all_data:
                return json.dumps({
                    "status": "no_data",
                    "message": f"No se encontraron datos en el bucket {config['minio']['hr_logs_bucket']}",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)
            full_data = []
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
                        full_data.append(item)
            processed_data = full_data
        else:
            params = {}
            if specific_dates:
                params["specific_date"] = specific_dates[0]
            elif start_date and end_date:
                params.update({"start_date": start_date, "end_date": end_date})
            response = auth_client.get("/employees/", params=params)
            response.raise_for_status()
            all_data = response.json()
            full_data = []
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
                        full_data.append(item)
            processed_data = full_data

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
    """Recupera el contenido de un archivo PDF almacenado en MinIO."""
    return minio_client.get_pdf_content(filename, bucket_name=config["minio"]["bucket"])

@mcp.tool()
def list_fields(ctx: Context) -> str:
    """
    Lista los campos disponibles en el dataset HR, incluyendo métricas numéricas y valores categóricos.
    """
    try:
        if config["data_source"]["use_minio_logs"]:
            all_data = minio_client.get_all_json_logs(bucket_name=config["minio"]["hr_logs_bucket"])
        else:
            response = auth_client.get("/employees/")
            response.raise_for_status()
            all_data = response.json()
        if not all_data:
            return json.dumps({
                "status": "no_data",
                "message": "No se encontraron registros en el sistema HR",
                "key_figures": [],
                "key_values": {}
            }, ensure_ascii=False)
        key_figures = set()
        key_values = {}
        for record in all_data:
            for k, v in record.items():
                if isinstance(v, (int, float)):
                    key_figures.add(k)
                elif isinstance(v, str):
                    if k not in key_values:
                        key_values[k] = set()
                    key_values[k].add(v)
        key_figures = list(key_figures)
        key_values = {k: sorted(list(v)) for k, v in key_values.items()}
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
    key_figures: Optional[List[Dict]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_dates: Optional[List[str]] = None
) -> str:
    """
    Analiza el cumplimiento de los datos HR contra políticas de horarios.

    Esta función recupera datos de empleados, los compara con documentos de políticas (PDFs)
    almacenados en MinIO y genera un informe de cumplimiento.

    Args:
        ctx (Context): Contexto de la solicitud proporcionado por el MCP.
        key_values (Optional[Dict[str, str]]): Filtros categóricos (e.g., {"employee_id": "001"}).
        key_figures (Optional[List[Dict]]): Campos numéricos a analizar con rangos opcionales (e.g., [{"field": "hours_worked", "min": 8, "max": 10}]).
        start_date (Optional[str]): Fecha de inicio para el rango de análisis (formato YYYY-MM-DD).
        end_date (Optional[str]): Fecha de fin para el rango de análisis (formato YYYY-MM-DD).
        specific_dates (Optional[List[str]]): Lista de fechas específicas para el análisis (formato YYYY-MM-DD).

    Returns:
        str: Respuesta en formato JSON con el estado, resultados, contenido de políticas y notas.

    INSTRUCCIONES PARA EL LLM:
    1. **Obtener campos válidos**: Antes de construir la consulta, llama a la función `list_fields` para obtener los
       campos disponibles en el dataset HR (key_figures y key_values).
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
               {"field": "<campo_numérico_1>", "min": <número>, "max": <número>},
               {"field": "<campo_numérico_2>", "min": <número>, "max": <número>}
           ],
           // Usa EITHER specific_dates OR start_date/end_date, no ambos
           "specific_dates": ["YYYY-MM-DD", ...], // Para fechas específicas
           // O
           "start_date": "YYYY-MM-DD", // Para un rango de fechas
           "end_date": "YYYY-MM-DD"
       ```
       Nota: `min` y `max` son opcionales. Si no se especifican, se incluyen todos los valores del campo.
    4. **Cuándo usar specific_dates vs. start_date/end_date**:
       - Usa `specific_dates` cuando la consulta menciona días concretos (e.g., "solo el 9 de abril de 2025").
       - Usa `start_date` y `end_date` cuando la consulta menciona un rango de fechas (e.g., "del 9 al 11 de abril de 2025").
       - No combina `specific_dates` con `start_date`/`end_date` en la misma consulta.
       - Si la consulta no especifica fechas, omite ambos parámetros.
    5. **Ejemplo dinámico**:
       Supón que `list_fields` devuelve:
       ```json
       {
           "key_figures": ["hours_worked", "overtime"],
           "key_values": {
               "employee_id": ["001", "002"],
               "role": ["Operator", "Supervisor"],
               "assigned_machine": ["ModelA", "ModelB"],
               "production_line": ["Line1", "Line2", "Line3"]
           }
       }
       ```
       Consultas válidas serían:
       - Para fechas específicas con rangos:
         ```json
         {
             "key_values": {
                 "employee_id": "001",
                 "production_line": "Line3"
             },
             "key_figures": [
                 {"field": "hours_worked", "min": 8, "max": 10},
                 {"field": "overtime"}
             ],
             "specific_dates": ["2025-04-09"]
         }
         ```
       - Para un rango de fechas sin rangos:
         ```json
         {
             "key_values": {
                 "employee_id": "001",
                 "production_line": "Line3"
             },
             "key_figures": [
                 {"field": "hours_worked"},
                 {"field": "overtime"}
             ],
             "start_date": "2025-04-09",
             "end_date": "2025-04-11"
         }
         ```
        - Para un rango de fechas sin rangos:
         ```json
         {
             "key_values": {
                 "employee_id": "001",
                 "production_line": "Line3"
             },
             "key_figures": [
                 {"field": "hours_worked", "min": 8, "max": 10},
                 {"field": "overtime", "min": 3, "max": 5}
             ],
             "start_date": "2025-04-09",
             "end_date": "2025-04-11"
         }
         ```
       - Para un rango de fechas sin key_figures:
         ```json
         {
             "key_values": {
                 "employee_id": "001",
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
       - Si los rangos (`min` o `max`) no son numéricos, devuelve un error solicitando valores válidos.
    """
    try:
        # Limpiar caché expirada
        clear_expired_cache()

        key_values = key_values or {}
        normalized_key_figures = []
        if key_figures:
            for item in key_figures:
                if isinstance(item, str):
                    normalized_key_figures.append(item)
                elif isinstance(item, dict) and "field" in item:
                    normalized_key_figures.append(item["field"])
        if not normalized_key_figures:
            fields_info = json.loads(list_fields(ctx))
            if fields_info["status"] != "success":
                logger.warning("No se pudieron obtener campos válidos, usando key_figures vacíos")
                normalized_key_figures = []
            else:
                normalized_key_figures = fields_info["key_figures"]
                logger.info(f"No key_figures provided, using all numeric fields: {normalized_key_figures}")
        else:
            normalized_key_figures = normalized_key_figures or []

        fields_info = DataValidator.validate_fields(ctx, normalized_key_figures, key_values, start_date, end_date, specific_dates)
        valid_values = fields_info["key_values"]
        logger.info(f"Analyzing data: key_figures={key_figures}, key_values={key_values}, start_date={start_date}, end_date={end_date}, specific_dates={specific_dates}")

        # Determinar los campos identificadores desde config.yaml
        identifier_fields = config["pdf_extraction"].get("identifier_fields", ["employee_id"])
        pdf_name_template = config["pdf_extraction"].get("pdf_name_template", "employee_{identifier}.pdf")

        # Obtener datos HR
        fetch_result = json.loads(fetch_hr_data(ctx, key_values, key_figures, start_date, end_date, specific_dates))
        analysis_notes = [fetch_result.get("message", "")] if fetch_result.get("message") else []
        if fetch_result["status"] == "no_data":
            return json.dumps({
                "status": "no_data",
                "message": fetch_result["message"],
                "period": f"{start_date or 'N/A'} to {end_date or 'N/A'}" if start_date else f"Specific dates: {specific_dates or 'N/A'}",
                "identifier": "all records",
                "metrics_analyzed": normalized_key_figures,
                "results": [],
                "policy_content": {},
                "analysis_notes": analysis_notes
            }, ensure_ascii=False)
        if fetch_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": fetch_result.get("message", "Error retrieving data"),
                "results": [],
                "analysis_notes": analysis_notes
            }, ensure_ascii=False)

        # Construir identifiers desde los datos HR
        identifiers = set()
        for record in fetch_result["data"]:
            identifier_parts = []
            for field in identifier_fields:
                if field in record:
                    identifier_parts.append(record[field])
            if identifier_parts:
                identifier = "_".join(str(part) for part in identifier_parts)
                identifiers.add(identifier)

        # Cargar y cachear el contenido de los PDFs en una sola pasada
        policy_content = {}
        try:
            objects = minio_client.client.list_objects(config["minio"]["bucket"], recursive=True)
            for obj in objects:
                if obj.object_name.endswith(".pdf"):
                    filename = obj.object_name
                    identifier = filename.replace("employee_", "").replace(".pdf", "")
                    if identifier in identifiers or not identifiers:  # Cargar solo PDFs relevantes
                        if identifier in sop_rules_cache and sop_rules_cache[identifier]["timestamp"] + CACHE_TTL > time.time():
                            policy_content[identifier] = sop_rules_cache[identifier]["content"]
                            logger.info(f"Contenido para identifier={identifier} recuperado de la caché")
                        else:
                            pdf_result = json.loads(minio_client.get_pdf_content(filename, bucket_name=config["minio"]["bucket"]))
                            if pdf_result["status"] == "success":
                                sop_rules_cache[identifier] = {
                                    "content": pdf_result["content"],
                                    "timestamp": time.time()
                                }
                                policy_content[identifier] = pdf_result["content"]
                                logger.info(f"Contenido para identifier={identifier} extraído y cacheado")
                            else:
                                policy_content[identifier] = ""
                                analysis_notes.append(f"Failed to load policy for identifier={identifier}: {pdf_result['message']}")
                                logger.warning(f"Failed to load policy for identifier={identifier}: {pdf_result['message']}")
        except Exception as e:
            logger.error(f"Error al listar PDFs en MinIO: {str(e)}")
            analysis_notes.append(f"Error al listar PDFs en MinIO: {str(e)}")

        # Analizar cumplimiento
        results = []
        for record in fetch_result["data"]:
            identifier_parts = []
            for field in identifier_fields:
                if field in record:
                    identifier_parts.append(record[field])
            identifier = "_".join(str(part) for part in identifier_parts) if identifier_parts else None

            analysis = {
                "date": record.get("date", "Desconocida"),
                **{k: record.get(k) for k in key_values},
                "metrics": {k: record[k] for k in normalized_key_figures if k in record and record[k] is not None},
                "policy_content": policy_content.get(identifier, "") if identifier else ""
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
            "identifier": f"{'_'.join(identifier_fields)}={'_'.join(key_values.get(f, '') for f in identifier_fields)}",
            "metrics_analyzed": normalized_key_figures,
            "results": results,
            "policy_content": policy_content,
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
def get_hr_dataset(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[Dict]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_dates: Optional[List[str]] = None
) -> str:
    """
    Recupera datos del sistema HR aplicando filtros por campos categóricos, métricas numéricas y fechas.

    INSTRUCCIONES PARA EL LLM:
    - Antes de construir la consulta, llama a `list_fields` para obtener los campos válidos (`key_figures` y `key_values`).
    - Usa solo campos presentes en la respuesta de `list_fields`.
    - Usa `specific_dates` (lista de fechas YYYY-MM-DD) para días concretos, o `start_date` y `end_date` (YYYY-MM-DD) para rangos. No combina ambos.
    - Para `key_figures`, acepta una lista de diccionarios con campos numéricos y rangos opcionales (e.g., [{"field": "hours_worked", "min": 8, "max": 10}]).
    - Si los campos o fechas no son válidos, devuelve un mensaje de error solicitando corrección.

    Ejemplos de uso:
    1. Fechas específicas con rangos:
       {
           "key_values": {"employee_id": "001"},
           "key_figures": [{"field": "hours_worked", "min": 8, "max": 10}],
           "specific_dates": ["2025-04-09", "2025-04-11"]
       }
    2. Rango de fechas sin rangos:
       {
           "key_values": {"employee_id": "001"},
           "key_figures": [{"field": "hours_worked"}],
           "start_date": "2025-04-09",
           "end_date": "2025-04-11"
       }
    3. Rango de fechas con 2 rangos key_figures:
       {
           "key_values": {"employee_id": "001"},
           "key_figures": [
               {"field": "hours_worked", "min": 8, "max": 10},
               {"field": "overtime", "min": 3, "max": 5}
           ],
           "start_date": "2025-04-09",
           "end_date": "2025-04-11"
       }
    4. Rango de fechas con un rango key_figures y otro sin rangos:
         {           
           "key_values": {"employee_id": "001"},
           "key_figures": [
               {"field": "hours_worked", "min": 8, "max": 10},
               {"field": "overtime"}
           ],
           "start_date": "2025-04-09",
           "end_date": "2025-04-11"
       }
    5. Sin filtros:
       {
           "key_values": {"employee_id": "001"},
           "key_figures": [],
           "start_date": "2025-04-09",
           "end_date": "2025-04-11"
       }

    Args:
        ctx (Context): Contexto FastMCP.
        key_values (Optional[Dict[str, str]]): Filtros categóricos.
        key_figures (Optional[List[Dict]]): Métricas numéricas con rangos opcionales.
        start_date (Optional[str]): Fecha inicio (YYYY-MM-DD).
        end_date (Optional[str]): Fecha fin (YYYY-MM-DD).
        specific_dates (Optional[List[str]]): Lista de fechas específicas (YYYY-MM-DD).

    Returns:
        str: JSON con los datos filtrados.
    """
    try:
        fetch_result = json.loads(fetch_hr_data(ctx, key_values, key_figures, start_date, end_date, specific_dates))
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
    Lista las herramientas disponibles en el MCP.
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