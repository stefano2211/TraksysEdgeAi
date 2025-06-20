import logging
import json
import yaml
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
from mcp.server.fastmcp import FastMCP, Context
from common.minio_utils import MinioClient
from common.qdrant_utils import QdrantManager
from common.auth_utils import AuthClient
from common.encryption_utils import EncryptionManager
from data_validators import DataValidator
from qdrant_client.http import models
import os
import inspect

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar configuración
with open("/app/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Inicializar MCP
mcp = FastMCP("Manufacturing Compliance Processor")

# Inicializar clientes
minio_client = MinioClient(
    endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin")
)
minio_client.ensure_bucket(config["minio"]["bucket"])

qdrant_manager = QdrantManager(host="qdrant", port=6333)
for collection in config["qdrant"]["collections"]:
    qdrant_manager.initialize_collection(collection)

auth_client = AuthClient(
    api_url=config["api"]["url"],
    token_api_url=config["api"]["token_url"]
)

encryption_manager = EncryptionManager(
    os.getenv("ENCRYPTION_KEY", "2X3y5z7A9bCdEfGhIjKlMnOpQrStUvWxYz012345678=")
)

@mcp.tool()
def fetch_mes_data(
    ctx: Context,
    key_values: Optional[Dict[str, str]] = None,
    key_figures: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    specific_dates: Optional[List[str]] = None
) -> str:
    """
    Recupera datos MES de la API y Qdrant.
    """
    try:
        key_values = key_values or {}
        fields_info = json.loads(list_fields(ctx))
        if fields_info["status"] != "success":
            return json.dumps({"status": "error", "message": "No se pudieron obtener campos válidos"})
        if not key_figures:
            key_figures = fields_info["key_figures"]
            logger.info(f"Using all numeric fields: {key_figures}")

        DataValidator.validate_fields(ctx, key_figures, key_values, start_date, end_date, specific_dates)

        must_conditions = []
        for k, v in key_values.items():
            must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        if specific_dates:
            normalized_dates = [DataValidator.validate_date(d, f"specific_date[{i}]") for i, d in enumerate(specific_dates)]
            must_conditions.append(models.FieldCondition(key="date", match=models.MatchAny(any=normalized_dates)))
        elif start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            date_range = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]
            must_conditions.append(models.FieldCondition(key="date", match=models.MatchAny(any=date_range)))

        # Recuperar de Qdrant
        qdrant_results = qdrant_manager.scroll_data(
            collection_name=config["qdrant"]["collections"][0],  # mes_logs
            filter_conditions=models.Filter(must=must_conditions) if must_conditions else None
        )
        processed_data = []
        for r in qdrant_results:
            encrypted_payload = r.payload.get("encrypted_payload")
            if encrypted_payload:
                decrypted_data = encryption_manager.decrypt_data(encrypted_payload)
                processed_data.append(decrypted_data)

        # Obtener datos frescos de la API
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
            item = {"date": DataValidator.detect_and_normalize_date(record.get("date", "")) or "Desconocida"}
            for field in fields_info["key_figures"] + list(fields_info["key_values"].keys()):
                if field in record:
                    item[field] = record[field]
            full_data.append(item)

        # Encriptar y almacenar en Qdrant
        if full_data:
            points = []
            for r in full_data:
                encrypted_payload = encryption_manager.encrypt_data(r)
                point = models.PointStruct(
                    id=encryption_manager.generate_id(r),
                    vector=qdrant_manager.model.encode(json.dumps(r)).tolist(),
                    payload={"encrypted_payload": encrypted_payload}
                )
                points.append(point)
            qdrant_manager.upsert_data(config["qdrant"]["collections"][0], points)

        response_data = [
            {k: r[k] for k in ["date"] + list(key_values.keys()) + key_figures if k in r}
            for r in processed_data or full_data
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
        return json.dumps({"status": "error", "message": str(e), "count": 0, "data": []})

@mcp.tool()
def get_pdf_content(ctx: Context, filename: str) -> str:
    """
    Obtiene el contenido de un PDF desde MinIO.
    """
    return minio_client.get_pdf_content(filename)

@mcp.tool()
def list_fields(ctx: Context) -> str:
    """
    Lista los campos disponibles en la API MES.
    """
    try:
        response = auth_client.get("/machines/")
        response.raise_for_status()
        records = response.json()
        if not records:
            return json.dumps({"status": "no_data", "message": "No se encontraron registros"})
        sample = records[0]
        key_figures = [k for k, v in sample.items() if isinstance(v, (int, float))]
        key_values = {k: sorted({rec[k] for rec in records if k in rec}) for k, v in sample.items() if isinstance(v, str)}
        return json.dumps({
            "status": "success",
            "key_figures": key_figures,
            "key_values": key_values
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Field listing failed: {str(e)}")
        return json.dumps({"status": "error", "message": str(e), "key_figures": [], "key_values": {}})

@mcp.tool()
def add_custom_rule(
    ctx: Context,
    machines: Union[List[str], str],
    key_figures: Union[Dict[str, float], str],
    key_values: Optional[Dict[str, str]] = None,
    operator: str = "<=",
    unit: Optional[str] = None,
    description: str = ""
) -> str:
    """
    Añade una regla personalizada a Qdrant.
    """
    try:
        if isinstance(machines, str):
            machines = json.loads(machines) if machines.startswith("[") else [machines.strip()]
        if isinstance(key_figures, str):
            parsed_figures = {}
            for pair in key_figures.split(","):
                field, value = pair.split("=") if "=" in pair else pair.split(":")
                parsed_figures[field.strip()] = float(value.strip())
            key_figures = parsed_figures
        fields_info = json.loads(list_fields(ctx))
        if fields_info["status"] != "success":
            raise ValueError("Could not validate against API")
        invalid_machines = [m for m in machines if m not in fields_info["key_values"].get("machine", [])]
        if invalid_machines:
            raise ValueError(f"Invalid machines: {invalid_machines}")
        invalid_metrics = [f for f in key_figures if f not in fields_info["key_figures"]]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}")
        final_rule = {
            "machines": machines,
            "key_figures": key_figures,
            "key_values": key_values or {},
            "operator": operator,
            "unit": unit,
            "description": description
        }
        embedding_text = description or " ".join([f"{k} {operator} {v}{unit or ''}" for k, v in key_figures.items()])
        point = models.PointStruct(
            id=encryption_manager.generate_id(final_rule),
            vector=qdrant_manager.model.encode(embedding_text).tolist(),
            payload=final_rule
        )
        qdrant_manager.upsert_data(config["qdrant"]["collections"][2], [point])  # custom_rules
        return json.dumps({"status": "success", "message": "Rule added", "rule": final_rule})
    except Exception as e:
        logger.error(f"Error adding rule: {str(e)}")
        return json.dumps({"status": "error", "message": str(e)})

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
    Analiza el cumplimiento de los datos MES contra reglas SOP y personalizadas.

    Esta función recupera datos del sistema MES, los compara con procedimientos operativos estándar (SOPs)
    almacenados en MinIO y reglas personalizadas en Qdrant, y genera un informe de cumplimiento.

    Args:
        ctx (Context): Contexto de la solicitud proporcionado por el MCP.
        key_values (Optional[Dict[str, str]]): Filtros categóricos (e.g., {"machine": "ModelA"}).
        key_figures (Optional[List[str]]): Campos numéricos a analizar (e.g., ["temperature", "uptime"]).
        start_date (Optional[str]): Fecha de inicio para el rango de análisis (formato YYYY-MM-DD).
        end_date (Optional[str]): Fecha de fin para el rango de análisis (formato YYYY-MM-DD).
        specific_dates (Optional[List[str]]): Lista de fechas específicas para el análisis (formato YYYY-MM-DD).

    Returns:
        str: Respuesta en formato JSON con el estado, resultados, contenido de SOPs, reglas aplicadas y notas.

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
       - No combina `specific_dates` con `start_date`/`end_date` en la misma consulta.
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
        # Si no se proporcionan key_figures, obtener todos los campos numéricos desde list_fields
        if not key_figures:
            fields_info = json.loads(list_fields(ctx))
            if fields_info["status"] != "success":
                logger.error("No se pudieron obtener campos válidos")
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
        valid_figures = fields_info["key_figures"]
        logger.info(f"Analyzing compliance: key_figures={key_figures}, key_values={key_values}, start_date={start_date}, end_date={end_date}, specific_dates={specific_dates}")

        # Selección dinámica de identifier_field
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
            logger.warning(fetch_result["message"])
            return json.dumps({
                "status": "no_data",
                "message": fetch_result["message"],
                "period": f"{start_date or 'N/A'} to {end_date or 'N/A'}" if start_date else f"Specific dates: {specific_dates or 'N/A'}",
                "identifier": f"{identifier_field}={identifier_value}" if identifier_field and identifier_value else "all records",
                "metrics_analyzed": key_figures,
                "results": [],
                "sop_content": {},
                "custom_rules_applied": 0,
                "analysis_notes": analysis_notes
            }, ensure_ascii=False)

        if fetch_result["status"] != "success":
            logger.error(fetch_result.get("message", "Error retrieving data"))
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
                    sop_content[identifier] = pdf_result["content"]
                    logger.info(f"SOP content for {identifier_field}={identifier}: {sop_content[identifier][:100]}...")
                else:
                    sop_content[identifier] = ""
                    analysis_notes.append(f"Failed to load SOP for {identifier_field}={identifier}: {pdf_result['message']}")
                    logger.warning(f"Failed to load SOP for {identifier_field}={identifier}: {pdf_result['message']}")
        else:
            analysis_notes.append("No identifier field or identifiers found; no SOPs loaded.")
            logger.info("No identifier field or identifiers found; no SOPs loaded.")

        custom_rules = []
        if identifiers and identifier_field:
            custom_result = qdrant_manager.scroll_data(
                config["qdrant"]["collections"][2],  # custom_rules
                filter_conditions=models.Filter(must=[
                    models.FieldCondition(key=identifier_field, match=models.MatchAny(any=list(identifiers)))
                ])
            )
            custom_rules = [r.payload for r in custom_result] if custom_result else []
            logger.info(f"Custom rules found: {len(custom_rules)}")

        results = []
        for record in fetch_result["data"]:
            analysis = {
                "date": record.get("date", "Desconocida")
            }
            for k in key_values:
                if k in record:
                    analysis[k] = record[k]
            analysis.update({
                "metrics": {k: record[k] for k in key_figures if k in record}
            })
            results.append(analysis)

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
            "custom_rules": custom_rules,
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
    Recupera datos del sistema MES aplicando filtros.
    """
    try:
        key_values = key_values or {}
        fields_info = json.loads(list_fields(ctx))
        if fields_info["status"] != "success":
            return json.dumps([])
        if not key_figures:
            key_figures = fields_info["key_figures"]
        DataValidator.validate_fields(ctx, key_figures, key_values, start_date, end_date, specific_dates)
        fetch_result = json.loads(fetch_mes_data(ctx, key_values, key_figures, start_date, end_date, specific_dates))
        return json.dumps(fetch_result.get("data", []), ensure_ascii=False)
    except Exception as e:
        logger.error(f"Dataset retrieval failed: {str(e)}")
        return json.dumps([])

@mcp.tool()
def list_available_tools(ctx: Context) -> str:
    """
    Lista todas las herramientas disponibles definidas con el decorador @mcp.tool().
    """
    try:
        tools = []
        # Intento 1: Acceder al registro interno de herramientas de FastMCP
        try:
            if hasattr(mcp, 'tools'):
                tool_registry = mcp.tools
                logger.info("Accediendo al registro interno de herramientas de FastMCP")
            elif hasattr(mcp, 'get_tools'):
                tool_registry = mcp.get_tools()
                logger.info("Accediendo a get_tools() de FastMCP")
            else:
                tool_registry = None
                logger.warning("No se encontró registro interno de herramientas en FastMCP")

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
                        logger.debug(f"Tool registrada desde FastMCP: {tool_name}")
        except Exception as e:
            logger.warning(f"Fallo al acceder al registro interno de FastMCP: {str(e)}")

        # Intento 2: Inspección del módulo como respaldo
        if not tools:
            logger.info("Realizando inspección del módulo como respaldo")
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
                            logger.debug(f"Tool detectada por inspección: {name}")
                    except Exception as e:
                        logger.debug(f"No se pudo inspeccionar la función {name}: {str(e)}")

        if not tools:
            logger.warning("No se encontraron herramientas disponibles")

        logger.info(f"Retrieved {len(tools)} available tools")
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