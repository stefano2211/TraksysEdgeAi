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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("/app/config.yaml", "r") as f:
    config = yaml.safe_load(f)

mcp = FastMCP("Manufacturing Compliance Processor")

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
    """Recupera datos del sistema MES y los almacena en Qdrant."""
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
            logger.info(f"No key_figures provided, using all numeric fields: {key_figures}")
        else:
            key_figures = key_figures or []

        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values, start_date, end_date, specific_dates)
        logger.info(f"Fetching MES data for key_values={key_values}, key_figures={key_figures}, start_date={start_date}, end_date={end_date}, specific_dates={specific_dates}")
        must_conditions = []
        for k, v in key_values.items():
            must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        if specific_dates:
            normalized_dates = [DataValidator.validate_date(d, f"specific_date[{i}]") for i, d in enumerate(specific_dates)]
            if not normalized_dates:
                return json.dumps({
                    "status": "error",
                    "message": "No se proporcionaron fechas válidas en specific_dates",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)
            must_conditions.append(models.FieldCondition(
                key="date",
                match=models.MatchAny(any=normalized_dates)
            ))
        elif start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            delta = (end - start).days + 1
            if delta > 0:
                date_range = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta)]
                must_conditions.append(models.FieldCondition(
                    key="date",
                    match=models.MatchAny(any=date_range)
                ))
        qdrant_results = qdrant_manager.scroll_data(
            collection_name=config["qdrant"]["collections"][0],
            filter_conditions=models.Filter(must=must_conditions) if must_conditions else None,
            limit=1000
        )
        processed_data = []
        for r in qdrant_results:
            encrypted_payload = r.payload.get("encrypted_payload")
            if encrypted_payload:
                decrypted_data = encryption_manager.decrypt_data(encrypted_payload)
                processed_data.append(decrypted_data)
            else:
                logger.warning(f"No encrypted payload for point {r.id}")
        logger.info(f"Fetched {len(processed_data)} records from Qdrant for {key_values}")
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
                item = {"date": DataValidator.detect_and_normalize_date(record.get("date", "")) or "Desconocida"}
                for field in fields_info["key_figures"] + list(fields_info["key_values"].keys()):
                    if field in record:
                        item[field] = record[field]
                full_data.append(item)
        if full_data and specific_dates:
            normalized_dates = [DataValidator.validate_date(d, f"specific_date[{i}]") for i, d in enumerate(specific_dates)]
            full_data = [r for r in full_data if r.get("date") in normalized_dates]
        elif full_data and start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            delta = (end - start).days + 1
            if delta > 0:
                for i, record in enumerate(full_data):
                    if record["date"] == "Desconocida":
                        record["date"] = (start + timedelta(days=i % delta)).strftime("%Y-%m-%d")
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
            logger.info(f"Stored {len(points)} encrypted points in Qdrant mes_logs")
        if not processed_data:
            processed_data = [
                r for r in full_data
                if all(r.get(k) == v for k, v in key_values.items())
            ]
            logger.info(f"Filtered {len(processed_data)} records in memory for {key_values}")
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
    """Extrae el contenido de un PDF almacenado en MinIO."""
    return minio_client.get_pdf_content(filename)

@mcp.tool()
def list_fields(ctx: Context) -> str:
    """Lista los campos disponibles en el sistema MES y sus tipos de datos."""
    try:
        response = auth_client.get("/machines/")
        response.raise_for_status()
        records = response.json()
        if not records:
            return json.dumps({
                "status": "no_data",
                "message": "No se encontraron registros en el sistema MES",
                "key_figures": [],
                "key_values": {}
            })
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
        return json.dumps({
            "status": "error",
            "message": str(e),
            "key_figures": [],
            "key_values": {}
        }, ensure_ascii=False)

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
    """Agrega una regla personalizada para monitorear el cumplimiento de máquinas."""
    try:
        if isinstance(machines, str):
            try:
                machines = json.loads(machines)
            except json.JSONDecodeError:
                machines = [machines.strip()]
        if isinstance(key_figures, str):
            try:
                parsed_figures = json.loads(key_figures)
                if not isinstance(parsed_figures, dict):
                    raise ValueError("key_figures JSON must be a dictionary")
                key_figures = parsed_figures
            except json.JSONDecodeError:
                parsed_figures = {}
                for pair in key_figures.split(','):
                    if '=' in pair:
                        field, value = pair.split('=', 1)
                    elif ':' in pair:
                        field, value = pair.split(':', 1)
                    else:
                        raise ValueError(f"Invalid format: {pair}")
                    field = field.strip()
                    parsed_figures[field] = float(value.strip())
                key_figures = parsed_figures
        fields_info = json.loads(list_fields(ctx))
        if fields_info["status"] != "success":
            raise ValueError("Could not validate against API")
        valid_machines = fields_info["key_values"].get("machine", [])
        invalid_machines = [m for m in machines if m not in valid_machines]
        if invalid_machines:
            raise ValueError(f"Invalid machines: {invalid_machines}. Valid machines: {valid_machines}")
        invalid_metrics = [f for f in key_figures if f not in fields_info["key_figures"]]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. Valid metrics: {fields_info['key_figures']}")
        valid_operators = [">=", "<=", ">", "<", "==", "!="]
        if operator not in valid_operators:
            raise ValueError(f"Invalid operator. Use one of: {valid_operators}")
        if key_values:
            for k, v in key_values.items():
                if k not in fields_info["key_values"] or v not in fields_info["key_values"].get(k, []):
                    raise ValueError(f"Invalid filter: {k}={v}")
        final_rule = {
            "machines": machines,
            "key_figures": key_figures,
            "key_values": key_values or {},
            "operator": operator,
            "unit": unit,
            "description": description
        }
        embedding_text = description or " ".join(
            [f"{k} {operator} {v}{unit or ''}" for k, v in key_figures.items()]
        )
        point = models.PointStruct(
            id=encryption_manager.generate_id(final_rule),
            vector=qdrant_manager.model.encode(embedding_text).tolist(),
            payload=final_rule
        )
        qdrant_manager.upsert_data(config["qdrant"]["collections"][2], [point])
        metrics_desc = ", ".join(
            [f"{k} {operator} {v}{unit or ''}" for k, v in key_figures.items()]
        )
        filters_desc = ", ".join([f"{k}={v}" for k, v in (key_values or {}).items()])
        message = f"Rule added for {len(machines)} machine(s): {metrics_desc}"
        if filters_desc:
            message += f" | Filters: {filters_desc}"
        return json.dumps({
            "status": "success",
            "message": message,
            "rule": final_rule,
            "details": {
                "machines_count": len(machines),
                "metrics_count": len(key_figures),
                "filters_count": len(key_values or {})
            }
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error adding rule: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "input_parameters": {
                "machines": machines,
                "key_figures": key_figures,
                "key_values": key_values,
                "operator": operator,
                "unit": unit,
                "description": description
            }
        }, ensure_ascii=False)

@mcp.tool()
def list_custom_rules(
    ctx: Context,
    rule_id: Optional[str] = None,
    machine: Optional[str] = None,
    limit: int = 10
) -> str:
    """Lista las reglas personalizadas almacenadas en Qdrant."""
    try:
        filter_conditions = []
        if rule_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="id",
                    match=models.MatchValue(value=rule_id)
                )
            )
        if machine:
            filter_conditions.append(
                models.FieldCondition(
                    key="machines",
                    match=models.MatchAny(any=[machine])
                )
            )
        scroll_filter = models.Filter(must=filter_conditions) if filter_conditions else None
        rules = qdrant_manager.scroll_data(
            config["qdrant"]["collections"][2],
            filter_conditions=scroll_filter,
            limit=limit
        )
        formatted_rules = []
        for rule in rules:
            formatted_rules.append({
                "id": rule.id,
                "machines": rule.payload.get("machines", []),
                "key_figures": rule.payload.get("key_figures", {}),
                "operator": rule.payload.get("operator", ""),
                "unit": rule.payload.get("unit", ""),
                "description": rule.payload.get("description", ""),
                "created_at": rule.payload.get("created_at", ""),
                "applies_to": f"{len(rule.payload.get('machines', []))} machines",
                "metrics": list(rule.payload.get("key_figures", {}).keys())
            })
        return json.dumps({
            "status": "success",
            "count": len(formatted_rules),
            "rules": formatted_rules,
            "metadata": {
                "collection": config["qdrant"]["collections"][2],
                "limit": limit,
                "filters": {
                    "by_id": bool(rule_id),
                    "by_machine": machine if machine else None
                }
            }
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error listing rules: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "rules": []
        }, ensure_ascii=False)

@mcp.tool()
def delete_custom_rule(
    ctx: Context,
    rule_id: str
) -> str:
    """Elimina una regla personalizada por su ID."""
    try:
        existing = qdrant_manager.client.retrieve(
            collection_name=config["qdrant"]["collections"][2],
            ids=[rule_id],
            with_payload=True
        )
        if not existing:
            return json.dumps({
                "status": "error",
                "message": f"Rule with ID {rule_id} not found"
            }, ensure_ascii=False)
        qdrant_manager.client.delete(
            collection_name=config["qdrant"]["collections"][2],
            points_selector=models.PointIdsList(points=[rule_id])
        )
        rule_data = existing[0].payload
        metrics = list(rule_data.get("key_figures", {}).keys())
        machines = rule_data.get("machines", [])
        return json.dumps({
            "status": "success",
            "message": f"Rule deleted: {', '.join(metrics)} for {len(machines)} machine(s)",
            "deleted_rule": {
                "id": rule_id,
                "affected_machines": machines,
                "metrics": metrics,
                "operator": rule_data.get("operator", ""),
                "description": rule_data.get("description", "")
            }
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error deleting rule {rule_id}: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "rule_id": rule_id
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
        logger.info(f"Analyzing compliance: key_figures={key_figures}, key_values={key_values}, start_date={start_date}, end_date={end_date}, specific_dates={specific_dates}")
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
                "custom_rules_applied": 0,
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
                    sop_content[identifier] = pdf_result["content"]
                    logger.info(f"SOP content for {identifier_field}={identifier}: {sop_content[identifier][:100]}...")
                else:
                    sop_content[identifier] = ""
                    analysis_notes.append(f"Failed to load SOP for {identifier_field}={identifier}: {pdf_result['message']}")
                    logger.warning(f"Failed to load SOP for {identifier_field}={identifier}: {pdf_result['message']}")
        else:
            analysis_notes.append("No identifier field or identifiers found; no SOPs loaded.")
        custom_rules = []
        if identifiers and identifier_field:
            custom_result = qdrant_manager.scroll_data(
                config["qdrant"]["collections"][2],
                filter_conditions=models.Filter(must=[
                    models.FieldCondition(key=identifier_field, match=models.MatchAny(any=list(identifiers)))
                ]),
                limit=100
            )
            custom_rules = [r.payload for r in custom_result]
            logger.info(f"Custom rules found: {len(custom_rules)}")
        results = []
        for record in fetch_result["data"]:
            analysis = {
                "date": record.get("date", "Desconocida"),
                **{k: record.get(k) for k in key_values},
                "metrics": {k: record[k] for k in key_figures if k in record},
                "compliance_status": {
                    k: "Compliant" if k == "temperature" and record.get(k, float('inf')) <= 74.0 else "Non-compliant"
                    for k in key_figures if k in record
                }
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
    """Recupera un dataset del sistema MES y lo almacena en Qdrant."""
    try:
        key_values = key_values or {}
        if not key_figures:
            fields_info = json.loads(list_fields(ctx))
            if fields_info["status"] != "success":
                return json.dumps([], ensure_ascii=False)
            key_figures = fields_info["key_figures"]
            logger.info(f"No key_figures provided, using all numeric fields: {key_figures}")
        else:
            key_figures = key_figures or []
        fields_info = DataValidator.validate_fields(ctx, key_figures, key_values, start_date, end_date, specific_dates)
        logger.info(f"Fetching MES dataset for key_values={key_values}, key_figures={key_figures}, start_date={start_date}, end_date={end_date}, specific_dates={specific_dates}")
        must_conditions = []
        for k, v in key_values.items():
            must_conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        if specific_dates:
            normalized_dates = [DataValidator.validate_date(d, f"specific_date[{i}]") for i, d in enumerate(specific_dates)]
            if not normalized_dates:
                return json.dumps([], ensure_ascii=False)
            must_conditions.append(models.FieldCondition(
                key="date",
                match=models.MatchAny(any=normalized_dates)
            ))
        elif start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            delta = (end - start).days + 1
            if delta > 0:
                date_range = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta)]
                must_conditions.append(models.FieldCondition(
                    key="date",
                    match=models.MatchAny(any=date_range)
                ))
        qdrant_results = qdrant_manager.scroll_data(
            config["qdrant"]["collections"][0],
            filter_conditions=models.Filter(must=must_conditions) if must_conditions else None,
            limit=1000
        )
        processed_data = []
        for r in qdrant_results:
            encrypted_payload = r.payload.get("encrypted_payload")
            if encrypted_payload:
                decrypted_data = encryption_manager.decrypt_data(encrypted_payload)
                processed_data.append(decrypted_data)
            else:
                logger.warning(f"No encrypted payload for point {r.id}")
        logger.info(f"Fetched {len(processed_data)} records from Qdrant for {key_values}")
        if not processed_data or not (key_values or key_figures or start_date or end_date or specific_dates):
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
                    item = {"date": DataValidator.detect_and_normalize_date(record.get("date", "")) or "Desconocida"}
                    for field in fields_info["key_figures"] + list(fields_info["key_values"].keys()):
                        if field in record:
                            item[field] = record[field]
                    full_data.append(item)
            if full_data and specific_dates:
                normalized_dates = [DataValidator.validate_date(d, f"specific_date[{i}]") for i, d in enumerate(specific_dates)]
                full_data = [r for r in full_data if r.get("date") in normalized_dates]
            elif full_data and start_date and end_date:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                delta = (end - start).days + 1
                if delta > 0:
                    for i, record in enumerate(full_data):
                        if record["date"] == "Desconocida":
                            record["date"] = (start + timedelta(days=i % delta)).strftime("%Y-%m-%d")
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
                logger.info(f"Stored {len(points)} encrypted points in Qdrant mes_logs")
            processed_data = full_data
        if processed_data:
            processed_data = [
                r for r in processed_data
                if all(r.get(k) == v for k, v in key_values.items())
            ]
            logger.info(f"Filtered {len(processed_data)} records in memory for {key_values}")
        response_fields = ["date"] + list(key_values.keys()) + key_figures
        response_data = [
            {k: r[k] for k in response_fields if k in r}
            for r in processed_data
        ]
        return json.dumps(response_data, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Dataset retrieval failed: {str(e)}")
        return json.dumps([], ensure_ascii=False)

@mcp.tool()
def list_available_tools(ctx: Context) -> str:
    """Lista las herramientas disponibles en el MCP."""
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