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
from ollama import Client, ResponseError

# Forzar nivel de logging a DEBUG para depuración
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("/app/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load config.yaml: {str(e)}")
    raise

config = expand_env_vars(config)
logger.info(f"API URL: {config['api']['url']}, TOKEN API URL: {config['api']['token_url']}")

# Initialize Ollama client
ollama_client = Client(host='http://ollama:11434')

# Prompts definidos directamente en el código
SYSTEM_PROMPT = "You are an expert prompt parser that responds ONLY with valid JSON. Do not include explanations, markdown, or any extra text. Output must be a single JSON object parseable by json.loads(). Start with { and end with }."

PARSE_PROMPTS = {
    "manufacturing": {
        "prompt": """Parse the user prompt: {user_prompt}. Extract:
- model_filters: dict {{field: [values]}} using field 'machine'. Only include explicitly mentioned models (e.g., ModelA).
- timestamp_filters: dict {{field: {{'from': date, 'to': date}}}} using field 'date', standardizing dates to YYYY-MM-DD.
- metric_filters: list of dict {{'field': str, 'op': '>'|'<'|'=', 'value': num, 'logical': 'AND'|'OR'}}.
Example: 'compliance for ModelA from 2025-04-10 to 2025-04-11, temperature < 80' outputs {{"model_filters": {{"machine": ["ModelA"]}}, "timestamp_filters": {{"date": {{"from": "2025-04-10", "to": "2025-04-11"}}}}, "metric_filters": [{{"field": "temperature", "op": "<", "value": 80, "logical": "AND"}}]}}.""",
        "defaults": {
            "model_field": "machine",
            "timestamp_field": "date"
        }
    },
    "human_resources": {
        "prompt": """Parse the user prompt: {user_prompt}. Extract:
- model_filters: dict {{field: [values]}} using field 'employee_id'. Only include explicitly mentioned IDs (e.g., 001).
- timestamp_filters: dict {{field: {{'from': date, 'to': date}}}} using field 'date', standardizing dates to YYYY-MM-DD.
- metric_filters: list of dict {{'field': str, 'op': '>'|'<'|'=', 'value': num, 'logical': 'AND'|'OR'}}.
Example: 'compliance for employee 001 from 2025-04-10 to 2025-04-11, hours_worked > 8' outputs {{"model_filters": {{"employee_id": ["001"]}}, "timestamp_filters": {{"date": {{"from": "2025-04-10", "to": "2025-04-11"}}}}, "metric_filters": [{{"field": "hours_worked", "op": ">", "value": 8, "logical": "AND"}}]}}.""",
        "defaults": {
            "model_field": "employee_id",
            "timestamp_field": "date"
        }
    }
}

mcp = FastMCP("Multi-Area Compliance Processor")
tool_name = mcp.name.lower().replace(" ", "-")

try:
    auth_client = AuthClient(
        api_url=config["api"]["url"],
        token_api_url=config["api"]["token_url"]
    )
except Exception as e:
    logger.error(f"Failed to initialize AuthClient: {str(e)}")
    raise

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

def call_llm(prompt: str, system: str = "", model: str = "llama3.1:8b") -> str:
    try:
        response = ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system} if system else {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']
    except ResponseError as e:
        logger.error(f"Ollama request failed: {str(e)}")
        raise ValueError(f"Ollama request failed: {str(e)}")

def parse_prompt(tool: str, user_prompt: str) -> Dict:
    if tool not in PARSE_PROMPTS:
        raise ValueError(f"Tool {tool} not supported for prompt parsing")
    
    tool_config = PARSE_PROMPTS[tool]
    parse_prompt_template = tool_config["prompt"]
    defaults = tool_config["defaults"]
    parse_prompt = parse_prompt_template.format(
        user_prompt=user_prompt,
        default_model_field=defaults["model_field"],
        default_timestamp_field=defaults["timestamp_field"]
    )
    response = call_llm(parse_prompt, system=SYSTEM_PROMPT)
    logger.debug(f"LLM parse response for {tool}: {response}")
    try:
        parsed = json.loads(response)
        # Validar estructura básica del JSON
        if not isinstance(parsed, dict):
            raise ValueError("LLM returned invalid JSON structure")
        if "metric_filters" in parsed:
            for filt in parsed["metric_filters"]:
                if not isinstance(filt, dict) or "field" not in filt:
                    raise ValueError(f"Invalid metric filter: {filt}")
        logger.debug(f"Parsed prompt result: {json.dumps(parsed, ensure_ascii=False)}")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"LLM did not return valid JSON: {response}, error: {str(e)}")
        raise ValueError("LLM did not return valid JSON")
    except ValueError as e:
        logger.error(f"JSON validation failed: {str(e)}")
        raise

def fetch_data(ctx: Context, tool_name: str, key_values: Optional[Dict[str, List[str]]] = None, key_figures: Optional[List[Dict]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, specific_dates: Optional[List[str]] = None) -> str:
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
                    "message": "Could not retrieve valid fields",
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
                    "message": f"No data found in bucket {minio_client.bucket}/{minio_client.mes_logs_prefix}",
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
            try:
                response = auth_client.get(api_endpoint, params=params)
                response.raise_for_status()
                all_data = response.json()
                logger.debug(f"API data for {tool_name}: {json.dumps(all_data, ensure_ascii=False)}")
            except Exception as e:
                logger.error(f"API request failed for {tool_name}: {str(e)}")
                return json.dumps({
                    "status": "error",
                    "message": f"API request failed: {str(e)}",
                    "count": 0,
                    "data": [],
                    "covered_dates": []
                }, ensure_ascii=False)

        # Log available categorical fields and values
        available_values = {k: sorted({rec[k] for rec in all_data if k in rec}) for k, v in all_data[0].items() if isinstance(v, str)}
        logger.debug(f"Available categorical fields and values: {available_values}")

        processed_data = []
        for record in all_data:
            # Check if record matches any of the values in key_values
            if all(record.get(k) in v for k, v in key_values.items() if v):
                date_field = DataValidator.identify_date_field([record])
                record_date = DataValidator.detect_and_normalize_date(record, date_field)
                if (not specific_dates and not start_date and not end_date) or \
                   (specific_dates and record_date in specific_dates) or \
                   (start_date and end_date and start_date <= record_date <= end_date):
                    item = {"date": record_date or "Unknown"}
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

        logger.debug(f"Processed data for {tool_name}: {json.dumps(processed_data, ensure_ascii=False)}")
        if processed_data:
            processed_data = [
                r for r in processed_data
                if all(r.get(k) in v for k, v in key_values.items() if v)
            ]
        if normalized_key_figures:
            missing_figures = [k for k in normalized_key_figures if not any(k in r for r in processed_data)]
            if missing_figures:
                return json.dumps({
                    "status": "no_data",
                    "count": 0,
                    "data": [],
                    "message": f"No data found for fields: {', '.join(missing_figures)}",
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
            "covered_dates": []
        }, ensure_ascii=False)

@mcp.tool()
def get_pdf_content(ctx: Context, tool_name: str, key_values: Dict[str, str]) -> str:
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
                "message": f"No records found in the system {tool_name}",
                "key_figures": [],
                "key_values": {}
            }, ensure_ascii=False)
        sample = all_data[0]
        key_figures = [k for k, v in sample.items() if isinstance(v, (int, float))]
        key_values = {k: sorted({rec[k] for rec in all_data if k in rec}) for k, v in sample.items() if isinstance(v, str)}
        logger.debug(f"Fields for {tool_name}: key_figures={key_figures}, key_values={key_values}")
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
def analyze_compliance(ctx: Context, tool_name: str, user_prompt: str) -> str:
    try:
        if tool_name not in config.get("tools", {}):
            return json.dumps({
                "status": "error",
                "message": f"Tool {tool_name} not configured",
                "results": [],
                "analysis_notes": [f"Tool {tool_name} not configured"]
            }, ensure_ascii=False)
        
        minio_client, qdrant_manager = get_tool_client(tool_name)
        parsed = parse_prompt(tool_name, user_prompt)
        logger.debug(f"Parsed prompt result: {json.dumps(parsed, ensure_ascii=False)}")
        
        key_values = parsed.get("model_filters", {})
        key_figures = parsed.get("metric_filters", [])
        start_date = None
        end_date = None
        specific_dates = None
        timestamp_filters = parsed.get("timestamp_filters", {})
        if timestamp_filters:
            for field, rg in timestamp_filters.items():
                start_date = rg.get("from")
                end_date = rg.get("to")
                if not start_date and not end_date:
                    specific_dates = [rg.get("specific_date")] if rg.get("specific_date") else None
        
        normalized_key_figures = []
        if key_figures:
            for item in key_figures:
                if isinstance(item, dict) and "field" in item:
                    normalized_key_figures.append({
                        "field": item["field"],
                        "min": item.get("value") if item.get("op") == ">" else None,
                        "max": item.get("value") if item.get("op") == "<" else None
                    })
                else:
                    logger.warning(f"Invalid metric filter: {item}")
                    continue
        
        # Obtener los campos de metric_filters para metrics_analyzed
        metrics_analyzed = [kf["field"] for kf in normalized_key_figures]
        
        # Validar campos
        try:
            DataValidator.validate_fields(ctx, metrics_analyzed, key_values, start_date, end_date, specific_dates, tool_name=tool_name)
        except ValueError as e:
            logger.error(f"Field validation failed: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Field validation failed: {str(e)}",
                "results": [],
                "analysis_notes": [str(e)]
            }, ensure_ascii=False)
        
        valid_values = json.loads(list_fields(ctx, tool_name))["key_values"]
        logger.info(f"Analyzing data for {tool_name}: key_values={key_values}, key_figures={normalized_key_figures}, start_date={start_date}, end_date={end_date}, specific_dates={specific_dates}")
        identifier_field = None
        identifier_value = None
        if valid_values:
            for field in valid_values.keys():
                if field in key_values:
                    identifier_field = field
                    identifier_value = key_values[field][0] if key_values[field] else None
                    break
            if not identifier_field:
                identifier_field = next(iter(valid_values))
                identifier_value = key_values.get(identifier_field, [None])[0]
        logger.info(f"Selected identifier_field: {identifier_field}, identifier_value: {identifier_value}")
        
        # Obtener datos de la API
        fetch_result = json.loads(fetch_data(ctx, tool_name, key_values, normalized_key_figures, start_date, end_date, specific_dates))
        analysis_notes = [fetch_result.get("message", "")] if fetch_result.get("message") else []
        if fetch_result["status"] == "no_data":
            return json.dumps({
                "status": "no_data",
                "message": fetch_result["message"],
                "period": f"{start_date or 'N/A'} to {end_date or 'N/A'}" if start_date else f"Specific dates: {specific_dates or 'N/A'}",
                "identifier": f"{identifier_field}={identifier_value}" if identifier_field and identifier_value else "all records",
                "metrics_analyzed": metrics_analyzed,
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
        
        # Obtener contenido SOP
        sop_content = {}
        for field, values in key_values.items():
            for value in values:
                pdf_result = json.loads(get_pdf_content(ctx, tool_name, {field: value}))
                if pdf_result["status"] == "success":
                    sop_content[f"{field}={value}"] = pdf_result.get("content", "")
                    logger.info(f"SOP content for {field}={value} in {tool_name}: {sop_content[f'{field}={value}'][:100]}...")
                else:
                    sop_content[f"{field}={value}"] = ""
                    analysis_notes.append(f"Failed to load SOP for {field}={value} in {tool_name}: {pdf_result['message']}")
                    logger.warning(f"Failed to load SOP for {field}={value} in {tool_name}: {pdf_result['message']}")

        # Construir results con datos estructurados
        results = []
        for record in fetch_result["data"]:
            result_entry = {
                "date": record.get("date", "Unknown"),
                **{k: record.get(k) for k in key_values.keys()},
                "metrics": {
                    field: record.get(field) for field in metrics_analyzed if field in record and record[field] is not None
                }
            }
            results.append(result_entry)
        
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
            "metrics_analyzed": metrics_analyzed,
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
def get_dataset(ctx: Context, tool_name: str, key_values: Optional[Dict[str, List[str]]] = None, key_figures: Optional[List[Dict]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, specific_dates: Optional[List[str]] = None) -> str:
    try:
        if tool_name not in config.get("tools", {}):
            return json.dumps({
                "status": "error",
                "message": f"{tool_name} not configured",
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
                        areas = list(config.get("tools", {}).keys())
                        tools.append({
                            "name": tool_name,
                            "parameters": parameters,
                            "Tools available": areas
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
                            areas = list(config.get("tools", {}).keys())
                            tools.append({
                                "name": name,
                                "parameters": parameters,
                                "Tools available": areas
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