import logging
import json
import yaml
import base64
from datetime import datetime
from typing import Optional, List, Dict
from mcp.server.fastmcp import FastMCP, Context
from common.minio_utils import MinioClient
from common.auth_utils import AuthClient
from common.encryption_utils import EncryptionManager
from utils import expand_env_vars, DataValidator
import os
from ollama import Client, ResponseError
import httpx
from PIL import Image as PILImage
from io import BytesIO

# Configurar logging
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

# Cargar configuración
try:
    with open("/app/config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load config.yaml: {str(e)}")
    raise

config = expand_env_vars(config)
logger.info(f"Starting server with API URL: {config['api']['url']}")

# Inicializar clientes
ollama_client = Client(host=os.getenv("OLLAMA_API_BASE_URL"))
encryption_manager = EncryptionManager(os.getenv("ENCRYPTION_KEY"))

# Validar prompts
try:
    PARSE_PROMPTS = {tool: config["tools"][tool]["prompts"] for tool in config.get("tools", {})}
    if not PARSE_PROMPTS:
        logger.error("No prompts defined in config.yaml")
        raise ValueError("prompts is required in config.yaml")
    for tool, tool_prompts in PARSE_PROMPTS.items():
        for func, prompt_config in tool_prompts.items():
            if not isinstance(prompt_config, dict) or "parse_prompt" not in prompt_config or "system_prompt" not in prompt_config:
                logger.error(f"Invalid prompt configuration for {tool}.{func}")
                raise ValueError(f"Invalid prompt configuration for {tool}.{func}")
            if func != "list_fields" and "{user_prompt}" not in prompt_config["parse_prompt"]:
                logger.error(f"Invalid parse_prompt for {tool}.{func}")
                raise ValueError(f"Invalid parse_prompt for {tool}.{func}: missing {{user_prompt}} placeholder")
except Exception as e:
    logger.error(f"Failed to validate prompts configuration: {str(e)}")
    raise

mcp = FastMCP("Manufacturing Image Processor")
tool_name = mcp.name.lower().replace(" ", "-")

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
        sop_prefix=tool_config["minio"]["image_prefix"],
        mes_logs_prefix=tool_config["minio"]["mes_logs_prefix"]
    )

def initialize_buckets():
    for tool_name in config.get("tools", {}):
        minio_client = get_minio_client(tool_name)
        minio_client.ensure_bucket()
        logger.info(f"Initialized bucket for {tool_name}")

initialize_buckets()

def get_tool_client(tool_name: str):
    minio_client = get_minio_client(tool_name)
    return minio_client

def call_llm(prompt: str, system: str = "", model: str = "llama3.1:8b") -> str:
    try:
        logger.debug(f"Sending LLM request with prompt: {prompt[:100]}...")
        response = ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system} if system else {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.get('message', {}).get('content', '')
        if not isinstance(content, str):
            logger.error(f"LLM response content is not a string: {content}")
            raise ValueError(f"Invalid LLM response content: {content}")
        logger.debug(f"LLM response: {content}")
        return content
    except ResponseError as e:
        logger.error(f"Ollama request failed: {str(e)}")
        raise ValueError(f"Ollama request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in LLM call: {str(e)}")
        raise ValueError(f"Unexpected error in LLM call: {str(e)}")

def parse_prompt(tool: str, user_prompt: str, ctx: Context = None, func_name: str = None) -> Dict:
    if tool not in PARSE_PROMPTS:
        logger.error(f"Tool {tool} not supported")
        raise ValueError(f"Tool {tool} not supported for prompt parsing")
    
    tool_prompts = PARSE_PROMPTS[tool]
    if func_name not in tool_prompts:
        logger.error(f"No prompt defined for function {func_name} in tool {tool}")
        raise ValueError(f"No prompt defined for function {func_name} in tool {tool}")
    
    prompt_config = tool_prompts[func_name]
    system_prompt = prompt_config.get("system_prompt", "")
    parse_prompt_template = prompt_config.get("parse_prompt", "")
    defaults = config["tools"][tool].get("defaults", {})
    
    try:
        fields_info = json.loads(list_fields(ctx, tool))
        if fields_info["status"] != "success":
            logger.error(f"Failed to retrieve fields for tool {tool}")
            raise ValueError(f"Failed to retrieve fields for tool {tool}")
        
        fields_info_str = json.dumps({
            "key_figures": fields_info["key_figures"],
            "key_values": fields_info["key_values"]
        }, indent=2)
        
        parse_prompt = parse_prompt_template.format(
            user_prompt=user_prompt,
            default_model_field=defaults.get("model_field", "unknown"),
            default_timestamp_field=defaults.get("timestamp_field", "date"),
            fields_info=fields_info_str
        )
    except KeyError as e:
        logger.error(f"Invalid prompt template for {tool}.{func_name}: missing placeholder {str(e)}")
        raise ValueError(f"Invalid prompt template for {tool}.{func_name}: {str(e)}")
    
    response = call_llm(parse_prompt, system=system_prompt)
    logger.debug(f"Raw LLM response for {tool}.{func_name}: {response}")
    try:
        parsed = json.loads(response)
        if not isinstance(parsed, dict):
            logger.error(f"LLM returned non-dict JSON for {tool}.{func_name}: {parsed}")
            return {}
        if "metric_filters" in parsed:
            for filt in parsed["metric_filters"]:
                if not isinstance(filt, dict) or "field" not in filt:
                    logger.error(f"Invalid metric filter for {tool}.{func_name}: {filt}")
                    return {}
        logger.debug(f"Parsed prompt result: {parsed}")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"LLM did not return valid JSON for {tool}.{func_name}: {response}")
        return {}
    except Exception as e:
        logger.error(f"Error processing LLM response for {tool}.{func_name}: {str(e)}")
        return {}

def get_nested_value(record, key):
    keys = key.replace("[]", "").split(".")
    current = record
    for k in keys:
        if isinstance(current, list) and current:
            current = current[0]
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return None
    return current

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
        api_endpoint = tool_config.get("api_endpoint")
        if not api_endpoint or api_endpoint == "null":
            return json.dumps({
                "status": "error",
                "message": f"api_endpoint not configured for tool {tool_name}",
                "count": 0,
                "data": [],
                "covered_dates": []
            }, ensure_ascii=False)
        
        token = os.getenv("TOKEN")
        if not token:
            logger.error(f"No TOKEN provided for API request in tool {tool_name}")
            return json.dumps({
                "status": "error",
                "message": f"No TOKEN provided for API request in tool {tool_name}",
                "count": 0,
                "data": [],
                "covered_dates": []
            }, ensure_ascii=False)
        
        try:
            auth_client = AuthClient(api_url=config["api"]["url"])
        except Exception as e:
            logger.error(f"Failed to initialize AuthClient for {tool_name}: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Failed to initialize AuthClient: {str(e)}",
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
            logger.debug(f"Raw API response for {tool_name}: {all_data}")
        except httpx.HTTPStatusError as e:
            logger.error(f"API request failed for {tool_name}: {e.response.status_code} {e.response.text}")
            return json.dumps({
                "status": "error",
                "message": f"API request failed: {e.response.text}",
                "count": 0,
                "data": [],
                "covered_dates": []
            }, ensure_ascii=False)
        
        if not all_data:
            return json.dumps({
                "status": "no_data",
                "message": f"No data found for tool {tool_name}",
                "count": 0,
                "data": [],
                "covered_dates": []
            }, ensure_ascii=False)
        
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
        
        fields_info = DataValidator.validate_fields(ctx, normalized_key_figures, key_values, start_date, end_date, specific_dates, tool_name=tool_name)
        
        processed_data = []
        for record in all_data:
            matches = True
            for k, v in key_values.items():
                value = get_nested_value(record, k)
                if isinstance(value, list):
                    matches = matches and any(val in value for val in v)
                else:
                    matches = matches and (value in v if value else False)
                if not matches:
                    break
            
            if matches:
                date_field = DataValidator.identify_date_field([record])
                record_date = DataValidator.detect_and_normalize_date(record, date_field)
                if (not specific_dates and not start_date and not end_date) or \
                   (specific_dates and record_date in specific_dates) or \
                   (start_date and end_date and start_date <= record_date <= end_date):
                    item = {
                        "date": record_date or "Unknown",
                        "image_id": record.get("image_id")
                    }
                    for field in fields_info["key_figures"] + list(fields_info["key_values"].keys()):
                        value = get_nested_value(record, field)
                        if value is not None:
                            item[field] = value
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
        
        logger.debug(f"Processed data after filtering: {processed_data}")
        
        if processed_data:
            processed_data = [
                r for r in processed_data
                if all(get_nested_value(r, k) in v for k, v in key_values.items() if v)
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
        
        response_fields = ["date", "image_id"] + list(key_values.keys()) + normalized_key_figures
        response_data = [
            {k: r[k] for k in response_fields if k in r and r[k] is not None}
            for r in processed_data
        ]
        
        logger.debug(f"Final response data: {response_data}")
        
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
def list_fields(ctx: Context, tool_name: str) -> str:
    """
    List Fields

    Lists the key fields (key_figures and key_values) available in the machine records.

    Parameters:
        ctx (Context): Execution context provided by FastMCP.
        tool_name (str): Name of the tool configured in config.yaml.

    Usage:
        Allows discovering which metrics and filters are available for querying machine data.
    """
    try:
        if tool_name not in config.get("tools", {}):
            return json.dumps({
                "status": "error",
                "message": f"Tool {tool_name} not configured",
                "key_figures": [],
                "key_values": {}
            }, ensure_ascii=False)
        
        tool_config = config["tools"][tool_name]
        auth_client = AuthClient(api_url=config["api"]["url"])
        response = auth_client.get(tool_config.get("api_endpoint"))
        response.raise_for_status()
        all_data = response.json()
        
        if not all_data:
            return json.dumps({
                "status": "no_data",
                "message": f"No records found in the system {tool_name}",
                "key_figures": [],
                "key_values": {}
            }, ensure_ascii=False)
        
        key_figures = []
        key_values = {}
        
        def extract_fields(data, prefix="", all_data_ref=None):
            if isinstance(data, dict):
                for key, value in data.items():
                    full_key = f"{prefix}{key}" if prefix else key
                    if isinstance(value, dict):
                        extract_fields(value, f"{full_key}.", all_data_ref)
                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                        extract_fields(value[0], f"{full_key}[].", all_data_ref)
                    elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        values = set()
                        for rec in all_data_ref:
                            nested_value = get_nested_value(rec, full_key)
                            if isinstance(nested_value, list) and all(isinstance(v, str) for v in nested_value):
                                values.update(nested_value)
                        if values:
                            key_values[full_key] = sorted(values)
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        key_figures.append(full_key)
                    elif isinstance(value, str) and value:
                        values = set()
                        for rec in all_data_ref:
                            nested_value = get_nested_value(rec, full_key)
                            if isinstance(nested_value, str) and nested_value:
                                values.add(nested_value)
                        if values:
                            key_values[full_key] = sorted(values)
        
        for record in all_data:
            extract_fields(record, all_data_ref=all_data)
        
        key_figures = sorted(list(set(key_figures)))
        key_values = {k: sorted(list(set(v))) for k, v in key_values.items() if v}
        
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
def get_images(ctx: Context, tool_name: str, user_prompt: str) -> str:
    """
    Get Images

    Retrieves images from MinIO based on a natural language prompt, filtered by machine data, and returns them as binary data for multimodal model analysis.

    Parameters:
        ctx (Context): Execution context provided by FastMCP.
        tool_name (str): Name of the tool configured in config.yaml.
        user_prompt (str): Natural language query specifying the desired images.

    Usage:
        Retrieves images associated with machine records, filtered by metrics and categorical fields.
        Example: get_images("manufacturing", "Get images for ModelA on 2025-04-10, temperature < 80, defect_type = scratch")
    """
    try:
        if tool_name not in config.get("tools", {}):
            return json.dumps({
                "status": "error",
                "message": f"Tool {tool_name} not configured",
                "data": [],
                "notes": [f"Tool {tool_name} not configured"]
            }, ensure_ascii=False)
        
        minio_client = get_tool_client(tool_name)
        parsed = parse_prompt(tool_name, user_prompt, ctx, func_name="get_images")
        logger.debug(f"Parsed filters: {parsed}")
        
        if not isinstance(parsed, dict):
            logger.error(f"parse_prompt returned non-dict: {parsed}")
            return json.dumps({
                "status": "error",
                "message": "Invalid prompt parsing result",
                "data": [],
                "notes": ["Invalid prompt parsing result"]
            }, ensure_ascii=False)
        
        key_values = parsed.get("model_filters", {})
        key_figures = parsed.get("metric_filters", [])
        start_date = None
        end_date = None
        specific_dates = None
        timestamp_filters = parsed.get("timestamp_filters", {})
        if isinstance(timestamp_filters, dict):
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
        
        metrics_analyzed = [kf["field"] for kf in normalized_key_figures]
        
        try:
            DataValidator.validate_fields(ctx, metrics_analyzed, key_values, start_date, end_date, specific_dates, tool_name=tool_name)
        except ValueError as e:
            logger.error(f"Field validation failed: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Field validation failed: {str(e)}",
                "data": [],
                "notes": [str(e)]
            }, ensure_ascii=False)
        
        fetch_result = json.loads(fetch_data(ctx, tool_name, key_values, normalized_key_figures, start_date, end_date, specific_dates))
        notes = [fetch_result.get("message", "")] if fetch_result.get("message") else []
        if fetch_result["status"] != "success":
            return json.dumps({
                "status": "no_data",
                "message": fetch_result.get("message", "No data found"),
                "data": [],
                "notes": notes
            }, ensure_ascii=False)
        
        results = []
        max_images = 5
        for record in fetch_result["data"][:max_images]:
            image_id = record.get("image_id")
            if not image_id:
                notes.append(f"No image_id found for record: {record}")
                continue
            
            try:
                image_path = f"{record['machine']}_{record['date']}_{image_id}.jpg"
                image_data = minio_client.get_object(image_path)
                img = PILImage.open(BytesIO(image_data))
                img_buffer = BytesIO()
                img.save(img_buffer, format="PNG")
                image_data = img_buffer.getvalue()
                image_format = "png"
                # Convert image_data to base64 for JSON serialization
                image_data_b64 = base64.b64encode(image_data).decode('utf-8')
            except Exception as e:
                notes.append(f"Failed to retrieve or process image {image_id}: {str(e)}")
                continue
            
            result_entry = {
                "date": record.get("date", "Unknown"),
                "machine": record.get("machine"),
                "image_id": image_id,
                "image": {
                    "data": image_data_b64,
                    "format": image_format
                },
                "metrics": {
                    field: record.get(field) for field in metrics_analyzed if field in record and record[field] is not None
                }
            }
            results.append(result_entry)
        
        notes.append(f"Retrieved {len(results)} images for {tool_name} with filters {key_values}")
        period = "all dates"
        if specific_dates:
            period = f"Specific dates: {', '.join(specific_dates)}"
        elif start_date and end_date:
            period = f"{start_date} to {end_date}"
        
        return json.dumps({
            "status": "success" if results else "no_data",
            "period": period,
            "data": results,
            "notes": notes
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Image retrieval failed for {tool_name}: {str(e)}")
        return json.dumps({
            "status": "error",
            "message": str(e),
            "data": [],
            "notes": [str(e)]
        }, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run()