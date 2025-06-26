import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from mcp.server.fastmcp import Context
import os

logger = logging.getLogger(__name__)


class DataValidator:
    DATE_FORMATS = [
        "%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y%m%d",
        "%Y/%m/%d", "%b %d, %Y", "%d %b %Y", "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M", "%Y/%m/%d %H:%M:%S"
    ]

    @staticmethod
    def identify_date_field(data: List[Dict]) -> str:
        """Identifica dinámicamente el campo de fecha en los datos."""
        if not data or not isinstance(data, list) or not data[0]:
            return "date"  # Fallback a "date" si no hay datos
        for record in data[:min(5, len(data))]:  # Revisa hasta 5 registros
            for field, value in record.items():
                if isinstance(value, str) and value.strip():
                    for fmt in DataValidator.DATE_FORMATS:
                        try:
                            datetime.strptime(value.strip(), fmt)
                            return field  # Retorna el primer campo que parece una fecha
                        except ValueError:
                            continue
        return "date"  # Fallback si no se encuentra un campo de fecha válido

    @staticmethod
    def detect_and_normalize_date(data_or_str: str | Dict, field_name: Optional[str] = None) -> Optional[str]:
        """Normaliza una cadena de fecha o busca la fecha en un diccionario de datos."""
        if isinstance(data_or_str, dict):
            if not field_name:
                raise ValueError("field_name is required when passing a dictionary")
            date_str = data_or_str.get(field_name)
        else:
            date_str = data_or_str

        if not isinstance(date_str, str) or not date_str.strip():
            return None
        for fmt in DataValidator.DATE_FORMATS:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
        match = re.match(r"(\d{4}-\d{2}-\d{2})(?:\s+\d{2}:\d{2}(?::\d{2})?)?", date_str)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def validate_date(date_str: str, field: str) -> str:
        """Valida y normaliza una fecha específica."""
        normalized_date = DataValidator.detect_and_normalize_date(date_str)
        if not normalized_date:
            raise ValueError(f"Invalid format for {field}: {date_str}")
        return normalized_date

    @staticmethod
    def check_date_coverage(data: List[Dict], start_date: Optional[str], end_date: Optional[str], specific_dates: Optional[List[str]] = None) -> Dict:
        """Verifica la cobertura de fechas en los datos."""
        if not data:
            message = "No records found."
            if specific_dates:
                message = f"No records for dates: {', '.join(specific_dates)}"
            elif start_date and end_date:
                message = f"No records from {start_date} to {end_date}"
            return {"has_data": False, "covered_dates": [], "message": message}

        # Identificar el campo de fecha dinámicamente
        date_field = DataValidator.identify_date_field(data)
        covered_dates = sorted(set(
            DataValidator.detect_and_normalize_date(r.get(date_field, "")) or "Desconocida"
            for r in data
            if r.get(date_field)
        ))
        covered_dates = [d for d in covered_dates if d != "Desconocida"]

        if specific_dates:
            expected_dates = [DataValidator.validate_date(d, "specific_date") for d in specific_dates]
            missing_dates = [d for d in expected_dates if d not in covered_dates]
            message = f"Found data for {len(covered_dates)} of {len(expected_dates)} dates."
            if missing_dates:
                message += f" Missing: {', '.join(missing_dates)}"
        elif start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            expected_dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((end - start).days + 1)]
            missing_dates = [d for d in expected_dates if d not in covered_dates]
            message = "All dates covered." if not missing_dates else f"Missing: {', '.join(missing_dates)}"
        else:
            message = "Data retrieved successfully."
        return {"has_data": bool(covered_dates), "covered_dates": covered_dates, "message": message}

    @staticmethod
    def validate_fields(ctx: Context, key_figures: List[str], key_values: Dict, start_date: Optional[str] = None, end_date: Optional[str] = None, specific_dates: Optional[List[str]] = None) -> Dict:
        """Valida los campos proporcionados contra los campos disponibles."""
        from main import list_fields  # Importar aquí para evitar circularidad
        fields_info = json.loads(list_fields(ctx))
        if fields_info["status"] != "success":
            raise ValueError("Failed to validate fields")
        if (start_date and not end_date) or (end_date and not start_date):
            raise ValueError("Both start_date and end_date required")
        if start_date and end_date and specific_dates:
            raise ValueError("Cannot use specific_dates with start_date/end_date")
        if start_date and end_date:
            start_date = DataValidator.validate_date(start_date, "start_date")
            end_date = DataValidator.validate_date(end_date, "end_date")
            if start_date > end_date:
                raise ValueError("start_date cannot be after end_date")
        if specific_dates:
            specific_dates = [DataValidator.validate_date(d, f"specific_date[{i}]") for i, d in enumerate(specific_dates)]
        errors = []
        invalid_figures = [f for f in key_figures if f not in fields_info["key_figures"]]
        if invalid_figures:
            errors.append(f"Invalid numeric fields: {invalid_figures}")
        for k, v in key_values.items():
            if k not in fields_info["key_values"] or v not in fields_info["key_values"].get(k, []):
                errors.append(f"Invalid categorical field/value: {k}={v}")
        if errors:
            raise ValueError(" | ".join(errors))
        return fields_info
    
def expand_env_vars(obj):
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_env_vars(i) for i in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj