"""Schema validation for run artifacts."""
from __future__ import annotations

from typing import Any, Dict, List


def validate_metrics_schema(metrics: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    _validate_value(metrics, schema, "metrics_json", errors)
    return errors


def _validate_value(value: Any, schema: Any, path: str, errors: List[str]) -> None:
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object")
            return
        for key, subschema in schema.items():
            if key not in value:
                errors.append(f"{path}.{key}: missing")
                continue
            _validate_value(value[key], subschema, f"{path}.{key}", errors)
        return

    if isinstance(schema, list):
        if not isinstance(value, list):
            errors.append(f"{path}: expected list")
            return
        if schema and all(isinstance(item, (str, int, float, bool)) for item in schema):
            if len(value) != len(schema):
                errors.append(f"{path}: expected list length {len(schema)}")
                return
            for idx, (val, expected) in enumerate(zip(value, schema)):
                if isinstance(expected, str) and not isinstance(val, str):
                    errors.append(f"{path}[{idx}]: expected string")
                if isinstance(expected, (int, float)) and not isinstance(val, (int, float)):
                    errors.append(f"{path}[{idx}]: expected number")
            return
        return

    if schema == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string")
        return
    if schema == "integer":
        if not isinstance(value, int):
            errors.append(f"{path}: expected integer")
        return
    if schema == "float":
        if not isinstance(value, (int, float)):
            errors.append(f"{path}: expected float")
        return
    if schema == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean")
        return
    if schema == "path":
        if not isinstance(value, str):
            errors.append(f"{path}: expected path string")
        return
    if schema == "matrix_int":
        if not (isinstance(value, list) and all(isinstance(row, list) for row in value)):
            errors.append(f"{path}: expected matrix")
            return
        for r_idx, row in enumerate(value):
            for c_idx, entry in enumerate(row):
                if not isinstance(entry, int):
                    errors.append(f"{path}[{r_idx}][{c_idx}]: expected integer")
        return
    if schema == "list_string":
        if not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
            errors.append(f"{path}: expected list of strings")
        return
    if schema == "string_or_int_or_null":
        if not (value is None or isinstance(value, (str, int))):
            errors.append(f"{path}: expected string, int, or null")
        return
