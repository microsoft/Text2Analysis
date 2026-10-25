import ast
import json
import math


MAX_CELL_BYTES = 1024 * 1024
MAX_NODES = 20000
MAX_DEPTH = 64


class DataFormatError(ValueError):
    pass


def _check_data(value, field):
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise DataFormatError(f"{field}: data exceeds nesting or item limits")
        if type(item) in (list, tuple):
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is dict:
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise DataFormatError(f"{field}: unsupported data type")
        elif isinstance(item, float) and not math.isfinite(item):
            raise DataFormatError(f"{field}: non-finite numbers are not supported")
        elif isinstance(item, str) and len(item.encode("utf-8")) > MAX_CELL_BYTES:
            raise DataFormatError(f"{field}: text exceeds the size limit")
    return value


def parse_literal(value, field="value"):
    if not isinstance(value, str):
        return _check_data(value, field)
    if len(value.encode("utf-8")) > MAX_CELL_BYTES:
        raise DataFormatError(f"{field}: text exceeds the size limit")
    source = value.strip()
    try:
        result = json.loads(source)
    except (ValueError, RecursionError):
        try:
            tree = ast.parse(source, mode="eval")
            pending = [(tree, 0)]
            count = 0
            allowed = (ast.Expression, ast.Constant, ast.List, ast.Tuple, ast.Dict,
                       ast.UnaryOp, ast.UAdd, ast.USub, ast.Load)
            while pending:
                node, depth = pending.pop()
                count += 1
                if count > MAX_NODES or depth > MAX_DEPTH or not isinstance(node, allowed):
                    raise DataFormatError(f"{field}: only bounded data literals are allowed")
                pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
            result = ast.literal_eval(tree)
        except (ValueError, SyntaxError, TypeError, RecursionError) as error:
            raise DataFormatError(f"{field}: invalid data literal") from error
    return _check_data(result, field)


def is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def parse_list_cell(value, field="value", require_list=False):
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_CELL_BYTES:
            raise DataFormatError(f"{field}: text exceeds the size limit")
        if value.lstrip().startswith("[") or require_list:
            result = parse_literal(value, field)
            if not isinstance(result, list):
                raise DataFormatError(f"{field}: expected a list")
            return result
    elif isinstance(value, list):
        return _check_data(value, field)
    if require_list:
        raise DataFormatError(f"{field}: expected a list")
    return _check_data([value], field)


def parse_example_fields(row):
    source = str(row["python"])
    multiple = source.lstrip().startswith("[")
    python = parse_list_cell(source, "python")
    if not python or not all(isinstance(code, str) for code in python):
        raise DataFormatError("python: expected a nonempty list of code strings")
    results = (parse_list_cell(str(row["python_res"]), "python_res", require_list=True)
               if multiple else _check_data([str(row["python_res"])], "python_res"))
    if len(results) != len(python):
        raise DataFormatError("python_res: length must match python")
    parameters = ([None] if is_missing(row["given_parameter"])
                  else parse_list_cell(row["given_parameter"], "given_parameter"))
    if not all(parameter is None or isinstance(parameter, str) for parameter in parameters):
        raise DataFormatError("given_parameter: expected strings or null")
    result_cell = row.get("code_result")
    code_result = None if is_missing(result_cell) else parse_list_cell(result_cell, "code_result")
    if code_result is not None and len(code_result) != len(python):
        raise DataFormatError("code_result: length must match python")
    return {"python": python, "python_res": results,
            "given_parameter": parameters, "code_result": code_result}