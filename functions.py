import json
import inspect
from typing import Any, Dict, List


def create_json_schema(function_name, function_description, parameters, required_parameters, hide_details, **kwargs):
    schema = {
        "name": function_name,
        "description": function_description,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": required_parameters
        }
    }

    for param_name, param_details in parameters.items():
        param_schema = {k: v for k, v in param_details.items() if k not in hide_details}
        if param_name in kwargs:
            param_schema.update(kwargs[param_name])
        schema["parameters"]["properties"][param_name] = param_schema

    return json.dumps(schema, indent=4)


def generate_schema_from_function(function: Any, parameter_details: Dict[str, Dict[str, Any]] = None,
                                  hide_details: List[str] = None) -> str:
    if parameter_details is None:
        parameter_details = {}

    if hide_details is None:
        hide_details = []

    # Mapping from Python types to JSON types
    type_mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
        None: "null"
    }

    # Extract the function name
    function_name = function.__name__

    # Extract the function description (docstring)
    function_description = inspect.getdoc(function) or ""

    # Extract the function parameters
    signature = inspect.signature(function)
    parameters = {}
    required_parameters = []
    for param_name, param in signature.parameters.items():
        # Map Python type annotations to JSON data types
        if param.annotation is not inspect.Parameter.empty:
            param_type = type_mapping.get(param.annotation, "object")
        else:
            param_type = "Any"

        param_details = {"type": param_type}

        # Check if the parameter is required (no default value)
        if param.default is not inspect.Parameter.empty:
            param_details["default"] = param.default
        else:
            required_parameters.append(param_name)

        parameters[param_name] = param_details

    return create_json_schema(function_name, function_description, parameters, required_parameters, hide_details,
                              **parameter_details)


# Example usage
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location"""
    # ...


additional_details = {
    "location": {
        "description": "The city and state, e.g. San Francisco, CA"
    },
    "unit": {
        "enum": ["celsius", "fahrenheit"]
    }
}

json_schema = generate_schema_from_function(get_current_weather, additional_details, hide_details=["default"])
print(json_schema)
