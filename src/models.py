from pydantic import BaseModel, Field
from typing import Any, Literal

class Prompt(BaseModel):
    """A single natural-language test prompt from function_calling_tests.json."""
    prompt: str

class ParamType(BaseModel):
    """The declared type of a function parameter or return value."""
    type: str # allows number, string, boolean, int, etc... 

class FunctionDefinition(BaseModel):
    """A callable function's name, description, and parameter schema."""
    name: str
    description: str
    parameters: dict[str, ParamType] = Field(default_factory=dict)
    returns: ParamType

class FunctionCallResult(BaseModel):
    """The required output format saved to function_calling_results.json."""
    prompt: str
    name: str
    parameters: dict[str, Any]