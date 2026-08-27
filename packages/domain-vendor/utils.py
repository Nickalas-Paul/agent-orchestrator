"""Shared helpers for vendor-agent JSON parsing and confidence scoring."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(content: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from model content.

    Args:
        content: Raw LLM response text.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If no valid JSON object can be parsed.
    """
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON was not an object")
    return data


def clamp_confidence(value: float) -> float:
    """Clamp a confidence value to the inclusive ``[0.0, 1.0]`` range."""
    return max(0.0, min(1.0, float(value)))
