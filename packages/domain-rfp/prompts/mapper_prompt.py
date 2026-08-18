"""Few-shot prompt templates for capability mapping."""

from __future__ import annotations

import json
from typing import Any


DEFAULT_CAPABILITIES: list[dict[str, Any]] = [
    {
        "name": "24/7 Enterprise Support",
        "description": "Round-the-clock support with dedicated escalation paths and 15-minute P1 response.",
    },
    {
        "name": "SOC 2 Type II Certified",
        "description": "Current SOC 2 Type II attestation covering security, availability, and confidentiality.",
    },
    {
        "name": "REST API Integration Platform",
        "description": "Documented REST APIs with OAuth2, webhooks, and SDKs for common languages.",
    },
    {
        "name": "Multi-region Cloud Deployment",
        "description": "Active-active deployment across multiple AWS regions with failover.",
    },
    {
        "name": "AES-256 Data Encryption",
        "description": "Encryption at rest with AES-256 and TLS 1.2+ in transit.",
    },
    {
        "name": "99.9% Uptime SLA",
        "description": "Contractual monthly uptime commitment of 99.9% with service credits.",
    },
    {
        "name": "Dedicated Customer Success Manager",
        "description": "Named CSM for enterprise accounts with quarterly business reviews.",
    },
    {
        "name": "US Data Residency Controls",
        "description": "Customer data residency limited to United States regions upon request.",
    },
]


def get_mapper_prompt() -> str:
    """Return the few-shot system prompt for the Capability Mapper agent."""
    return """You are a capability mapping specialist for enterprise RFP responses.

Map each requirement to the best matching capability from the provided knowledge base.
Assess match_level as "full", "partial", or "none". Assess confidence honestly —
use 0.9+ only for exact matches.

Output ONLY valid JSON with this schema:
{
  "mappings": [
    {
      "requirement_id": "REQ-001",
      "requirement_text": "...",
      "match_level": "full|partial|none",
      "capability": "capability name or null",
      "response_draft": "draft response language",
      "confidence": 0.0,
      "gap_note": "optional gap note or null"
    }
  ],
  "fully_matched": 0,
  "partially_matched": 0,
  "unmatched": 0,
  "overall_confidence": 0.0
}

Few-shot examples:

Example 1 — full match:
Requirement: "Vendor must maintain SOC 2 Type II certification."
Output mapping:
{
  "requirement_id": "REQ-EX1",
  "requirement_text": "Vendor must maintain SOC 2 Type II certification.",
  "match_level": "full",
  "capability": "SOC 2 Type II Certified",
  "response_draft": "We maintain a current SOC 2 Type II attestation covering security, availability, and confidentiality.",
  "confidence": 0.96,
  "gap_note": null
}

Example 2 — partial match:
Requirement: "Provide on-site technical support within 4 hours for Severity-1 incidents."
Output mapping:
{
  "requirement_id": "REQ-EX2",
  "requirement_text": "Provide on-site technical support within 4 hours for Severity-1 incidents.",
  "match_level": "partial",
  "capability": "24/7 Enterprise Support",
  "response_draft": "We provide 24/7 remote enterprise support with 15-minute P1 response and can arrange on-site dispatch where available.",
  "confidence": 0.62,
  "gap_note": "Remote coverage is strong; guaranteed 4-hour on-site arrival is not universally available."
}

Example 3 — unmatched:
Requirement: "Solution must run exclusively on-premises with no cloud components."
Output mapping:
{
  "requirement_id": "REQ-EX3",
  "requirement_text": "Solution must run exclusively on-premises with no cloud components.",
  "match_level": "none",
  "capability": null,
  "response_draft": "Our standard offering is cloud-native; an on-premises-only deployment is not currently supported.",
  "confidence": 0.88,
  "gap_note": "No on-premises-only capability in the knowledge base."
}
"""


def get_mapper_user_prompt(
    requirements: list[dict[str, Any]],
    capabilities: list[dict[str, Any]],
) -> str:
    """Build the user prompt with requirements and capability knowledge base.

    Args:
        requirements: Requirement dicts to map.
        capabilities: Capability knowledge base entries.

    Returns:
        User prompt string for the mapper LLM call.
    """
    return (
        "Map each requirement to the best capability from the knowledge base.\n\n"
        f"CAPABILITIES:\n{json.dumps(capabilities, indent=2)}\n\n"
        f"REQUIREMENTS:\n{json.dumps(requirements, indent=2)}\n\n"
        "Respond with JSON only."
    )
