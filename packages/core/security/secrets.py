"""Secrets management abstraction.

Maps to AWS architecture:
- SecretsProvider ABC ≈ the interface your application calls
- EnvironmentSecretsProvider ≈ local dev (reads os.environ / .env)
- In production: swap to AWSSecretsProvider that calls Secrets Manager / KMS

The model/agent code never knows where secrets come from.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from packages.core.logging import get_logger

logger = get_logger("secrets_provider")


class SecretsProvider(ABC):
    """Abstract interface for secrets retrieval.

    Equivalent to the abstract ModelProvider / EmbeddingProvider pattern
    used throughout this codebase — swap the implementation, not the caller.
    """

    @abstractmethod
    def get_secret(self, key: str, default: Any = None) -> Any:
        """Retrieve a secret value by key.

        Args:
            key: The secret identifier (e.g., 'AWS_SECRET_ACCESS_KEY').
            default: Fallback value if the secret is not found.

        Returns:
            The secret value, or default if not found.
        """

    @abstractmethod
    def has_secret(self, key: str) -> bool:
        """Check if a secret exists without retrieving it."""


class EnvironmentSecretsProvider(SecretsProvider):
    """Reads secrets from environment variables.

    This is the local-dev implementation. In production, replace with
    AWSSecretsProvider that calls AWS Secrets Manager:

        # Production example (not implemented here):
        # class AWSSecretsProvider(SecretsProvider):
        #     def __init__(self, region: str = "us-east-1"):
        #         self._client = boto3.client("secretsmanager", region_name=region)
        #
        #     def get_secret(self, key: str, default=None):
        #         try:
        #             response = self._client.get_secret_value(SecretId=key)
        #             return response["SecretString"]
        #         except ClientError:
        #             return default
    """

    def __init__(self, prefix: str = "") -> None:
        """Initialize with optional prefix for namespacing.

        Args:
            prefix: Optional prefix prepended to all key lookups.
                    e.g., prefix="APP_" makes get_secret("DB_PASS") look up "APP_DB_PASS".
        """
        self._prefix = prefix
        logger.info(
            "secrets_provider_initialized",
            provider="environment",
            prefix=prefix or "(none)",
        )

    def get_secret(self, key: str, default: Any = None) -> Any:
        """Read a secret from os.environ."""
        full_key = f"{self._prefix}{key}" if self._prefix else key
        value = os.environ.get(full_key, default)
        if value is None:
            logger.warning("secret_not_found", key=full_key)
        return value

    def has_secret(self, key: str) -> bool:
        """Check if an environment variable exists."""
        full_key = f"{self._prefix}{key}" if self._prefix else key
        return full_key in os.environ
