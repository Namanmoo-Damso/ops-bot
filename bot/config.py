"""Configuration and environment validation for bot client"""

import os
from typing import Optional


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid"""
    pass


def validate_env_vars() -> dict[str, str]:
    """
    Validate required environment variables for bot client.

    Returns:
        dict: Validated environment variables

    Raises:
        ConfigError: If any required environment variable is missing
    """
    required_vars = [
        "OPS_API_URL",
        "ADMIN_ACCESS_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]

    missing_vars = []
    config = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            config[var] = value

    if missing_vars:
        error_msg = (
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please set these variables in your .env file or environment."
        )
        raise ConfigError(error_msg)

    # Validate AWS credentials format (basic check)
    aws_key = config["AWS_ACCESS_KEY_ID"]
    if not aws_key.startswith("AKIA"):
        raise ConfigError(
            "AWS_ACCESS_KEY_ID appears invalid (should start with 'AKIA')"
        )

    return config


def get_optional_config() -> dict[str, Optional[str]]:
    """Get optional configuration variables with defaults"""
    return {
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    }
